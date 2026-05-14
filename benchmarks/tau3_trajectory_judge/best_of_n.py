"""Best-of-N downstream selection for tau3 trajectory judges.

This is intentionally a small experiment driver, not a meta-agent benchmark.
It reuses a trajectory pool produced by build_pool.py, samples N trajectories
within the same tau task, asks each judge to pick a winner via pairwise
round-robin comparisons, then scores the selected trajectory by tau's
``gold_reward``.

Example:
    python -m benchmarks.tau3_trajectory_judge.best_of_n \
      --pool benchmarks/tau3_trajectory_judge/data/airline_pool_v1_val.jsonl \
      --task-split judge-val \
      --judge baseline=harnesses/claude_vanilla_tau3_trajectory_judge \
      --judge optimized=/tmp/evo_023_strip_comparison/harness.py \
      --n-values 2,4 \
      --samples-per-task 5 \
      --model claude-haiku-4-5 \
      --out /tmp/tau3_val_best_of_n.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meta_agent.task_runner.judge_runner import (
    JudgePair,
    _run_program_one_ordering,
    _run_one_ordering,
    resolve_harness_path,
)
from benchmarks.plan_rewardbench.adapter import _run_program_score_once
from benchmarks.tau3_trajectory_judge.formatting import flatten_conversation
from meta_agent.core.targets import detect_target


_QUESTION = (
    "Below are two candidate customer-service agent trajectories for the same "
    "tau-airline task. Select the trajectory that better resolves the "
    "customer's issue according to airline policy, factual correctness, and "
    "tool-result evidence."
)

_POINTWISE_QUESTION = (
    "Score exactly one customer-service agent trajectory for a tau-airline "
    "task. The agent converses with a customer and may call airline-database "
    "tools. Higher scores should mean the trajectory is more likely to have "
    "correctly resolved the customer's issue according to airline policy, "
    "factual correctness, and tool-result evidence."
)

_SAMPLING_MODES = ("controlled_mixed", "natural")


def _log(message: str) -> None:
    print(f"[best-of-n] {message}", flush=True)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    actor_model: str
    gold_reward: float
    conversation_text: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    n: int
    task_id: str
    candidates: list[Trajectory]


def _load_pool(
    path: Path,
    task_split: str,
    task_ids: set[str] | None = None,
) -> list[Trajectory]:
    out: list[Trajectory] = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task_split") != task_split:
                continue
            if task_ids is not None and str(rec.get("task_id")) not in task_ids:
                continue
            out.append(
                Trajectory(
                    trajectory_id=str(rec["trajectory_id"]),
                    task_id=str(rec["task_id"]),
                    actor_model=str(rec.get("actor_model", "?")),
                    gold_reward=float(rec.get("gold_reward", 0.0)),
                    conversation_text=flatten_conversation(rec.get("conversation") or []),
                )
            )
    if not out:
        raise ValueError(
            f"No trajectories loaded from {path} with task_split={task_split!r}"
        )
    return out


def _build_episodes(
    trajectories: list[Trajectory],
    *,
    n_values: list[int],
    samples_per_task: int,
    seed: int,
    task_limit: int | None,
    sampling_mode: str = "controlled_mixed",
) -> list[Episode]:
    if sampling_mode not in _SAMPLING_MODES:
        raise ValueError(
            f"sampling_mode must be one of {_SAMPLING_MODES}, got {sampling_mode!r}"
        )

    by_task: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        by_task.setdefault(traj.task_id, []).append(traj)

    task_ids = sorted(by_task)
    if task_limit is not None:
        task_ids = task_ids[:task_limit]

    episodes: list[Episode] = []
    for n in n_values:
        for task_id in task_ids:
            pool = by_task[task_id]
            if len(pool) < n:
                continue
            pass_pool = [t for t in pool if t.gold_reward > 0.0]
            fail_pool = [t for t in pool if t.gold_reward <= 0.0]
            for rep in range(samples_per_task):
                rng = random.Random(f"{seed}::{n}::{task_id}::{rep}")
                # Controlled mode isolates selector quality by ensuring each
                # pool has at least one success and one failure when possible.
                # Natural mode samples pools directly from the rollout
                # distribution, so oracle@N reflects rollout availability.
                if (
                    sampling_mode == "controlled_mixed"
                    and n >= 2
                    and pass_pool
                    and fail_pool
                ):
                    selected = [rng.choice(pass_pool), rng.choice(fail_pool)]
                    remaining = [
                        t for t in pool
                        if t.trajectory_id not in {s.trajectory_id for s in selected}
                    ]
                    selected.extend(rng.sample(remaining, n - 2))
                    rng.shuffle(selected)
                    candidates = selected
                else:
                    candidates = rng.sample(pool, n)
                episodes.append(
                    Episode(
                        episode_id=f"N{n}_task{task_id}_rep{rep:03d}",
                        n=n,
                        task_id=task_id,
                        candidates=candidates,
                    )
                )
    return episodes


async def _score_pointwise_cache(
    *,
    trajectories: list[Trajectory],
    pointwise_judges: dict[str, Path],
    model: str,
    timeout: int,
    concurrency: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score each unique trajectory once per pointwise judge."""

    by_id: dict[str, Trajectory] = {}
    for traj in trajectories:
        by_id.setdefault(traj.trajectory_id, traj)
    unique = sorted(by_id.values(), key=lambda t: (t.task_id, t.trajectory_id))

    cache: dict[str, dict[str, dict[str, Any]]] = {
        judge_name: {} for judge_name in pointwise_judges
    }
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    total = len(unique) * len(pointwise_judges)
    completed = 0

    async def run_one(judge_name: str, harness_path: Path, traj: Trajectory) -> None:
        nonlocal completed
        async with sem:
            started = time.time()
            pair = JudgePair(
                pair_id=f"best_of_n_score_cache_{judge_name}_{traj.trajectory_id}",
                question=_POINTWISE_QUESTION,
                response_a="",
                response_b="",
                gold="A>B",
                source=traj.trajectory_id,
                category=f"task{traj.task_id}",
            )
            try:
                outcome = await _run_program_score_once(
                    harness_path=harness_path,
                    pair=pair,
                    model=model,
                    timeout=timeout,
                    ordering_label="best_of_n_cache",
                    trajectory_label="rollout",
                    trajectory_ref=traj.trajectory_id,
                    trajectory=traj.conversation_text,
                )
                score = outcome.score
                error = outcome.error
                cost_usd = outcome.cost_usd
                num_turns = outcome.num_turns
                wall_time_s = outcome.wall_time_s
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - SDK can raise CancelledError.
                score = None
                error = f"{type(exc).__name__}: {exc}"
                cost_usd = None
                num_turns = None
                wall_time_s = time.time() - started
                _log(
                    f"score cache crashed judge={judge_name} "
                    f"trajectory={traj.trajectory_id} error={error}"
                )
                traceback.print_exception(type(exc), exc, exc.__traceback__)

            record = {
                "trajectory_id": traj.trajectory_id,
                "task_id": traj.task_id,
                "actor_model": traj.actor_model,
                "gold_reward": traj.gold_reward,
                "score": score,
                "error": error,
                "cost_usd": cost_usd,
                "num_turns": num_turns,
                "wall_time_s": wall_time_s,
            }
            async with lock:
                cache[judge_name][traj.trajectory_id] = record
                completed += 1
                _log(
                    f"pointwise cache [{completed}/{total}] judge={judge_name} "
                    f"task={traj.task_id} reward={traj.gold_reward:.0f} "
                    f"score={score} error={error or 'none'} "
                    f"dur={wall_time_s:.1f}s"
                )

    await asyncio.gather(
        *[
            run_one(judge_name, harness_path, traj)
            for judge_name, harness_path in pointwise_judges.items()
            for traj in unique
        ]
    )
    return cache


def _summarize_score_cache(
    score_cache: dict[str, dict[str, dict[str, Any]]] | None,
) -> dict[str, Any] | None:
    if score_cache is None:
        return None
    summary: dict[str, Any] = {}
    for judge_name, rows_by_id in score_cache.items():
        rows = list(rows_by_id.values())
        valid = [r for r in rows if r.get("score") is not None and not r.get("error")]
        total_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
        summary[judge_name] = {
            "n_trajectories": len(rows),
            "valid_score_rate": len(valid) / len(rows) if rows else None,
            "total_cost_usd": total_cost,
            "avg_cost_usd": total_cost / len(rows) if rows else None,
        }
    return summary


async def _select_with_judge(
    *,
    harness_path: Path,
    judge_name: str,
    episode: Episode,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    votes = [0 for _ in episode.candidates]
    pair_results: list[dict[str, Any]] = []
    start = time.time()
    n_pairs = len(episode.candidates) * (len(episode.candidates) - 1) // 2
    target_name = detect_target(harness_path).name

    _log(
        f"start episode={episode.episode_id} judge={judge_name} "
        f"N={episode.n} task={episode.task_id} pairs={n_pairs} target={target_name}"
    )

    pair_num = 0
    for i in range(len(episode.candidates)):
        for j in range(i + 1, len(episode.candidates)):
            pair_num += 1
            a = episode.candidates[i]
            b = episode.candidates[j]
            _log(
                f"pair {pair_num}/{n_pairs} episode={episode.episode_id} "
                f"judge={judge_name} compare={i}v{j} "
                f"a_reward={a.gold_reward:.0f} b_reward={b.gold_reward:.0f}"
            )
            pair = JudgePair(
                pair_id=f"{episode.episode_id}_{judge_name}_{i}_vs_{j}",
                question=_QUESTION,
                response_a=a.conversation_text,
                response_b=b.conversation_text,
                gold="A>B",
                source=f"{a.trajectory_id}|{b.trajectory_id}",
                category=f"task{episode.task_id}",
            )
            started = time.time()
            try:
                if target_name == "program_harness":
                    outcome = await _run_program_one_ordering(
                        harness_path=harness_path,
                        label="best_of_n",
                        pair=pair,
                        flip_responses=False,
                        model=model,
                        timeout=timeout,
                    )
                else:
                    outcome = await _run_one_ordering(
                        harness_path=harness_path,
                        label="best_of_n",
                        pair=pair,
                        flip_responses=False,
                        model=model,
                        timeout=timeout,
                    )
                decision = outcome.decision_final
                error = outcome.error
                cost_usd = outcome.cost_usd
                num_turns = outcome.num_turns
                wall_time_s = outcome.wall_time_s
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - SDK can raise CancelledError.
                decision = None
                error = f"{type(exc).__name__}: {exc}"
                cost_usd = None
                num_turns = None
                wall_time_s = time.time() - started
                _log(
                    f"pair crashed episode={episode.episode_id} "
                    f"judge={judge_name} error={error}"
                )
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            winner_idx: int | None = None
            if error is None and decision == "A>B":
                winner_idx = i
                votes[i] += 1
            elif error is None and decision == "B>A":
                winner_idx = j
                votes[j] += 1

            _log(
                f"pair done episode={episode.episode_id} judge={judge_name} "
                f"decision={decision} winner={winner_idx} "
                f"error={error or 'none'} "
                f"dur={wall_time_s:.1f}s"
            )

            pair_results.append(
                {
                    "i": i,
                    "j": j,
                    "a": a.trajectory_id,
                    "b": b.trajectory_id,
                    "decision": decision,
                    "winner_idx": winner_idx,
                    "error": error,
                    "cost_usd": cost_usd,
                    "num_turns": num_turns,
                    "wall_time_s": wall_time_s,
                }
            )

    valid_selection = any(votes)
    # Deterministic tie-break: earlier sampled trajectory wins, but only after
    # at least one pair produced a usable verdict.
    selected_idx = (
        max(range(len(votes)), key=lambda idx: (votes[idx], -idx))
        if valid_selection
        else None
    )
    selected = episode.candidates[selected_idx] if selected_idx is not None else None
    _log(
        f"episode done episode={episode.episode_id} judge={judge_name} "
        f"selected={selected_idx} "
        f"selected_reward={(selected.gold_reward if selected else -1):.0f} "
        f"votes={votes} dur={time.time() - start:.1f}s"
    )
    return {
        "judge": judge_name,
        "selected_idx": selected_idx,
        "selected_trajectory_id": selected.trajectory_id if selected else None,
        "selected_actor_model": selected.actor_model if selected else None,
        "selected_gold_reward": selected.gold_reward if selected else None,
        "votes": votes,
        "valid_selection": valid_selection,
        "success": bool(selected and selected.gold_reward > 0.0),
        "cost_usd": sum((p.get("cost_usd") or 0.0) for p in pair_results),
        "wall_time_s": time.time() - start,
        "pairs": pair_results,
    }


async def _select_with_pointwise_judge(
    *,
    harness_path: Path,
    judge_name: str,
    episode: Episode,
    model: str,
    timeout: int,
    score_cache: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    start = time.time()
    score_results: list[dict[str, Any]] = []

    _log(
        f"start episode={episode.episode_id} judge={judge_name} "
        f"N={episode.n} task={episode.task_id} pointwise_scores={len(episode.candidates)}"
    )

    pair = JudgePair(
        pair_id=f"{episode.episode_id}_{judge_name}_pointwise",
        question=_POINTWISE_QUESTION,
        response_a="",
        response_b="",
        gold="A>B",
        source="|".join(t.trajectory_id for t in episode.candidates),
        category=f"task{episode.task_id}",
    )
    scores: list[float | None] = []
    for idx, traj in enumerate(episode.candidates):
        started = time.time()
        cached_record = (
            score_cache.get(judge_name, {}).get(traj.trajectory_id)
            if score_cache is not None
            else None
        )
        if cached_record is not None:
            score = cached_record.get("score")
            error = cached_record.get("error")
            cost_usd = 0.0
            cache_cost_usd = cached_record.get("cost_usd")
            num_turns = cached_record.get("num_turns")
            wall_time_s = 0.0
        elif score_cache is not None:
            score = None
            error = f"missing cached score for {traj.trajectory_id}"
            cost_usd = 0.0
            cache_cost_usd = None
            num_turns = None
            wall_time_s = 0.0
        else:
            cache_cost_usd = None
            try:
                outcome = await _run_program_score_once(
                    harness_path=harness_path,
                    pair=pair,
                    model=model,
                    timeout=timeout,
                    ordering_label="best_of_n",
                    trajectory_label=f"candidate_{idx}",
                    trajectory_ref=traj.trajectory_id,
                    trajectory=traj.conversation_text,
                )
                score = outcome.score
                error = outcome.error
                cost_usd = outcome.cost_usd
                num_turns = outcome.num_turns
                wall_time_s = outcome.wall_time_s
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - SDK can raise CancelledError.
                score = None
                error = f"{type(exc).__name__}: {exc}"
                cost_usd = None
                num_turns = None
                wall_time_s = time.time() - started
                _log(
                    f"score crashed episode={episode.episode_id} "
                    f"judge={judge_name} candidate={idx} error={error}"
                )
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        if score is not None:
            score = float(score)

        scores.append(score)
        _log(
            f"score done episode={episode.episode_id} judge={judge_name} "
            f"candidate={idx} score={score} "
            f"gold_reward={traj.gold_reward:.0f} error={error or 'none'} "
            f"dur={wall_time_s:.1f}s"
        )
        score_results.append(
            {
                "i": idx,
                "trajectory_id": traj.trajectory_id,
                "score": score,
                "error": error,
                "cost_usd": cost_usd,
                "cached": cached_record is not None,
                "cache_cost_usd": cache_cost_usd,
                "num_turns": num_turns,
                "wall_time_s": wall_time_s,
            }
        )

    valid_selection = any(score is not None for score in scores)
    selected_idx = (
        max(
            range(len(scores)),
            key=lambda idx: (
                float("-inf") if scores[idx] is None else float(scores[idx]),
                -idx,
            ),
        )
        if valid_selection
        else None
    )
    selected = episode.candidates[selected_idx] if selected_idx is not None else None
    _log(
        f"episode done episode={episode.episode_id} judge={judge_name} "
        f"selected={selected_idx} "
        f"selected_reward={(selected.gold_reward if selected else -1):.0f} "
        f"scores={scores} dur={time.time() - start:.1f}s"
    )
    return {
        "judge": judge_name,
        "judge_mode": "pointwise_score",
        "selected_idx": selected_idx,
        "selected_trajectory_id": selected.trajectory_id if selected else None,
        "selected_actor_model": selected.actor_model if selected else None,
        "selected_gold_reward": selected.gold_reward if selected else None,
        "scores": scores,
        "valid_selection": valid_selection,
        "success": bool(selected and selected.gold_reward > 0.0),
        "cost_usd": sum((r.get("cost_usd") or 0.0) for r in score_results),
        "wall_time_s": time.time() - start,
        "score_runs": score_results,
    }


async def _run_all(
    *,
    episodes: list[Episode],
    judges: dict[str, Path],
    pointwise_judges: dict[str, Path],
    model: str,
    timeout: int,
    concurrency: int,
    pointwise_score_cache: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    total = len(episodes) * (len(judges) + len(pointwise_judges))
    completed = 0
    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def run_one(
        episode: Episode,
        judge_name: str,
        harness_path: Path,
        *,
        pointwise: bool,
    ) -> None:
        nonlocal completed
        async with sem:
            judge_result = await (
                _select_with_pointwise_judge(
                    harness_path=harness_path,
                    judge_name=judge_name,
                    episode=episode,
                    model=model,
                    timeout=timeout,
                    score_cache=pointwise_score_cache,
                )
                if pointwise
                else _select_with_judge(
                    harness_path=harness_path,
                    judge_name=judge_name,
                    episode=episode,
                    model=model,
                    timeout=timeout,
                )
            )
            oracle_success = any(t.gold_reward > 0.0 for t in episode.candidates)
            random_success = sum(t.gold_reward for t in episode.candidates) / len(
                episode.candidates
            )
            row = {
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "n": episode.n,
                "candidate_ids": [t.trajectory_id for t in episode.candidates],
                "candidate_actor_models": [t.actor_model for t in episode.candidates],
                "candidate_gold_rewards": [t.gold_reward for t in episode.candidates],
                "oracle_success": oracle_success,
                "random_expected_success": random_success,
                **judge_result,
            }
            async with lock:
                completed += 1
                results.append(row)
                mark = "PASS" if row["success"] else "FAIL"
                selected_reward = row["selected_gold_reward"]
                selected_reward_s = (
                    "none" if selected_reward is None else f"{selected_reward:.0f}"
                )
                print(
                    f"[{completed:>4}/{total}] {mark} N={episode.n} "
                    f"task={episode.task_id} judge={judge_name} "
                    f"selected_reward={selected_reward_s}",
                    flush=True,
                )

    await asyncio.gather(
        *[
            run_one(episode, judge_name, harness_path, pointwise=False)
            for episode in episodes
            for judge_name, harness_path in judges.items()
        ],
        *[
            run_one(episode, judge_name, harness_path, pointwise=True)
            for episode in episodes
            for judge_name, harness_path in pointwise_judges.items()
        ],
    )
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    n_values = sorted({int(r["n"]) for r in results})
    judges = sorted({str(r["judge"]) for r in results})
    for n in n_values:
        n_rows = [r for r in results if int(r["n"]) == n]
        summary[str(n)] = {
            "n_episodes_per_judge": len(n_rows) // max(len(judges), 1),
            "oracle_upper_bound": sum(r["oracle_success"] for r in n_rows)
            / len(n_rows),
            "random_expected_success": sum(
                float(r["random_expected_success"]) for r in n_rows
            )
            / len(n_rows),
            "judges": {},
        }
        for judge in judges:
            rows = [r for r in n_rows if r["judge"] == judge]
            valid_rows = [r for r in rows if r.get("valid_selection")]
            summary[str(n)]["judges"][judge] = {
                "success_rate": sum(r["success"] for r in rows) / len(rows),
                "success_rate_valid_only": (
                    sum(r["success"] for r in valid_rows) / len(valid_rows)
                    if valid_rows else None
                ),
                "valid_selection_rate": len(valid_rows) / len(rows),
                "avg_cost_usd": sum(float(r["cost_usd"]) for r in rows) / len(rows),
                "n": len(rows),
            }
    return summary


def _parse_judge(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("judge must be NAME=CONFIG_PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("judge name cannot be empty")
    return name, resolve_harness_path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tau3 validation best-of-N.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--task-split", default="judge-val")
    parser.add_argument("--judge", action="append", type=_parse_judge, default=[])
    parser.add_argument(
        "--pointwise-judge",
        action="append",
        type=_parse_judge,
        default=[],
        help=(
            "NAME=CONFIG_PATH for a program-harness pointwise scalar scorer. "
            "Each rollout is scored once and the max score is selected."
        ),
    )
    parser.add_argument("--n-values", default="2,4")
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=_SAMPLING_MODES,
        default="controlled_mixed",
        help=(
            "controlled_mixed forces at least one pass and one fail when possible; "
            "natural samples pools directly from the rollout distribution."
        ),
    )
    parser.add_argument(
        "--cache-pointwise-scores",
        action="store_true",
        help=(
            "For pointwise scalar judges, score each unique trajectory once and "
            "reuse those scores across all Best-of-N episodes."
        ),
    )
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--task-ids",
        default=None,
        help=(
            "Comma-separated task IDs to include after task_split filtering. "
            "Needed for v2 splits, where train/val/test share task_split=judge-v2-all."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]
    judges = dict(args.judge)
    pointwise_judges = dict(args.pointwise_judge)
    if not judges and not pointwise_judges:
        raise SystemExit("At least one --judge or --pointwise-judge is required.")
    task_ids = (
        {x.strip() for x in args.task_ids.split(",") if x.strip()}
        if args.task_ids
        else None
    )
    trajectories = _load_pool(Path(args.pool), args.task_split, task_ids)
    episodes = _build_episodes(
        trajectories,
        n_values=n_values,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        task_limit=args.task_limit,
        sampling_mode=args.sampling_mode,
    )
    if not episodes:
        raise SystemExit("No best-of-N episodes could be built.")

    _log(
        f"Loaded {len(trajectories)} trajectories; built {len(episodes)} episodes; "
        f"sampling_mode={args.sampling_mode} "
        f"judges={list(judges)} pointwise_judges={list(pointwise_judges)}"
    )
    sys.stdout.flush()
    pointwise_score_cache = None
    if args.cache_pointwise_scores and pointwise_judges:
        _log(
            f"Caching pointwise scores for {len(trajectories)} trajectories "
            f"and {len(pointwise_judges)} judge(s)..."
        )
        pointwise_score_cache = asyncio.run(
            _score_pointwise_cache(
                trajectories=trajectories,
                pointwise_judges=pointwise_judges,
                model=args.model,
                timeout=args.timeout,
                concurrency=args.concurrency,
            )
        )

    results = asyncio.run(
        _run_all(
            episodes=episodes,
            judges=judges,
            pointwise_judges=pointwise_judges,
            model=args.model,
            timeout=args.timeout,
            concurrency=args.concurrency,
            pointwise_score_cache=pointwise_score_cache,
        )
    )
    payload = {
        "pool": args.pool,
        "task_split": args.task_split,
        "task_ids": sorted(task_ids) if task_ids is not None else None,
        "n_values": n_values,
        "samples_per_task": args.samples_per_task,
        "sampling_mode": args.sampling_mode,
        "cache_pointwise_scores": bool(args.cache_pointwise_scores),
        "model": args.model,
        "judges": {name: str(path) for name, path in judges.items()},
        "pointwise_judges": {
            name: str(path) for name, path in pointwise_judges.items()
        },
        "pointwise_score_cache_summary": _summarize_score_cache(
            pointwise_score_cache
        ),
        "pointwise_score_cache": pointwise_score_cache,
        "summary": _summarize(results),
        "episodes": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
