"""Multi-actor trajectory pool builder for the τ³-airline task-success judge.

Phase-1 Day-4 deliverable (see docs/internal/EXPERIMENT_A_TASK_SUCCESS_PLAN.md
§9 Phase 2). Generates a JSONL pool of τ²-labeled airline rollouts from a
mix of Anthropic + OpenAI actors, bypassing the meta-agent tau3 adapter by
calling `tau2.runner.run_tasks` directly.

Design:
    - One `TextRunConfig` per actor (different `llm_agent` values).
    - Results stream to JSONL as each actor finishes (crash-safe partial output).
    - Manifest sidecar records actors, task_ids, pass rates, costs, timestamps.
    - CLI + importable: `generate_pool(...)` is the importable entry used by
      the perturbation module (Phase 1 Day 2 hard negatives) and any future
      multi-domain extension.

Each record in the output JSONL matches the schema in EXPERIMENT_A §5::

    {
      "trajectory_id": "claude-haiku-4-5_airline_12_trial0",
      "task_id": "12",
      "task_split": "judge-train" | "judge-val",
      "actor_model":      "claude-haiku-4-5",              # short name
      "actor_model_full": "bedrock/global.anthropic....",  # LiteLLM id
      "seed": 42,
      "trial": 0,
      "conversation": [Message.model_dump() for m in sim.get_messages()],
      "outcome_label": "pass" | "fail",    # derived from τ² reward_info.reward > 0
      "gold_reward": 1.0 | 0.0,
      "duration_s": 27.8,
      "agent_cost": 0.048,
      "user_cost": 0.003,
      "termination_reason": "agent_stop",
      "num_messages": 16
    }

Required env vars::

    TAU2_DATA_DIR               # path to a tau2-bench repo clone's data/ dir
    AWS_BEARER_TOKEN_BEDROCK    # for Bedrock-routed Anthropic actors
    AWS_REGION                  # e.g. us-east-1
    OPENAI_API_KEY              # for OpenAI actors (default user-sim + gpt-* actors)

Usage::

    # Mini pool (smoke dev — 3 tasks × 2 actors × 1 seed = 6 rollouts, ~$1)
    TAU2_DATA_DIR=../tau2-bench/data \\
        python -m benchmarks.tau3_trajectory_judge.build_pool \\
        --out benchmarks/tau3_trajectory_judge/data/airline_pool_smoke.jsonl

    # Full judge-train pool (28 tasks × 4 actors × 2 seeds = 224 rollouts)
    TAU2_DATA_DIR=../tau2-bench/data \\
        python -m benchmarks.tau3_trajectory_judge.build_pool \\
        --out benchmarks/tau3_trajectory_judge/data/airline_pool_v1_train.jsonl \\
        --task-ids 0,1,3,4,5,7,8,9,12,13,14,15,17,18,19,21,24,25,26,27,29,30,31,32,33,35,36,37 \\
        --actors claude-haiku-4-5,claude-sonnet-4-6,gpt-5.4-mini,gpt-4o-mini \\
        --seeds 2 \\
        --task-split judge-train \\
        --max-concurrency 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Short-name → LiteLLM-qualified model id mapping. Mirrors
# `meta_agent.services.llm.BEDROCK_MODEL_MAP` for Anthropic (Bedrock cross-region
# inference profiles) and passes OpenAI names through unchanged.
ACTOR_MODEL_MAP: dict[str, str] = {
    "claude-haiku-4-5":  "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6": "bedrock/global.anthropic.claude-sonnet-4-6",
    "gpt-5.4-mini":      "gpt-5.4-mini",
    "gpt-4o-mini":       "gpt-4o-mini",
}


def _resolve_actor(name: str) -> str:
    """Expand short name to LiteLLM id; pass-through on unknown names."""
    return ACTOR_MODEL_MAP.get(name, name)


def _short_name(resolved: str) -> str:
    """Reverse lookup: LiteLLM id → short name. Fallback: sanitized id."""
    for short, full in ACTOR_MODEL_MAP.items():
        if full == resolved:
            return short
    return resolved.replace("/", "_").replace(":", "_").replace(".", "_")


def _parse_float_csv(value: Optional[str]) -> Optional[list[float]]:
    """Parse a comma-separated float list used by domain-transfer pool builders."""
    if not value:
        return None
    out: list[float] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        out.append(float(stripped))
    return out or None


def _temperature_slug(value: Optional[float]) -> str:
    """Stable filename/id fragment for a sampling temperature."""
    if value is None:
        return "default"
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass
class RolloutRecord:
    trajectory_id: str
    task_id: str
    task_split: str
    actor_model: str            # short name (e.g. "claude-haiku-4-5")
    actor_model_full: str       # LiteLLM id
    seed: Optional[int]
    trial: Optional[int]
    conversation: list[dict[str, Any]]
    outcome_label: str          # "pass" | "fail"
    gold_reward: float
    duration_s: float
    agent_cost: Optional[float]
    user_cost: Optional[float]
    termination_reason: str
    num_messages: int


@dataclass
class ActorStats:
    actor: str
    n_rollouts: int = 0
    n_pass: int = 0
    n_skipped: int = 0
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        return self.n_pass / self.n_rollouts if self.n_rollouts else 0.0


@dataclass
class PoolManifest:
    pool_path: str
    actors_short: list[str]
    actors_full: list[str]
    task_ids: list[str]
    task_split: str
    seeds_per_task: int
    user_model: str
    max_concurrency: int
    started_at: str
    ended_at: str
    total_cost_usd: float
    total_duration_s: float
    total_rollouts: int
    total_pass: int
    pool_pass_rate: float
    per_actor_stats: list[dict[str, Any]] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _term_reason_str(tr: Any) -> str:
    if tr is None:
        return ""
    return str(tr.value) if hasattr(tr, "value") else str(tr)


def _run_one_actor(
    *,
    actor_short: str,
    actor_full: str,
    tau2_tasks: list[Any],
    seeds_per_task: int,
    user_model: str,
    max_concurrency: int,
    max_steps: int,
    seed: int,
    verbose: bool,
) -> tuple[list[RolloutRecord], ActorStats]:
    """Run all tasks × trials for one actor. Returns (records, stats)."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_tasks

    stats = ActorStats(actor=actor_short)
    records: list[RolloutRecord] = []

    config = TextRunConfig(
        domain="airline",
        agent="llm_agent",
        user="user_simulator",
        llm_agent=actor_full,
        llm_user=user_model,
        num_trials=seeds_per_task,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
        seed=seed,
        log_level="WARNING",
    )

    try:
        results = run_tasks(
            config, tau2_tasks, save_path=None, console_display=False,
        )
    except Exception as exc:
        stats.error = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[pool] ERROR actor={actor_short}: {stats.error}", flush=True)
        return records, stats

    for sim in results.simulations:
        reward_info = sim.reward_info
        reward = reward_info.reward if reward_info is not None else None
        if reward is None:
            stats.n_skipped += 1
            if verbose:
                print(
                    f"[pool] WARN actor={actor_short} task={sim.task_id} "
                    f"trial={sim.trial}: reward missing, skipping",
                    flush=True,
                )
            continue

        passed = reward > 0
        messages = sim.get_messages()
        trial_idx = sim.trial if sim.trial is not None else 0
        trajectory_id = f"{actor_short}_airline_{sim.task_id}_trial{trial_idx}"
        rec = RolloutRecord(
            trajectory_id=trajectory_id,
            task_id=str(sim.task_id),
            task_split="",                        # filled in by caller
            actor_model=actor_short,
            actor_model_full=actor_full,
            seed=sim.seed,
            trial=trial_idx,
            conversation=[m.model_dump() for m in messages],
            outcome_label="pass" if passed else "fail",
            gold_reward=float(reward),
            duration_s=sim.duration,
            agent_cost=sim.agent_cost,
            user_cost=sim.user_cost,
            termination_reason=_term_reason_str(sim.termination_reason),
            num_messages=len(messages),
        )
        records.append(rec)
        stats.n_rollouts += 1
        if passed:
            stats.n_pass += 1
        stats.total_cost_usd += (sim.agent_cost or 0.0) + (sim.user_cost or 0.0)
        stats.total_duration_s += sim.duration

    return records, stats


def generate_pool(
    *,
    actors: list[str],               # short or full names
    task_ids: list[str],
    seeds_per_task: int,
    user_model: str,
    max_concurrency: int,
    out_path: Path,
    task_split: str,
    max_steps: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> PoolManifest:
    """Build a labeled trajectory pool. Streams JSONL as rollouts complete.

    Returns the manifest and writes both the JSONL pool and a sidecar
    `<out_path_stem>_manifest.json`.
    """
    from tau2.runner import get_tasks

    started_at = _now_iso()
    start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tau2_tasks = get_tasks("airline", task_ids=list(task_ids))
    if not tau2_tasks:
        raise ValueError(
            f"No airline tasks found for task_ids={task_ids}. "
            "Check TAU2_DATA_DIR and the tau3:search / tau3:holdout id lists."
        )

    resolved = [(_short_name(_resolve_actor(a)), _resolve_actor(a)) for a in actors]
    per_actor_stats: list[ActorStats] = []

    if verbose:
        print(
            f"[pool] Building pool — actors={[s for s, _ in resolved]}, "
            f"tasks={task_ids}, seeds/task={seeds_per_task}, "
            f"user_model={user_model}, concurrency={max_concurrency}",
            flush=True,
        )
        print(f"[pool] Output: {out_path}", flush=True)

    with out_path.open("w") as f_out:
        for actor_short, actor_full in resolved:
            if verbose:
                print(
                    f"[pool] actor={actor_short} ({actor_full}) "
                    f"on {len(tau2_tasks)} tasks × {seeds_per_task} seeds ...",
                    flush=True,
                )
            records, stats = _run_one_actor(
                actor_short=actor_short,
                actor_full=actor_full,
                tau2_tasks=tau2_tasks,
                seeds_per_task=seeds_per_task,
                user_model=user_model,
                max_concurrency=max_concurrency,
                max_steps=max_steps,
                seed=seed,
                verbose=verbose,
            )
            for rec in records:
                rec.task_split = task_split
                f_out.write(json.dumps(asdict(rec)) + "\n")
            f_out.flush()
            per_actor_stats.append(stats)
            if verbose:
                err_tag = f"  ERROR={stats.error}" if stats.error else ""
                print(
                    f"[pool]   actor={actor_short}: n={stats.n_rollouts}  "
                    f"pass_rate={stats.pass_rate:.0%}  "
                    f"cost=${stats.total_cost_usd:.3f}  "
                    f"skipped={stats.n_skipped}{err_tag}",
                    flush=True,
                )

    ended_at = _now_iso()
    total_duration = time.time() - start
    total_cost = sum(s.total_cost_usd for s in per_actor_stats)
    total_rollouts = sum(s.n_rollouts for s in per_actor_stats)
    total_pass = sum(s.n_pass for s in per_actor_stats)

    manifest = PoolManifest(
        pool_path=str(out_path),
        actors_short=[s for s, _ in resolved],
        actors_full=[f for _, f in resolved],
        task_ids=[str(t) for t in task_ids],
        task_split=task_split,
        seeds_per_task=seeds_per_task,
        user_model=user_model,
        max_concurrency=max_concurrency,
        started_at=started_at,
        ended_at=ended_at,
        total_cost_usd=total_cost,
        total_duration_s=total_duration,
        total_rollouts=total_rollouts,
        total_pass=total_pass,
        pool_pass_rate=total_pass / total_rollouts if total_rollouts else 0.0,
        per_actor_stats=[asdict(s) for s in per_actor_stats],
    )

    manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))

    if verbose:
        print(
            f"[pool] Done — {total_rollouts} rollouts "
            f"({total_pass} pass / {total_rollouts - total_pass} fail, "
            f"pass_rate={manifest.pool_pass_rate:.0%}), "
            f"cost=${total_cost:.3f}, {total_duration:.0f}s",
            flush=True,
        )
        print(f"[pool] Pool:     {out_path}", flush=True)
        print(f"[pool] Manifest: {manifest_path}", flush=True)

    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a τ²-labeled τ³-airline trajectory pool for Stage 1 judge training.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    p.add_argument(
        "--actors",
        default="claude-haiku-4-5,gpt-4o-mini",
        help=(
            "Comma-separated actor model names. Short names ({}) expand via "
            "ACTOR_MODEL_MAP; other values are passed through as LiteLLM ids. "
            "Default: claude-haiku-4-5,gpt-4o-mini."
        ).format(",".join(ACTOR_MODEL_MAP)),
    )
    p.add_argument(
        "--task-ids",
        default="0,20,40",
        help=(
            "Comma-separated airline task ids. Default: 0,20,40 "
            "(mini-pool spanning early/middle/late search-split tasks for "
            "initial adapter smoke)."
        ),
    )
    p.add_argument("--seeds", type=int, default=1, help="Trials per actor-task combo. Default: 1.")
    p.add_argument("--user-model", default="gpt-4.1", help="τ² user-simulator LLM. Default: gpt-4.1.")
    p.add_argument(
        "--task-split",
        default="judge-train",
        help=(
            "Annotation on each record (judge-train | judge-val). "
            "Default: judge-train. Used by the adapter to partition pairs."
        ),
    )
    p.add_argument("--max-concurrency", type=int, default=3, help="Parallel rollouts. Default: 3.")
    p.add_argument("--max-steps", type=int, default=100, help="Max conversation turns. Default: 100.")
    p.add_argument("--seed", type=int, default=42, help="Top-level RNG seed for trial expansion. Default: 42.")
    args = p.parse_args(argv)

    actors = [a.strip() for a in args.actors.split(",") if a.strip()]
    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]

    manifest = generate_pool(
        actors=actors,
        task_ids=task_ids,
        seeds_per_task=args.seeds,
        user_model=args.user_model,
        max_concurrency=args.max_concurrency,
        out_path=args.out,
        task_split=args.task_split,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    return 0 if manifest.total_rollouts > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
