"""τ³-airline trajectory-judge adapter for Stage 1 pairwise task-success search.

Reads a pool JSONL produced by ``build_pool.py``, materializes within-task
(pass, fail) pairs, and dispatches to ``meta_agent.task_runner.judge_runner
::run_judge_benchmark`` for the actual pairwise judge evaluation. No new
MCP tool and no new stop-hook — we reuse ``submit_verdict`` verbatim.

Contract on top of the generic ``BenchmarkAdapter``:

* **Task filtering**: the adapter's "tasks" are *pairs*. Each JudgePair's
  ``category`` is set to ``task{task_id}`` so the generic
  ``write_category_scores`` sidecar emits per-task pairwise accuracy for free.
* **Macro-averaged metric**: ``macro_by_task_post_process`` overrides
  ``scores.json::mean_reward`` with the macro-averaged-across-tasks pairwise
  accuracy (one per-task accuracy computed from that task's pairs, then
  averaged across tasks). Matches Stage 2's per-task structure and the
  SOTA convention (Plan-RewardBench, ToolRM). The original pooled
  pair-level accuracy is preserved as ``pooled_pair_accuracy``.
* **Adapter-injected exit contract**: position_swap is on by default, so
  every pair is evaluated under both orderings; a pair is "correct" only
  when both orderings match gold. This is the plan's Stage-1 position-bias
  control (§5).

Pool JSONL schema is documented in ``build_pool.py``.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

from pydantic import BaseModel

from meta_agent.core import adapters
from meta_agent.core.benchmark import Benchmark
from meta_agent.utils.logging import get_logger
from meta_agent.core.targets import detect_target
from meta_agent.task_runner import TaskResult

from meta_agent.task_runner.judge_runner import (
    JudgeDecision,
    JudgePair,
    run_program_judge_benchmark,
    run_judge_benchmark,
)
from benchmarks.tau3_trajectory_judge.formatting import flatten_conversation

logger = get_logger("tau3_trajectory_judge")


# ---------------------------------------------------------------------------
# Backend schema
# ---------------------------------------------------------------------------


class TrajectoryJudgeBackend(BaseModel):
    """Benchmark backend for the pairwise trajectory-judge benchmark.

    ``pool_path`` is the JSONL pool written by ``build_pool.py``. ``task_split``
    filters records by their per-record ``task_split`` field, which is how
    judge-train / judge-val get separated within a single pool file (the
    canonical setup: one pool file → two splits via annotation).

    ``max_pairs_per_task`` caps Cartesian pair count per task. High-pair-count
    tasks otherwise dominate the pooled/macro metric; the plan targets 12.
    """

    pool_path: str
    task_split: str = "judge-train"
    max_pairs_per_task: int = 12
    position_swap: bool = True
    pair_sampling_seed: int = 42
    timeout: int = 180
    task_ids: Optional[List[str]] = None
    program_harness_mode: Literal["pairwise_decision", "pointwise_score"] = "pairwise_decision"


def parse_backend(bench: Benchmark) -> TrajectoryJudgeBackend:
    return TrajectoryJudgeBackend.model_validate(bench.backend or {})


# ---------------------------------------------------------------------------
# Pool loading + conversation flattening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """Deserialized pool record — only the fields pair construction needs."""

    trajectory_id: str
    task_id: str
    task_split: str
    actor_model: str
    outcome_label: str          # "pass" | "fail"
    conversation_text: str


def load_pool(pool_path: Path, task_split: str) -> list[Trajectory]:
    """Load a pool JSONL, filter by ``task_split``, pre-flatten conversations."""
    if not pool_path.is_file():
        raise FileNotFoundError(f"Pool JSONL not found: {pool_path}")

    out: list[Trajectory] = []
    with pool_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num} of {pool_path}: {exc}")
            if rec.get("task_split") != task_split:
                continue
            out.append(Trajectory(
                trajectory_id=str(rec["trajectory_id"]),
                task_id=str(rec["task_id"]),
                task_split=str(rec["task_split"]),
                actor_model=str(rec.get("actor_model", "?")),
                outcome_label=str(rec["outcome_label"]),
                conversation_text=flatten_conversation(
                    rec.get("conversation") or []
                ),
            ))
    return out


# ---------------------------------------------------------------------------
# Pair materialization (within-task, capped, deterministically shuffled)
# ---------------------------------------------------------------------------


_PAIR_FRAMING_QUESTION = (
    "Below are two customer-service agent trajectories on the same "
    "τ²-airline task. The agent converses with a customer (user) and may "
    "call airline-database tools (assistant tool_calls → tool results) to "
    "read or modify state. Decide which trajectory better resolves the "
    "customer's issue according to airline policy."
)

_POINTWISE_FRAMING_QUESTION = (
    "Score exactly one customer-service agent trajectory for a τ²-airline "
    "task. The agent converses with a customer (user) and may call "
    "airline-database tools (assistant tool_calls → tool results) to read or "
    "modify state. Higher scores should mean the trajectory is more likely to "
    "have correctly resolved the customer's issue according to airline policy."
)


def materialize_pairs(
    trajectories: list[Trajectory],
    *,
    max_pairs_per_task: int,
    seed: int = 42,
    task_filter: Optional[list[str]] = None,
    question: str = _PAIR_FRAMING_QUESTION,
) -> list[JudgePair]:
    """Group by task, enumerate (pass × fail), shuffle, cap at ``max_pairs_per_task``.

    Pass is fixed in slot A with gold=A>B. Position bias is handled downstream
    by ``run_judge_benchmark``'s ``position_swap=True`` path, which evaluates
    each pair twice and only counts it correct when both orderings agree.
    """
    by_task: dict[str, dict[str, list[Trajectory]]] = {}
    for t in trajectories:
        if task_filter and t.task_id not in task_filter:
            continue
        by_task.setdefault(t.task_id, {"pass": [], "fail": []}) \
               [t.outcome_label].append(t)

    pairs: list[JudgePair] = []
    # Deterministic per-task shuffling so the same pool + seed → same pair set.
    rng_base = random.Random(seed)
    for task_id in sorted(by_task):
        bucket = by_task[task_id]
        pass_trajs = bucket["pass"]
        fail_trajs = bucket["fail"]
        if not pass_trajs or not fail_trajs:
            continue

        task_pairs: list[tuple[Trajectory, Trajectory]] = [
            (p, f) for p in pass_trajs for f in fail_trajs
        ]
        # Sub-seed per task so adding a task later doesn't reshuffle earlier tasks.
        rng_task = random.Random(f"{seed}::{task_id}")
        rng_task.shuffle(task_pairs)
        task_pairs = task_pairs[:max_pairs_per_task]

        for ordinal, (pass_t, fail_t) in enumerate(task_pairs):
            pair_id = f"task{task_id}_p{ordinal:02d}"
            pairs.append(JudgePair(
                pair_id=pair_id,
                question=question,
                response_a=pass_t.conversation_text,
                response_b=fail_t.conversation_text,
                gold=cast(JudgeDecision, "A>B"),
                # `source` is free-form metadata; stuff enough here to trace
                # a pair back to its underlying trajectories from the trace.
                source=f"{pass_t.trajectory_id}|{fail_t.trajectory_id}",
                # `category` is read by write_category_scores → per-task
                # pairwise accuracy in category_scores.json for free.
                category=f"task{task_id}",
            ))
    # Surface rng_base usage so mypy + humans know seed shapes the outer loop.
    _ = rng_base
    return pairs


# ---------------------------------------------------------------------------
# Macro-by-task post-processing
# ---------------------------------------------------------------------------


_PAIR_ID_RE = re.compile(r"^task(?P<tid>[^_]+)_p(?P<ord>\d+)$")


def macro_by_task_post_process(candidate_dir: Path) -> None:
    """Overwrite ``mean_reward`` in scores.json with macro-avg pairwise accuracy.

    Default ``experience.write_candidate`` computes ``pass_rate = n_passed /
    n_tasks`` — pooled pair-level accuracy, which over-weights tasks with
    many pairs. We recompute per-task accuracy from per-pair results and
    average across tasks (each task = one datapoint). Matches Stage 2's
    per-task aggregation structure and Plan-RewardBench convention.

    The pooled accuracy is preserved under ``pooled_pair_accuracy`` for
    audit/backward-compat.
    """
    scores_path = candidate_dir / "scores.json"
    per_task_dir = candidate_dir / "per_task"
    if not scores_path.exists() or not per_task_dir.exists():
        return
    try:
        scores = json.loads(scores_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    by_task: dict[str, list[bool]] = {}
    for f in sorted(per_task_dir.glob("*.json")):
        if f.name.endswith("_agent_result.json"):
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        pair_id = str(data.get("short_name") or f.stem)
        m = _PAIR_ID_RE.match(pair_id)
        if not m:
            continue
        by_task.setdefault(m.group("tid"), []).append(bool(data.get("passed", False)))

    if not by_task:
        return

    per_task_acc = {t: (sum(v) / len(v)) for t, v in by_task.items()}
    macro_acc = sum(per_task_acc.values()) / len(per_task_acc)

    scores["pooled_pair_accuracy"] = scores.get("mean_reward")
    scores["mean_reward"] = macro_acc
    scores["pairwise_accuracy_macro_by_task"] = macro_acc
    scores["n_tasks_with_pairs"] = len(per_task_acc)
    scores["per_task_accuracies"] = per_task_acc
    scores_path.write_text(json.dumps(scores, indent=2))


# ---------------------------------------------------------------------------
# Adapter registration
# ---------------------------------------------------------------------------


def task_pool(bench: Benchmark) -> List[str]:
    """Return unique task_ids appearing in ``pool_path`` for this split.

    Used by the loop's batching mechanism. Loads the pool once; cost is
    amortized because ``_build_task_pool`` is called at most once per run.
    """
    backend = parse_backend(bench)
    trajs = load_pool(Path(backend.pool_path), backend.task_split)
    task_ids = sorted({t.task_id for t in trajs})
    if backend.task_ids:
        want = set(backend.task_ids)
        task_ids = [t for t in task_ids if t in want]
    return task_ids


async def run(
    *,
    benchmark: Benchmark,
    config_path: str,
    model: str,
    concurrency: int,
    task_filter: Optional[List[str]] = None,
    **_unused: Any,
) -> List[TaskResult]:
    backend = parse_backend(benchmark)
    pool_path = Path(backend.pool_path)

    trajs = load_pool(pool_path, backend.task_split)
    if not trajs:
        raise ValueError(
            f"Empty pool after filtering: pool_path={pool_path} "
            f"task_split={backend.task_split!r}. Check build_pool.py output + "
            "the task_split annotation per record."
        )

    effective_filter = task_filter if task_filter is not None else backend.task_ids
    pairs = materialize_pairs(
        trajs,
        max_pairs_per_task=backend.max_pairs_per_task,
        seed=backend.pair_sampling_seed,
        task_filter=effective_filter,
        question=(
            _POINTWISE_FRAMING_QUESTION
            if backend.program_harness_mode == "pointwise_score"
            else _PAIR_FRAMING_QUESTION
        ),
    )
    if not pairs:
        n_tasks_seen = len({t.task_id for t in trajs})
        raise ValueError(
            f"No (pass, fail) pairs could be materialized: {len(trajs)} "
            f"trajectories across {n_tasks_seen} task(s). Each task needs "
            "at least one pass AND one fail trajectory in the pool."
        )

    logger.info(
        f"tau3_trajectory_judge: {len(pairs)} pairs from "
        f"{len({p.category for p in pairs})} task(s) (split={backend.task_split}, "
        f"position_swap={backend.position_swap})"
    )
    target = detect_target(Path(config_path))
    if target.name == "program_harness" and backend.program_harness_mode == "pointwise_score":
        from benchmarks.plan_rewardbench.adapter import run_program_pointwise_score_benchmark

        return await run_program_pointwise_score_benchmark(
            pairs=pairs,
            config_path=config_path,
            model=model,
            concurrency=concurrency,
            timeout=backend.timeout,
            position_swap=backend.position_swap,
            trace_type="tau3_trajectory_pointwise_score_outcome",
            logger=logger,
        )

    runner = (
        run_program_judge_benchmark
        if target.name == "program_harness"
        else run_judge_benchmark
    )
    return await runner(
        pairs=pairs,
        config_path=config_path,
        model=model,
        concurrency=concurrency,
        timeout=backend.timeout,
        position_swap=backend.position_swap,
        trace_type="tau3_trajectory_judge_outcome",
        logger=logger,
    )


# Trajectory-judge supports both the Claude Agent SDK submit_verdict contract
# and the repo-owned program_harness contract for candidate-owned judge code.
_SUPPORTED_TARGETS = frozenset({"claude_agent_sdk", "program_harness"})


adapters.register(adapters.BenchmarkAdapter(
    name="tau3_trajectory_judge",
    run=run,
    task_pool=task_pool,
    post_process_scores=macro_by_task_post_process,
    supported_targets=_SUPPORTED_TARGETS,
))
