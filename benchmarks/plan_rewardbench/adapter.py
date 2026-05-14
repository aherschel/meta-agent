"""Plan-RewardBench trajectory preference adapter.

Plan-RewardBench is already pairwise: each example has a user query, tool
definitions, a preferred tool-agent trajectory, and a rejected trajectory.
This adapter formats the two trajectories as candidate responses and reuses
the shared `submit_verdict` MCP contract from the judge runner.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

from pydantic import BaseModel

from meta_agent.core import adapters
from meta_agent.core.benchmark import Benchmark
from meta_agent.utils.logging import get_logger
from meta_agent.services.llm import resolve_bedrock_model
from meta_agent.harness_contracts.program import HarnessContext, run_program_harness
from meta_agent.core.targets import detect_target
from meta_agent.task_runner import TaskResult

from meta_agent.task_runner.judge_runner import (
    JudgeDecision,
    JudgePair,
    run_judge_benchmark,
    run_program_judge_benchmark,
)

logger = get_logger("plan_rewardbench")

PLAN_REWARDBENCH_BUCKETS: tuple[str, ...] = (
    "planning_multi_easy",
    "planning_multi_hard",
    "planning_single_easy",
    "planning_single_hard",
    "planning_robustness",
    "refusal",
    "irrelevance_unavailable",
)


class PlanRewardBenchBackend(BaseModel):
    dataset: str = "wyy1112/Plan-RewardBench"
    split: str = "train"
    n_examples: Optional[int] = None
    pair_ids: Optional[List[str]] = None
    pair_ids_path: Optional[str] = None
    include_buckets: Optional[List[str]] = None
    samples_per_bucket: Optional[int] = None
    sample_seed: int = 42
    position_swap: bool = True
    timeout: int = 300
    program_harness_mode: Literal["pairwise_decision", "pointwise_score"] = "pairwise_decision"


def parse_backend(bench: Benchmark) -> PlanRewardBenchBackend:
    return PlanRewardBenchBackend.model_validate(bench.backend or {})


def task_pool(bench: Benchmark) -> List[str]:
    backend = parse_backend(bench)
    pair_ids = _effective_pair_ids(backend)
    if pair_ids:
        return pair_ids
    return [pair.pair_id for pair in load_pairs(backend)]


def _format_tools(tools: Any) -> str:
    try:
        text = json.dumps(tools, indent=2, ensure_ascii=False)
    except TypeError:
        text = str(tools)
    return text


def _format_message(message: dict[str, Any], index: int) -> str:
    role = str(message.get("role") or "unknown")
    content = message.get("content")
    if not isinstance(content, str):
        try:
            content_text = json.dumps(content, ensure_ascii=False)
        except TypeError:
            content_text = str(content)
    else:
        content_text = content
    return f"{index}. [{role}]\n{content_text}"


def _format_trajectory(trajectory: Any) -> str:
    if not isinstance(trajectory, dict):
        return str(trajectory)
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        return json.dumps(trajectory, ensure_ascii=False)
    parts = [
        _format_message(message, index)
        for index, message in enumerate(messages, start=1)
        if isinstance(message, dict)
    ]
    return "\n\n".join(parts)


def _pointwise_task_context(question: str) -> str:
    marker = "User task:\n"
    if marker in question:
        return question[question.index(marker):].strip()
    return question.strip()


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def _load_pair_ids_from_manifest(path: str) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pair_ids_path does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    raw_ids = payload.get("pair_ids") if isinstance(payload, dict) else payload
    if not isinstance(raw_ids, list):
        raise ValueError(
            f"pair_ids_path must contain a JSON list or object with pair_ids: {manifest_path}"
        )
    pair_ids = [str(pair_id).strip() for pair_id in raw_ids if str(pair_id).strip()]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"pair_ids_path contains duplicate pair IDs: {manifest_path}")
    return pair_ids


def _effective_pair_ids(backend: PlanRewardBenchBackend) -> Optional[list[str]]:
    if backend.pair_ids and backend.pair_ids_path:
        raise ValueError("Set only one of pair_ids or pair_ids_path")
    if backend.pair_ids_path:
        return _load_pair_ids_from_manifest(backend.pair_ids_path)
    return list(backend.pair_ids) if backend.pair_ids else None


def _pair_from_row(
    row: dict[str, Any],
    backend: PlanRewardBenchBackend,
    row_index: int,
) -> Optional[JudgePair]:
    uuid = str(row.get("uuid") or "").strip()
    query = str(row.get("query") or "").strip()
    bucket = str(row.get("_lcp_bucket") or "unknown").strip() or "unknown"
    chosen = row.get("chosen")
    reject = row.get("reject")
    if not (uuid and query and isinstance(chosen, dict) and isinstance(reject, dict)):
        return None
    pair_id = f"{_safe_id(bucket)}__{uuid}__{row_index}"

    question = (
        "Evaluate which complete tool-augmented agent trajectory better handles "
        "the user's task. Consider planning quality, correct and relevant tool "
        "use, robustness to tool errors, safety/refusal behavior, and whether "
        "the final behavior satisfies the user.\n\n"
        f"User task:\n{query}\n\n"
        f"Available tools:\n{_format_tools(row.get('tools'))}"
    )
    return JudgePair(
        pair_id=pair_id,
        question=question,
        response_a=_format_trajectory(chosen),
        response_b=_format_trajectory(reject),
        gold=cast(JudgeDecision, "A>B"),
        source="plan_rewardbench",
        category=bucket,
    )


def _apply_sampling(
    pairs: list[JudgePair],
    backend: PlanRewardBenchBackend,
) -> list[JudgePair]:
    pair_ids = _effective_pair_ids(backend)
    if pair_ids:
        wanted = set(pair_ids)
        pairs = [pair for pair in pairs if pair.pair_id in wanted]
    if backend.include_buckets:
        buckets = set(backend.include_buckets)
        pairs = [pair for pair in pairs if pair.category in buckets]
    if backend.samples_per_bucket is not None:
        rng = random.Random(backend.sample_seed)
        grouped: dict[str, list[JudgePair]] = {}
        for pair in pairs:
            grouped.setdefault(pair.category, []).append(pair)
        sampled: list[JudgePair] = []
        for bucket, bucket_pairs in sorted(grouped.items()):
            bucket_pairs = sorted(bucket_pairs, key=lambda pair: pair.pair_id)
            rng.shuffle(bucket_pairs)
            sampled.extend(bucket_pairs[: backend.samples_per_bucket])
        pairs = sorted(sampled, key=lambda pair: (pair.category, pair.pair_id))
    if backend.n_examples is not None:
        pairs = pairs[: backend.n_examples]
    return pairs


def load_pairs(backend: PlanRewardBenchBackend) -> list[JudgePair]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(backend.dataset, split=backend.split)
    pairs = [
        pair
        for row_index, row in enumerate(ds)
        if isinstance(row, dict)
        if (pair := _pair_from_row(row, backend, row_index)) is not None
    ]
    return _apply_sampling(pairs, backend)


async def run(
    *,
    benchmark: Benchmark,
    config_path: str,
    model: str,
    concurrency: int,
    task_filter: Optional[list[str]] = None,
    **_unused: Any,
) -> list[TaskResult]:
    backend = parse_backend(benchmark)
    logger.info(f"Loading {backend.dataset} split={backend.split}...")
    pairs = load_pairs(backend)
    if task_filter:
        wanted = set(task_filter)
        pairs = [pair for pair in pairs if pair.pair_id in wanted]
    if not pairs:
        raise ValueError(
            f"No Plan-RewardBench pairs loaded (dataset={backend.dataset}, split={backend.split})"
        )

    target = detect_target(Path(config_path))
    if target.name == "program_harness" and backend.program_harness_mode == "pointwise_score":
        return await run_program_pointwise_score_benchmark(
            pairs=pairs,
            config_path=config_path,
            model=model,
            concurrency=concurrency,
            timeout=backend.timeout,
            position_swap=backend.position_swap,
            trace_type="plan_rewardbench_pointwise_score_outcome",
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
        trace_type="plan_rewardbench_outcome",
        logger=logger,
    )


@dataclass(frozen=True)
class ProgramScoreTask:
    """One trajectory scoring task exposed to a pointwise reward harness."""

    name: str
    pair_id: str
    question: str
    trajectory: str
    source: str = ""
    category: str = "Other"
    ordering_label: str = "original"
    trajectory_label: str = "trajectory"
    trajectory_ref: str = "response_a_original"


@dataclass
class _ScoreRunOutcome:
    label: str
    trajectory_ref: str
    score: Optional[float]
    wall_time_s: float
    final_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None


@dataclass
class _PointwiseOrderingOutcome:
    label: str
    decision_raw: Optional[JudgeDecision]
    decision_final: Optional[JudgeDecision]
    score_a: Optional[float]
    score_b: Optional[float]
    margin: Optional[float]
    score_runs: list[_ScoreRunOutcome]
    wall_time_s: float
    error: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PointwisePairOutcome:
    pair_id: str
    gold: JudgeDecision
    source: str
    category: str
    orderings: list[_PointwiseOrderingOutcome]
    wall_time_s: float
    question: str = ""
    response_a_original: str = ""
    response_b_original: str = ""

    @property
    def decisions(self) -> list[JudgeDecision]:
        return [o.decision_final for o in self.orderings if o.decision_final is not None]

    @property
    def error(self) -> Optional[str]:
        errs = [f"{o.label}: {o.error}" for o in self.orderings if o.error]
        return "; ".join(errs) if errs else None


def _resolve_harness_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_file() and path.name == "harness.py":
        return path
    candidate = path / "harness.py"
    if path.is_dir() and candidate.is_file():
        return candidate
    raise ValueError(
        "config_path must be a directory containing harness.py or the harness.py "
        f"file itself; got {config_path!r}"
    )


def _parse_score_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
        return score if math.isfinite(score) else None
    if isinstance(value, dict):
        for key in ("score", "reward", "scalar"):
            if key in value:
                score = _parse_score_value(value.get(key))
                if score is not None:
                    return score
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return _parse_score_value(json.loads(text))
    except json.JSONDecodeError:
        pass
    try:
        score = float(text)
    except ValueError:
        return None
    return score if math.isfinite(score) else None


def _score_from_program_result(result: Any) -> Optional[float]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        score = _parse_score_value(metadata)
        if score is not None:
            return score
    return _parse_score_value(getattr(result, "final_output", result))


def _compare_pointwise_scores(score_a: Optional[float], score_b: Optional[float]) -> Optional[JudgeDecision]:
    if score_a is None or score_b is None:
        return None
    if score_a > score_b:
        return "A>B"
    if score_b > score_a:
        return "B>A"
    return None


def _flip(decision: JudgeDecision) -> JudgeDecision:
    return "B>A" if decision == "A>B" else "A>B"


def _abstention_decision(label: str) -> JudgeDecision:
    return "A>B" if label == "swapped" else "B>A"


def _pointwise_position_consistent_correct(outcome: _PointwisePairOutcome) -> bool:
    if outcome.error is not None or not outcome.decisions:
        return False
    return all(decision == outcome.gold for decision in outcome.decisions)


async def _run_program_score_once(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
    ordering_label: str,
    trajectory_label: str,
    trajectory_ref: str,
    trajectory: str,
) -> _ScoreRunOutcome:
    start = time.time()
    cwd = Path(tempfile.mkdtemp(prefix=f"program_score_{pair.pair_id}_{ordering_label}_{trajectory_label}_"))
    task = ProgramScoreTask(
        name=f"{pair.pair_id}::{ordering_label}::{trajectory_label}",
        pair_id=pair.pair_id,
        question=_pointwise_task_context(pair.question),
        trajectory=trajectory,
        source=pair.source,
        category=pair.category,
        ordering_label=ordering_label,
        trajectory_label=trajectory_label,
        trajectory_ref=trajectory_ref,
    )
    ctx = HarnessContext(
        task=task,
        model=resolve_bedrock_model(model),
        cwd=cwd,
        timeout=timeout,
        metadata={
            "pair_id": pair.pair_id,
            "category": pair.category,
            "source": pair.source,
            "ordering_label": ordering_label,
            "trajectory_label": trajectory_label,
            "trajectory_ref": trajectory_ref,
            "pointwise_score": True,
        },
    )

    err: Optional[str] = None
    final_output: Any = None
    result_metadata: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    score: Optional[float] = None
    try:
        result = await run_program_harness(harness_path, ctx, timeout=timeout)
        final_output = result.final_output
        result_metadata = dict(result.metadata)
        score = _score_from_program_result(result)
        events = list(result.events)
        num_turns = result.num_turns
        cost_usd = result.cost_usd
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        cache_tokens = result.cache_tokens
        session_id = result.session_id
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
        events = ctx.events
    except Exception as exc:  # noqa: BLE001 - candidate failure is an outcome
        err = f"{type(exc).__name__}: {exc}"
        events = ctx.events
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    if score is None and err is None:
        err = "program score harness did not return a valid scalar score"

    events.append({
        "type": "program_pointwise_score",
        "ordering_label": ordering_label,
        "trajectory_label": trajectory_label,
        "trajectory_ref": trajectory_ref,
        "score": score,
        "final_output": final_output,
        "metadata": result_metadata,
        "error": err,
    })
    return _ScoreRunOutcome(
        label=trajectory_label,
        trajectory_ref=trajectory_ref,
        score=score,
        wall_time_s=time.time() - start,
        final_output=final_output,
        metadata=result_metadata,
        error=err,
        events=events,
        num_turns=num_turns,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
    )


async def _run_pointwise_ordering(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
    label: str,
    flip_responses: bool,
) -> _PointwiseOrderingOutcome:
    start = time.time()
    response_a = pair.response_b if flip_responses else pair.response_a
    response_b = pair.response_a if flip_responses else pair.response_b
    response_a_ref = "response_b_original" if flip_responses else "response_a_original"
    response_b_ref = "response_a_original" if flip_responses else "response_b_original"

    score_a = await _run_program_score_once(
        harness_path=harness_path,
        pair=pair,
        model=model,
        timeout=timeout,
        ordering_label=label,
        trajectory_label="current_a",
        trajectory_ref=response_a_ref,
        trajectory=response_a,
    )
    score_b = await _run_program_score_once(
        harness_path=harness_path,
        pair=pair,
        model=model,
        timeout=timeout,
        ordering_label=label,
        trajectory_label="current_b",
        trajectory_ref=response_b_ref,
        trajectory=response_b,
    )

    decision_raw = _compare_pointwise_scores(score_a.score, score_b.score)
    if decision_raw is None and score_a.error is None and score_b.error is None:
        decision_raw = _abstention_decision(label)
    decision_final = _flip(decision_raw) if (flip_responses and decision_raw is not None) else decision_raw
    margin = None if score_a.score is None or score_b.score is None else score_a.score - score_b.score
    errors = [run.error for run in (score_a, score_b) if run.error]
    error = "; ".join(errors) if errors else None
    events = [
        {
            "type": "pointwise_score_comparison",
            "label": label,
            "score_a": score_a.score,
            "score_b": score_b.score,
            "margin": margin,
            "decision_raw": decision_raw,
            "decision_final": decision_final,
            "error": error,
        }
    ]
    return _PointwiseOrderingOutcome(
        label=label,
        decision_raw=decision_raw,
        decision_final=decision_final,
        score_a=score_a.score,
        score_b=score_b.score,
        margin=margin,
        score_runs=[score_a, score_b],
        wall_time_s=time.time() - start,
        error=error,
        events=events,
    )


async def _judge_pair_pointwise_program(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
    position_swap: bool,
) -> _PointwisePairOutcome:
    start = time.time()
    orderings = [
        await _run_pointwise_ordering(
            harness_path=harness_path,
            pair=pair,
            model=model,
            timeout=timeout,
            label="original",
            flip_responses=False,
        )
    ]
    if position_swap:
        orderings.append(
            await _run_pointwise_ordering(
                harness_path=harness_path,
                pair=pair,
                model=model,
                timeout=timeout,
                label="swapped",
                flip_responses=True,
            )
        )
    return _PointwisePairOutcome(
        pair_id=pair.pair_id,
        gold=pair.gold,
        source=pair.source,
        category=pair.category,
        orderings=orderings,
        wall_time_s=time.time() - start,
        question=_pointwise_task_context(pair.question),
        response_a_original=pair.response_a,
        response_b_original=pair.response_b,
    )


def _write_pointwise_pair_trace(
    outcome: _PointwisePairOutcome,
    pair_dir: Path,
    *,
    trace_type: str,
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "pair_id": outcome.pair_id,
                "category": outcome.category,
                "source": outcome.source,
                "gold": outcome.gold,
                "input": {
                    "question": outcome.question,
                    "response_a_original": outcome.response_a_original,
                    "response_b_original": outcome.response_b_original,
                    "interface": "pointwise_score",
                },
                "decisions": list(outcome.decisions),
                "passed": _pointwise_position_consistent_correct(outcome),
                "orderings": [
                    {
                        "label": ordering.label,
                        "score_a": ordering.score_a,
                        "score_b": ordering.score_b,
                        "margin": ordering.margin,
                        "decision_raw": ordering.decision_raw,
                        "decision_final": ordering.decision_final,
                        "wall_time_s": ordering.wall_time_s,
                        "error": ordering.error,
                        "events": ordering.events,
                        "score_runs": [
                            {
                                "label": run.label,
                                "trajectory_ref": run.trajectory_ref,
                                "score": run.score,
                                "wall_time_s": run.wall_time_s,
                                "final_output": run.final_output,
                                "metadata": run.metadata,
                                "error": run.error,
                                "events": run.events,
                            }
                            for run in ordering.score_runs
                        ],
                    }
                    for ordering in outcome.orderings
                ],
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
            },
            default=str,
        )
        + "\n"
    )


def _task_result_from_pointwise_outcome(
    outcome: _PointwisePairOutcome,
    work_dir: Path,
    position_swap: bool,
) -> TaskResult:
    passed = _pointwise_position_consistent_correct(outcome)
    if outcome.error is not None:
        verify_exit_code = 1
        verify_output = f"error: {outcome.error}"
    elif not passed:
        verify_exit_code = 1
        decisions_str = ", ".join(outcome.decisions) if outcome.decisions else "(none)"
        verify_output = (
            f"gold={outcome.gold}  decisions=[{decisions_str}]"
            + ("  position_swap=on" if position_swap else "")
        )
    else:
        verify_exit_code = 0
        verify_output = f"gold={outcome.gold}  all pointwise comparisons correct"

    score_runs = [run for ordering in outcome.orderings for run in ordering.score_runs]
    total_cost = sum((run.cost_usd or 0.0) for run in score_runs if run.cost_usd is not None) or None
    total_input = sum((run.input_tokens or 0) for run in score_runs) or None
    total_output = sum((run.output_tokens or 0) for run in score_runs) or None
    total_cache = sum((run.cache_tokens or 0) for run in score_runs) or None
    total_turns = sum((run.num_turns or 0) for run in score_runs) or None
    first_session = next((run.session_id for run in score_runs if run.session_id), None)

    return TaskResult(
        task_name=outcome.pair_id,
        passed=passed,
        reward=1.0 if passed else 0.0,
        cost_usd=total_cost,
        num_turns=total_turns,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_tokens=total_cache,
        session_id=first_session,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


async def run_program_pointwise_score_benchmark(
    *,
    pairs: list[JudgePair],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    position_swap: bool,
    trace_type: str,
    logger: Any = None,
) -> list[TaskResult]:
    """Drive Plan-RB with a true pointwise program-harness score interface."""
    harness_path = _resolve_harness_path(config_path)
    n_total = len(pairs)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"program_pointwise_score_{trace_type}_"))

    if logger is not None:
        logger.info(
            f"Running {n_total} pointwise-score program-harness pairs  "
            f"concurrency={concurrency}  position_swap={position_swap}  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    lock = asyncio.Lock()

    async def _run_one(pair: JudgePair) -> TaskResult:
        nonlocal completed, running_passed
        async with sem:
            outcome = await _judge_pair_pointwise_program(
                harness_path=harness_path,
                pair=pair,
                model=model,
                timeout=timeout,
                position_swap=position_swap,
            )

        pair_dir = tmp_root / pair.pair_id
        _write_pointwise_pair_trace(outcome, pair_dir, trace_type=trace_type)
        result = _task_result_from_pointwise_outcome(outcome, pair_dir, position_swap)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            rate = running_passed / completed if completed else 0.0
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {pair.pair_id[:8]}..  "
                    f"{dur}  pass_rate={rate:.0%}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(pair) for pair in pairs])
    return list(results)


def plan_rewardbench_post_process(candidate_dir: Path) -> None:
    """Promote paper-style macro average across Plan-RB buckets to mean_reward."""
    scores_path = candidate_dir / "scores.json"
    category_path = candidate_dir / "category_scores.json"
    if not scores_path.is_file() or not category_path.is_file():
        return

    scores = json.loads(scores_path.read_text())
    category_scores = json.loads(category_path.read_text())
    if not isinstance(scores, dict) or not isinstance(category_scores, dict):
        return

    bucket_rates: dict[str, float] = {}
    missing_buckets: list[str] = []
    for bucket in PLAN_REWARDBENCH_BUCKETS:
        payload = category_scores.get(bucket)
        rate = payload.get("pass_rate") if isinstance(payload, dict) else None
        if isinstance(rate, (int, float)):
            bucket_rates[bucket] = float(rate)
        else:
            missing_buckets.append(bucket)

    if not bucket_rates:
        return

    macro_avg = sum(bucket_rates.values()) / len(bucket_rates)
    category_scores["_summary"] = {
        "paper_macro_avg": macro_avg,
        "metric": "macro_average_across_plan_rewardbench_buckets",
        "buckets": bucket_rates,
        "missing_buckets": missing_buckets,
    }
    category_path.write_text(json.dumps(category_scores, indent=2))

    if "pooled_pair_accuracy" not in scores:
        scores["pooled_pair_accuracy"] = scores.get("mean_reward")
    scores["plan_rewardbench_macro_avg"] = macro_avg
    scores["plan_rewardbench_macro_buckets"] = bucket_rates
    scores["plan_rewardbench_missing_buckets"] = missing_buckets
    scores["plan_rewardbench_metric"] = "macro_avg_7_buckets"
    scores["mean_reward"] = macro_avg
    scores_path.write_text(json.dumps(scores, indent=2))


_SUPPORTED_TARGETS = frozenset({"claude_agent_sdk", "program_harness"})

adapters.register(adapters.BenchmarkAdapter(
    name="plan_rewardbench",
    run=run,
    task_pool=task_pool,
    post_process_scores=plan_rewardbench_post_process,
    supported_targets=_SUPPORTED_TARGETS,
))
