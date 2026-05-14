"""Smoke-gate: lightweight runtime validation before full eval.

Meta-Harness (Lee et al. 2026) Algorithm 1 step 11 prescribes an
``interface validation`` step between proposer output and the expensive
EVALUATE call. Appendix D spells this out concretely:

    Write a small validation test that imports the module, instantiates
    the class, and *calls both methods on a tiny set of examples*.
    Harnesses proposed during the search should pass this test before
    being fully evaluated. A simple test script can catch most malformed
    or nonfunctional candidates in seconds and keep the cost of failures
    near zero.

Our pre-existing ``validate_config`` covers the "import and instantiate"
half but does not exercise the harness at runtime. That gap let
proposer-written hooks with malformed output shapes (e.g., a
``hookSpecificOutput`` block missing ``hookEventName``) pass validation
and then crash on every pair during full evaluation, wasting ~$25 per
broken candidate.

This module closes that gap. For supported benchmark types, we load one
real item from the benchmark's input pool, run it end-to-end through the
proposer's harness (full SDK + hooks + benchmark-injected MCP tools),
and inspect the resulting stderr for hook-callback errors. If the hook
throws a ZodError on the smoke pair it will throw on every pair; we
reject the candidate immediately and spare the full eval budget.

Design principles (from Meta-Harness Appendix D):
    * "Automate evaluation outside the proposer." The proposer never
      runs tests; the loop does smoke-gating on its behalf. This keeps
      the proposer stateless and cheap.
    * Don't reject on transient infra errors (Bedrock throttling,
      timeouts). A non-hook error in the smoke outcome is logged as a
      warning but doesn't block the candidate — the full eval's own
      retry machinery can handle it.
    * No new storage semantics. Smoke runs use ``judge_pair`` directly
      with an ephemeral cwd; no volume state is written.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from meta_agent.core.benchmark import Benchmark
from meta_agent.utils.logging import get_logger

logger = get_logger("loop")


# Stderr fragments that indicate the candidate's own code crashed, not a
# flaky infra issue. Any of these in the per-pair outcome's error field
# triggers rejection. Match against the judge runner's ``_merge_hook_errors``
# vocabulary so the same strings the judge runner surfaces here land in
# our rejection gate.
_HOOK_ERROR_PATTERNS: tuple[str, ...] = (
    "ZodError",
    "hook_stderr",
    "Error in hook callback",
    "Fatal error in message reader",
)


_POINTWISE_SEVERITIES = {"none", "minor", "major", "critical"}


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of a smoke run against a single candidate.

    ``ok`` is True if the candidate survived smoke and can proceed to
    full evaluation. False if we matched a hook-error pattern in the
    outcome. ``error`` carries the matched pattern + a truncated
    outcome.error snippet; store this verbatim in
    ``validation_error.json`` so the next proposer sees it.
    """

    ok: bool
    error: Optional[str] = None
    duration_s: float = 0.0
    pair_id: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def _match_hook_error(err_str: str) -> Optional[str]:
    """Return the first _HOOK_ERROR_PATTERNS fragment found in ``err_str``.

    Separate from the async driver to keep pure-logic testable without
    spawning SDK subprocesses.
    """
    for pattern in _HOOK_ERROR_PATTERNS:
        if pattern in err_str:
            return pattern
    return None


def _record_score_tool_input(raw_response: Any) -> Optional[dict[str, Any]]:
    """Return the forced record_score input from a Claude response, if present."""
    if not isinstance(raw_response, dict):
        return None
    content = raw_response.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use" or block.get("name") != "record_score":
            continue
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            continue
        return tool_input
    return None


def _record_score_tool_call_error(raw_response: Any) -> Optional[str]:
    """Validate the forced tool contains score plus proposer diagnostics."""
    tool_input = _record_score_tool_input(raw_response)
    if tool_input is None:
        return "score model did not use the forced record_score tool"

    score = tool_input.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 100:
        return "record_score tool call did not contain a numeric score in [1, 100]"

    critique = tool_input.get("critique")
    if not isinstance(critique, str) or not critique.strip():
        return "record_score tool call did not contain a nonempty critique"

    rubric_issue = tool_input.get("rubric_issue")
    if not isinstance(rubric_issue, str) or not rubric_issue.strip():
        return "record_score tool call did not contain a nonempty rubric_issue"
    if len(rubric_issue.strip()) > 80:
        return "record_score rubric_issue is too long; use a compact diagnostic label"

    severity = tool_input.get("severity")
    if not isinstance(severity, str) or severity.strip().lower() not in _POINTWISE_SEVERITIES:
        return "record_score severity must be one of none, minor, major, critical"

    return None


def _metadata_diagnostics_error(metadata: dict[str, Any]) -> Optional[str]:
    critique = metadata.get("critique")
    if not isinstance(critique, str) or not critique.strip():
        return "score run did not expose a nonempty critique in metadata"

    rubric_issue = metadata.get("rubric_issue")
    if not isinstance(rubric_issue, str) or not rubric_issue.strip():
        return "score run did not expose a nonempty rubric_issue in metadata"
    if len(rubric_issue.strip()) > 80:
        return "score run metadata rubric_issue is too long"

    severity = metadata.get("severity")
    if not isinstance(severity, str) or severity.strip().lower() not in _POINTWISE_SEVERITIES:
        return "score run metadata severity must be one of none, minor, major, critical"

    return None


def _has_record_score_tool_call(raw_response: Any) -> bool:
    """Return True when a Claude response contains the required score tool call."""
    return _record_score_tool_call_error(raw_response) is None


def _pointwise_score_contract_error(outcome: Any) -> Optional[str]:
    """Reject pointwise candidates that weaken scalar diagnostic plumbing."""
    err_str = getattr(outcome, "error", None)
    if err_str:
        return f"pointwise score smoke outcome error: {str(err_str)[:800]}"

    for ordering in getattr(outcome, "orderings", []) or []:
        for run in getattr(ordering, "score_runs", []) or []:
            run_error = getattr(run, "error", None)
            if run_error:
                return f"score run failed: {str(run_error)[:800]}"
            if getattr(run, "score", None) is None:
                return "score run did not return a valid scalar"

            metadata = getattr(run, "metadata", None)
            if not isinstance(metadata, dict):
                return "score run did not expose metadata for structured-output validation"
            if metadata.get("output_mode") != "forced_tool_score":
                return (
                    "score harness did not report output_mode='forced_tool_score'; "
                    "keep the forced record_score tool plumbing from the scaffold"
                )
            diagnostics_error = _metadata_diagnostics_error(metadata)
            if diagnostics_error:
                return diagnostics_error
            model_text = str(metadata.get("model_text") or "").strip()
            if model_text:
                return (
                    "score model emitted free text/prose instead of tool-only structured output: "
                    f"{model_text[:300]}"
                )
            tool_error = _record_score_tool_call_error(metadata.get("model_raw"))
            if tool_error:
                return tool_error
    return None


async def smoke_candidate(
    harness_path: Path,
    benchmark: Benchmark,
    model: str,
    timeout_s: int = 180,
) -> SmokeResult:
    """Run one representative benchmark item through ``harness_path``.

    Dispatches by ``benchmark.type``. Currently supports
    ``tau3_trajectory_judge`` and ``plan_rewardbench`` end-to-end; other benchmark types fall
    through to a no-op ``ok=True`` so the smoke-gate is safe to wire
    into the loop globally before all benchmarks are covered.
    """
    start = time.time()

    if benchmark.type == "tau3_trajectory_judge":
        return await _smoke_tau3_trajectory_judge(
            harness_path=harness_path,
            benchmark=benchmark,
            model=model,
            timeout_s=timeout_s,
            start=start,
        )

    if benchmark.type == "plan_rewardbench":
        return await _smoke_plan_rewardbench(
            harness_path=harness_path,
            benchmark=benchmark,
            model=model,
            timeout_s=timeout_s,
            start=start,
        )

    if benchmark.type in {"tau", "tau3"}:
        return await _smoke_tau3_agent(
            harness_path=harness_path,
            benchmark=benchmark,
            model=model,
            timeout_s=timeout_s,
            start=start,
        )

    # Benchmarks without a dedicated smoke path pass through unchanged.
    logger.info(
        f"smoke-gate: no handler for benchmark type {benchmark.type!r}; "
        "passing through (ok=True, no runtime check)"
    )
    return SmokeResult(ok=True, duration_s=0.0)


async def _smoke_tau3_agent(
    *,
    harness_path: Path,
    benchmark: Benchmark,
    model: str,
    timeout_s: int,
    start: float,
) -> SmokeResult:
    """Run one tau3 task through the detected actor harness."""
    from benchmarks.tau3.adapter import parse_backend, run as run_tau3

    backend = parse_backend(benchmark)
    task_ids = list(backend.task_ids or [])
    if not task_ids:
        return SmokeResult(
            ok=True,
            error="tau3 smoke has no explicit task_ids; skipping smoke",
            duration_s=time.time() - start,
        )

    task_id = str(task_ids[0])
    smoke_bench = benchmark.model_copy(deep=True)
    smoke_bench.backend["task_ids"] = [task_id]
    logger.info(
        f"smoke-gate: running tau3 task {task_id} on {harness_path}"
    )

    try:
        results = await run_tau3(
            benchmark=smoke_bench,
            config_path=str(harness_path),
            model=model,
            concurrency=1,
        )
    except Exception as exc:  # noqa: BLE001 - smoke rejection only for candidate errors
        return SmokeResult(
            ok=False,
            error=f"tau3 smoke crashed: {type(exc).__name__}: {str(exc)[:800]}",
            duration_s=time.time() - start,
            pair_id=task_id,
        )

    duration = time.time() - start
    if not results:
        return SmokeResult(
            ok=False,
            error="tau3 smoke produced no result",
            duration_s=duration,
            pair_id=task_id,
        )

    result = results[0]
    err_str = result.verify_output or ""
    matched = _match_hook_error(err_str)
    if matched is not None:
        return SmokeResult(
            ok=False,
            error=f"{matched} in tau3 smoke outcome: {err_str[:800]}",
            duration_s=duration,
            pair_id=task_id,
        )
    if err_str:
        return SmokeResult(
            ok=False,
            error=f"tau3 candidate error in smoke: {err_str[:800]}",
            duration_s=duration,
            pair_id=task_id,
        )

    logger.info(
        f"smoke-gate: tau3 task {task_id} completed in {duration:.1f}s"
    )
    return SmokeResult(ok=True, duration_s=duration, pair_id=task_id)


async def _smoke_plan_rewardbench(
    *,
    harness_path: Path,
    benchmark: Benchmark,
    model: str,
    timeout_s: int,
    start: float,
) -> SmokeResult:
    """Run one Plan-RewardBench pair through the real SDK judge path.

    Plan-RB candidate failures are expensive because a malformed harness can
    fan out across hundreds of pairwise SDK calls before crashing. This smoke
    path executes a single deterministic pair with position swapping disabled,
    which is enough to catch import errors, hook schema errors, and invalid
    Claude SDK option shapes before the full batch.
    """
    from meta_agent.task_runner.judge_runner import judge_pair, judge_pair_program
    from benchmarks.plan_rewardbench.adapter import (
        _judge_pair_pointwise_program,
        load_pairs,
        parse_backend,
    )
    from meta_agent.core.targets import detect_target

    backend = parse_backend(benchmark)
    pairs = load_pairs(backend)
    if not pairs:
        return SmokeResult(
            ok=True,
            error="no Plan-RewardBench pairs loaded; skipping smoke",
            duration_s=time.time() - start,
        )

    pair = pairs[0]
    logger.info(
        f"smoke-gate: running Plan-RB pair {pair.pair_id} on {harness_path}"
    )

    target = detect_target(harness_path)
    if target.name == "program_harness" and backend.program_harness_mode == "pointwise_score":
        outcome = await _judge_pair_pointwise_program(
            harness_path=harness_path,
            pair=pair,
            model=model,
            timeout=min(timeout_s, backend.timeout),
            position_swap=False,
        )
        contract_error = _pointwise_score_contract_error(outcome)
        duration = time.time() - start
        if contract_error is not None:
            return SmokeResult(
                ok=False,
                error=contract_error,
                duration_s=duration,
                pair_id=pair.pair_id,
            )
    else:
        runner = judge_pair_program if target.name == "program_harness" else judge_pair
        outcome = await runner(
            harness_path=harness_path,
            pair=pair,
            model=model,
            timeout=min(timeout_s, backend.timeout),
            position_swap=False,
        )
        duration = time.time() - start

    err_str = outcome.error or ""
    matched = _match_hook_error(err_str)
    if matched is not None:
        return SmokeResult(
            ok=False,
            error=f"{matched} in smoke outcome: {err_str[:800]}",
            duration_s=duration,
            pair_id=pair.pair_id,
        )

    if err_str:
        logger.warning(
            f"smoke-gate: non-hook error for {harness_path.name} "
            f"(not rejecting): {err_str[:300]}"
        )

    logger.info(
        f"smoke-gate: {harness_path.name} passed Plan-RB pair {pair.pair_id} "
        f"in {duration:.1f}s"
    )
    return SmokeResult(ok=True, duration_s=duration, pair_id=pair.pair_id)


async def _smoke_tau3_trajectory_judge(
    *,
    harness_path: Path,
    benchmark: Benchmark,
    model: str,
    timeout_s: int,
    start: float,
) -> SmokeResult:
    """Run one materialized pair from the judge-train pool through the harness.

    Uses ``position_swap=False`` (single SDK session instead of two) to
    cut smoke cost in half. Hook errors surface on either ordering, so
    one ordering is sufficient to catch malformed-output bugs.
    """
    from meta_agent.task_runner.judge_runner import judge_pair, judge_pair_program
    from benchmarks.tau3_trajectory_judge.adapter import (
        load_pool,
        materialize_pairs,
        parse_backend,
    )
    from meta_agent.core.targets import detect_target

    backend = parse_backend(benchmark)
    trajs = load_pool(Path(backend.pool_path), backend.task_split)
    if not trajs:
        # Pool missing or empty — don't block the candidate; the full
        # eval will fail with a clearer diagnostic.
        return SmokeResult(
            ok=True,
            error="no trajectories in pool; skipping smoke",
            duration_s=time.time() - start,
        )

    pairs = materialize_pairs(
        trajs,
        max_pairs_per_task=backend.max_pairs_per_task,
        seed=backend.pair_sampling_seed,
        question=(
            "Score exactly one customer-service agent trajectory for a "
            "tau-airline task. Higher scores mean the trajectory is more "
            "likely to have correctly resolved the customer's issue according "
            "to airline policy."
            if getattr(backend, "program_harness_mode", "pairwise_decision") == "pointwise_score"
            else (
                "Below are two customer-service agent trajectories on the same "
                "tau-airline task. Decide which trajectory better resolves the "
                "customer's issue according to airline policy."
            )
        ),
    )
    if not pairs:
        return SmokeResult(
            ok=True,
            error="no pairs materialized from pool; skipping smoke",
            duration_s=time.time() - start,
        )

    # Deterministic first-pair pick — materialize_pairs is seeded, so
    # every candidate in an epoch sees the same smoke input.
    pair = pairs[0]
    logger.info(
        f"smoke-gate: running pair {pair.pair_id} on {harness_path}"
    )

    target = detect_target(harness_path)
    if target.name == "program_harness" and backend.program_harness_mode == "pointwise_score":
        from benchmarks.plan_rewardbench.adapter import _judge_pair_pointwise_program

        outcome = await _judge_pair_pointwise_program(
            harness_path=harness_path,
            pair=pair,
            model=model,
            timeout=min(timeout_s, backend.timeout),
            position_swap=False,
        )
        contract_error = _pointwise_score_contract_error(outcome)
        duration = time.time() - start
        if contract_error is not None:
            return SmokeResult(
                ok=False,
                error=contract_error,
                duration_s=duration,
                pair_id=pair.pair_id,
            )
    else:
        runner = judge_pair_program if target.name == "program_harness" else judge_pair
        outcome = await runner(
            harness_path=harness_path,
            pair=pair,
            model=model,
            timeout=timeout_s,
            position_swap=False,
        )
        duration = time.time() - start

    err_str = outcome.error or ""
    matched = _match_hook_error(err_str)
    if matched is not None:
        return SmokeResult(
            ok=False,
            error=f"{matched} in smoke outcome: {err_str[:800]}",
            duration_s=duration,
            pair_id=pair.pair_id,
        )

    # Non-hook errors (timeout, Bedrock throttle, transient network) are
    # not the candidate's fault. Log + let full eval handle them.
    if err_str:
        logger.warning(
            f"smoke-gate: non-hook error for {harness_path.name} "
            f"(not rejecting): {err_str[:300]}"
        )

    logger.info(
        f"smoke-gate: {harness_path.name} passed pair {pair.pair_id} "
        f"in {duration:.1f}s"
    )
    return SmokeResult(ok=True, duration_s=duration, pair_id=pair.pair_id)
