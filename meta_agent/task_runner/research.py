"""Research-harness retry loop.

The `research_single_file` target owns prompt/context/examples via a typed
`ResearchHarnessSpec`; the adapter owns the per-task loop (run once, verify,
optionally run one repair pass, return).

This file is the loop. Everything about `ResearchHarnessSpec` itself lives
in `meta_agent.harness_contracts.research`.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from meta_agent.core.benchmark import Task
from meta_agent.harness_contracts.research import (
    build_research_harness_spec,
    compose_initial_prompt,
)
from meta_agent.core.run_context import RunContext
from meta_agent.core.targets import get_target

from .artifacts import (
    _persist_agent_run_artifacts,
    _write_agent_result_metadata,
)
from .commands import run_command
from .results import (
    AgentRunResult,
    TaskResult,
    _task_result_from_agent_run,
)
from .runtimes import _run_runtime_once

_RESEARCH_HARNESS_FILENAME = get_target("research_single_file").module_filename


def _is_research_harness_path(config_path: str) -> bool:
    p = Path(config_path)
    return p.is_file() and p.name == _RESEARCH_HARNESS_FILENAME


def _jsonl_event(event_type: str, **payload: object) -> str:
    record = {"type": event_type, **payload}
    return json.dumps(record) + "\n"


def _concat_jsonl(parts: list[str]) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        normalized_parts.append(part if part.endswith("\n") else part + "\n")
    return "".join(normalized_parts)


def _sum_optional_ints(values: list[Optional[int]]) -> Optional[int]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _sum_optional_floats(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _truncate_for_trace(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _build_repair_prompt(
    task: Task,
    previous_result: AgentRunResult,
    verify_result: subprocess.CompletedProcess[str],
    runtime: str,
) -> str:
    from .results import _build_verify_output

    verify_output = _build_verify_output(verify_result, previous_result, runtime).strip()
    previous_response = previous_result.final_response.strip() or "(empty)"
    sections = [
        "The previous attempt did not pass verification.",
        f"Original task:\n{task.instruction.strip()}",
        f"Verifier output:\n{verify_output or '(no verifier output)'}",
        f"Previous final response:\n{previous_response}",
        (
            "You are still in the same workspace. Inspect the current files, fix the issue, "
            "and finish only when the verifier would pass."
        ),
    ]
    return "\n\n".join(sections)


def _merge_research_attempts(
    results: list[AgentRunResult],
    trace_parts: list[str],
    raw_trace_parts: list[str],
) -> AgentRunResult:
    final_result = results[-1]
    return AgentRunResult(
        final_response=final_result.final_response,
        trace_jsonl=_concat_jsonl(trace_parts),
        raw_trace_jsonl=_concat_jsonl(raw_trace_parts),
        stderr="\n".join(part for part in (r.stderr.strip() for r in results) if part),
        exit_code=final_result.exit_code,
        hook_failures=list(final_result.hook_failures),
        hook_warnings=list(final_result.hook_warnings),
        cost_usd=_sum_optional_floats([r.cost_usd for r in results]),
        num_turns=_sum_optional_ints([r.num_turns for r in results]),
        duration_ms=_sum_optional_ints([r.duration_ms for r in results]),
        wall_time_s=_sum_optional_floats([r.wall_time_s for r in results]),
        input_tokens=_sum_optional_ints([r.input_tokens for r in results]),
        output_tokens=_sum_optional_ints([r.output_tokens for r in results]),
        cache_tokens=_sum_optional_ints([r.cache_tokens for r in results]),
        session_id=final_result.session_id,
    )


def _run_research_harness_task(
    task: Task,
    config_path: str,
    model: str,
    work_dir: Path,
    runtime: str,
) -> TaskResult:
    """Run one research-harness task end to end, with an optional repair pass.

    `run_command` is imported from the package root via a late import to
    keep the test-patch surface at a single canonical path.
    """
    if runtime != "codex_sdk":
        raise ValueError(
            f"research_single_file currently supports only codex_sdk, got {runtime}"
        )

    start_time = time.time()
    harness_path = Path(config_path)
    ctx = RunContext(cwd=str(work_dir), model=model, task_instruction=task.instruction)
    spec = build_research_harness_spec(harness_path, ctx)

    trace_parts: list[str] = []
    raw_trace_parts: list[str] = []
    attempt_results: list[AgentRunResult] = []

    attempt = 1
    prompt = compose_initial_prompt(spec, task.instruction)
    latest_verify_result: Optional[subprocess.CompletedProcess[str]] = None

    while True:
        trace_parts.append(
            _jsonl_event(
                "ResearchHarnessAttempt",
                attempt=attempt,
                prompt_preview=_truncate_for_trace(prompt),
            )
        )
        agent_result = _run_runtime_once(
            prompt=prompt,
            config_dir=config_path,
            model=model,
            work_dir=work_dir,
            timeout=task.timeout,
            runtime=runtime,
            approval_policy=spec.runtime_settings.approval_policy,
            sandbox=spec.runtime_settings.sandbox,
        )
        attempt_results.append(agent_result)
        trace_parts.append(agent_result.trace_jsonl)
        raw_trace_parts.append(agent_result.raw_trace_jsonl)

        latest_verify_result = run_command(task.verify, cwd=work_dir, timeout=task.timeout)
        trace_parts.append(
            _jsonl_event(
                "ResearchHarnessVerify",
                attempt=attempt,
                exit_code=latest_verify_result.returncode,
                output_preview=_truncate_for_trace(
                    (latest_verify_result.stdout or "") + (latest_verify_result.stderr or "")
                ),
            )
        )

        can_retry = (
            attempt < spec.max_attempts
            and agent_result.exit_code == 0
            and latest_verify_result.returncode != 0
        )
        if not can_retry:
            break

        prompt = _build_repair_prompt(task, agent_result, latest_verify_result, runtime)
        attempt += 1

    combined_result = _merge_research_attempts(
        results=attempt_results,
        trace_parts=trace_parts,
        raw_trace_parts=raw_trace_parts,
    )
    _persist_agent_run_artifacts(work_dir, combined_result)
    _write_agent_result_metadata(work_dir, combined_result)
    assert latest_verify_result is not None
    return _task_result_from_agent_run(
        task=task,
        runtime=runtime,
        work_dir=work_dir,
        agent_result=combined_result,
        verify_result=latest_verify_result,
        fallback_wall_time_s=combined_result.wall_time_s or (time.time() - start_time),
    )
