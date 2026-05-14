from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from meta_agent.core.benchmark import Task
from meta_agent.harness_contracts.program import (
    HarnessContext,
    HarnessResult,
    events_to_jsonl,
    run_program_harness,
)
from meta_agent.core.targets import TargetDetectionError, detect_target

from .artifacts import _persist_agent_run_artifacts, _write_agent_result_metadata
from .commands import run_command
from .results import AgentRunResult, TaskResult, _task_result_from_agent_run


@dataclass(frozen=True)
class ProgramTask:
    """Minimal local-task payload exposed to program harness candidates."""

    name: str
    instruction: str


def _is_program_harness_path(config_path: str) -> bool:
    try:
        return detect_target(Path(config_path)).name == "program_harness"
    except TargetDetectionError:
        return False


def _stringify_final_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


def _agent_result_from_harness_result(result: HarnessResult) -> AgentRunResult:
    events_jsonl = events_to_jsonl(result.events)
    final_response = _stringify_final_output(result.final_output)
    trace_jsonl = events_jsonl
    if final_response:
        trace_jsonl += json.dumps({
            "type": "final_output",
            "final_output": final_response,
        }) + "\n"
    return AgentRunResult(
        final_response=final_response,
        trace_jsonl=trace_jsonl,
        events_jsonl=events_jsonl,
        stderr="",
        exit_code=0,
        metadata=result.metadata,
        cost_usd=result.cost_usd,
        num_turns=result.num_turns,
        duration_ms=result.duration_ms,
        wall_time_s=result.wall_time_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_tokens=result.cache_tokens,
        session_id=result.session_id,
    )


def _error_agent_result(exc: BaseException, wall_time_s: float) -> AgentRunResult:
    message = f"{type(exc).__name__}: {exc}"
    trace_jsonl = json.dumps({"type": "error", "error": message}) + "\n"
    return AgentRunResult(
        final_response="",
        trace_jsonl=trace_jsonl,
        events_jsonl=trace_jsonl,
        stderr=message,
        exit_code=1,
        duration_ms=int(wall_time_s * 1000),
        wall_time_s=wall_time_s,
    )


async def _run_program_harness_task(
    task: Task,
    config_path: str,
    model: str,
    work_dir: Path,
    runtime: str,
) -> TaskResult:
    if runtime != "program_harness":
        raise ValueError(
            f"program_harness target requires runtime='program_harness', got {runtime}"
        )

    start_time = time.time()
    harness_path = Path(config_path)
    ctx = HarnessContext(
        task=ProgramTask(name=task.name, instruction=task.instruction),
        model=model,
        cwd=work_dir,
        timeout=task.timeout,
        metadata={"task_name": task.name},
    )

    try:
        result = await run_program_harness(
            harness_path,
            ctx,
            timeout=task.timeout,
        )
        agent_result = _agent_result_from_harness_result(result)
    except asyncio.TimeoutError as exc:
        agent_result = _error_agent_result(exc, time.time() - start_time)
    except Exception as exc:
        agent_result = _error_agent_result(exc, time.time() - start_time)

    _persist_agent_run_artifacts(work_dir, agent_result)
    _write_agent_result_metadata(work_dir, agent_result)
    verify_result = run_command(task.verify, cwd=work_dir, timeout=task.timeout)
    return _task_result_from_agent_run(
        task=task,
        runtime=runtime,
        work_dir=work_dir,
        agent_result=agent_result,
        verify_result=verify_result,
        fallback_wall_time_s=agent_result.wall_time_s or (time.time() - start_time),
    )
