"""Task execution: dispatch a Task to a runtime, persist artifacts, return a result.

Public surface:
    run_task_with_runtime(task, config_dir, model, work_dir, runtime) -> TaskResult
    run_agent(prompt, config_dir, model, work_dir, timeout, runtime) -> AgentRunResult
    run_command(cmd, cwd, timeout) -> CompletedProcess
    TaskResult, AgentRunResult

Internal layout:
    results.py    — TaskResult / AgentRunResult dataclasses + builders
    artifacts.py  — harness file copy-in, trace serialization, disk persistence
    hooks.py      — Codex lifecycle hooks (probing + emulation)
    runtimes.py   — per-runtime agent execution + run_agent + _run_runtime_once
    research.py   — research-harness retry loop
"""
from __future__ import annotations

import time
from pathlib import Path

from meta_agent.core.benchmark import Task

from .artifacts import (
    _copy_harness_files,
    _ensure_claude_md,
    _extract_last_agent_message_from_codex_trace,
    _persist_agent_run_artifacts,
    _write_agent_result_metadata,
    serialize_block,
    serialize_message,
)
from .commands import run_command
from .hooks import (
    _build_codex_exec_cmd,
    _CODEX_HOOK_EVENTS_UNSUPPORTED,
    _codex_native_hooks_supported,
    _hook_group_matches,
    _load_codex_hooks_config,
    _run_codex_hook_event,
    _run_codex_sdk_with_hooks,
    run_codex_cli_with_hooks,
)
from .program import _is_program_harness_path, _run_program_harness_task
from .research import _is_research_harness_path, _run_research_harness_task
from .results import (
    AgentRunResult,
    TaskResult,
    _build_verify_output,
    _task_result_from_agent_run,
    _timeout_agent_result,
)
from .runtimes import (
    _run_agent_claude_sdk_sync,
    _run_claude_code_cli_agent,
    _run_codex_cli_agent,
    _run_codex_sdk_agent,
    _run_runtime_once,
    run_agent,
)

__all__ = [
    "AgentRunResult",
    "TaskResult",
    "run_agent",
    "run_command",
    "run_task_with_runtime",
]


async def run_task_with_runtime(
    task: Task,
    config_dir: str,
    model: str,
    work_dir: Path,
    runtime: str,
) -> TaskResult:
    """Run one task against one harness and return the verifier's verdict.

    This is the one entry point every evaluator should use. It dispatches to
    the research-harness retry loop when `config_dir` is a single harness.py,
    or otherwise to the standard `run_agent` path.
    """
    if _is_program_harness_path(config_dir):
        return await _run_program_harness_task(
            task=task,
            config_path=config_dir,
            model=model,
            work_dir=work_dir,
            runtime=runtime,
        )

    if _is_research_harness_path(config_dir):
        return _run_research_harness_task(
            task=task,
            config_path=config_dir,
            model=model,
            work_dir=work_dir,
            runtime=runtime,
        )

    start_time = time.time()
    agent_result = run_agent(
        prompt=task.instruction,
        config_dir=config_dir,
        model=model,
        work_dir=work_dir,
        timeout=task.timeout,
        runtime=runtime,
    )
    _write_agent_result_metadata(work_dir, agent_result)
    verify_result = run_command(task.verify, cwd=work_dir, timeout=task.timeout)
    return _task_result_from_agent_run(
        task=task,
        runtime=runtime,
        work_dir=work_dir,
        agent_result=agent_result,
        verify_result=verify_result,
        fallback_wall_time_s=time.time() - start_time,
    )
