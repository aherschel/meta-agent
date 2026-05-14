"""Typed outputs from a task run.

Two dataclasses share the results boundary between runtimes (AgentRunResult)
and evaluators (TaskResult). Any code that wants to look at what happened
during a run reads from these types — no dict poking.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from meta_agent.core.benchmark import Task


@dataclass
class TaskResult:
    """Verifier-level outcome for one task trial. Consumed by eval_runner."""

    task_name: str
    passed: bool
    reward: float
    cost_usd: Optional[float]
    num_turns: Optional[int]
    duration_ms: Optional[int]
    wall_time_s: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_tokens: Optional[int]
    session_id: Optional[str]
    work_dir: str
    verify_exit_code: int
    verify_output: str


@dataclass
class AgentRunResult:
    """Uniform result from any runtime — what benchmark adapters consume.

    Not every field is populated by every runtime. Fields populated by each:
    - codex_sdk:      all core fields (final_response, traces, usage, timing)
    - codex_cli:      final_response, traces, exit code, hook failures
    - claude_sdk:     all fields including usage/tokens/session_id/cost_usd
    - claude_code_cli: stream JSON in trace_jsonl; minimal timing/usage
    """

    final_response: str = ""
    trace_jsonl: str = ""
    raw_trace_jsonl: str = ""
    events_jsonl: str = ""
    stderr: str = ""
    exit_code: int = 0
    hook_failures: List[str] = field(default_factory=list)
    hook_warnings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    wall_time_s: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None


def _timeout_agent_result(timeout: int) -> AgentRunResult:
    """Return a clean timeout result with a parseable trace entry."""
    trace = json.dumps(
        {"type": "error", "error": f"TIMEOUT after {timeout}s"}
    ) + "\n"
    return AgentRunResult(
        final_response="",
        trace_jsonl=trace,
        stderr=f"TIMEOUT after {timeout}s",
        exit_code=1,
    )


def _build_verify_output(
    verify_result: subprocess.CompletedProcess[str],
    agent_result: AgentRunResult,
    runtime: str,
) -> str:
    output = (verify_result.stdout or "") + (verify_result.stderr or "")
    if agent_result.exit_code != 0:
        output += f"\n[{runtime}] exit={agent_result.exit_code}\n"
        if agent_result.stderr.strip():
            output += f"{agent_result.stderr.strip()}\n"
    if agent_result.hook_warnings or agent_result.hook_failures:
        output += "\n[runtime_hooks]\n"
        for warning in agent_result.hook_warnings:
            output += f"warning: {warning}\n"
        for failure in agent_result.hook_failures:
            output += f"failure: {failure}\n"
    return output


def _task_result_from_agent_run(
    *,
    task: Task,
    runtime: str,
    work_dir: Path,
    agent_result: AgentRunResult,
    verify_result: subprocess.CompletedProcess[str],
    fallback_wall_time_s: float,
) -> TaskResult:
    runtime_ok = agent_result.exit_code == 0
    hooks_ok = len(agent_result.hook_failures) == 0
    passed = verify_result.returncode == 0 and runtime_ok and hooks_ok

    verify_exit_code = verify_result.returncode
    if verify_exit_code == 0 and not runtime_ok:
        verify_exit_code = agent_result.exit_code or 1
    if verify_exit_code == 0 and not hooks_ok:
        verify_exit_code = 1

    wall_time_s = agent_result.wall_time_s
    if wall_time_s is None:
        wall_time_s = fallback_wall_time_s

    duration_ms = agent_result.duration_ms
    if duration_ms is None:
        duration_ms = int(wall_time_s * 1000)

    return TaskResult(
        task_name=task.name,
        passed=passed,
        reward=1.0 if passed else 0.0,
        cost_usd=agent_result.cost_usd,
        num_turns=agent_result.num_turns,
        duration_ms=duration_ms,
        wall_time_s=wall_time_s,
        input_tokens=agent_result.input_tokens,
        output_tokens=agent_result.output_tokens,
        cache_tokens=agent_result.cache_tokens,
        session_id=agent_result.session_id,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=_build_verify_output(verify_result, agent_result, runtime),
    )
