"""Per-runtime agent execution. One function per runtime, one dispatcher.

The dispatcher `_run_runtime_once` picks the right function based on
`runtime=`; `run_agent` is the public one-shot that also does file copy-in
and artifact persistence.

Supported runtimes:
- codex_sdk       — native Python SDK (shared runner in codex_sdk_runner)
- codex_cli       — `codex exec` subprocess with hook emulation
- claude_sdk      — claude-agent-sdk in-process async query
- claude_code_cli — `claude` CLI subprocess with stream-json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .artifacts import (
    _copy_harness_files,
    _ensure_claude_md,
    _extract_last_agent_message_from_codex_trace,
    _persist_agent_run_artifacts,
    serialize_message,
)
from .hooks import _run_codex_sdk_with_hooks, run_codex_cli_with_hooks
from .results import AgentRunResult, _timeout_agent_result


def _run_codex_sdk_agent(
    prompt: str,
    model: str,
    work_dir: Path,
    timeout: int,
    approval_policy: str = "never",
    sandbox: str = "workspace-write",
) -> AgentRunResult:
    sdk_result = _run_codex_sdk_with_hooks(
        prompt, model, work_dir, timeout,
        approval_policy=approval_policy, sandbox=sandbox,
    )
    return AgentRunResult(
        final_response=sdk_result.final_response,
        trace_jsonl=sdk_result.normalized_trace_jsonl,
        raw_trace_jsonl=sdk_result.raw_events_jsonl,
        stderr=sdk_result.stderr,
        exit_code=sdk_result.exit_code,
        hook_failures=sdk_result.hook_failures,
        hook_warnings=sdk_result.hook_warnings,
    )


def _run_codex_cli_agent(
    prompt: str,
    model: str,
    work_dir: Path,
    timeout: int,
) -> AgentRunResult:
    try:
        result, hook_failures, hook_warnings = run_codex_cli_with_hooks(
            prompt=prompt, model=model, work_dir=work_dir, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _timeout_agent_result(timeout)

    trace = result.stdout or ""
    final = _extract_last_agent_message_from_codex_trace(trace) or ""
    return AgentRunResult(
        final_response=final,
        trace_jsonl=trace,
        stderr=result.stderr or "",
        exit_code=result.returncode,
        hook_failures=hook_failures,
        hook_warnings=hook_warnings,
    )


def _run_claude_code_cli_agent(
    prompt: str,
    model: str,
    work_dir: Path,
    timeout: int,
) -> AgentRunResult:
    _ensure_claude_md(work_dir)
    permission_mode = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions").strip()
    cmd = ["claude", "--print", "--verbose", "--output-format", "stream-json", "-p", prompt]
    if model:
        cmd.extend(["--model", model])
    if permission_mode:
        cmd.extend(["--permission-mode", permission_mode])
    try:
        result = subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _timeout_agent_result(timeout)

    return AgentRunResult(
        final_response="",
        trace_jsonl=result.stdout or "",
        stderr=result.stderr or "",
        exit_code=result.returncode,
    )


_HOOK_ERROR_RE = re.compile(r"Error in hook callback|ZodError", re.IGNORECASE)


def _extract_cli_hook_errors(stderr_lines: list[str]) -> list[str]:
    """Return stderr lines from the Claude CLI that indicate hook failures."""
    return [line for line in stderr_lines if _HOOK_ERROR_RE.search(line)]


def _run_agent_claude_sdk_sync(
    prompt: str,
    config_dir: str,
    model: str,
    work_dir: Path,
    timeout: int,
) -> AgentRunResult:
    """Run Claude Agent SDK against a candidate `build_options(ctx)` harness.

    Loads `config_dir/harness.py`, invokes `build_options(RunContext(...))`,
    validates the returned `ClaudeAgentOptions`, then issues `query()`.
    The proposer owns every field on `ClaudeAgentOptions` except `cwd` and
    `model`, which come from the runtime via ctx.
    """
    import asyncio as _asyncio
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    from meta_agent.harness_contracts.claude_agent_sdk import build_claude_agent_options
    from meta_agent.services.llm import ensure_bedrock_env, resolve_bedrock_model
    from meta_agent.core.run_context import RunContext

    ensure_bedrock_env()

    harness_path = Path(config_dir) / "harness.py"
    resolved_model = resolve_bedrock_model(model)
    ctx = RunContext(
        cwd=str(work_dir),
        model=resolved_model,
        task_instruction=prompt,
    )
    options = build_claude_agent_options(harness_path, ctx)

    stderr_lines: list[str] = []
    options.stderr = stderr_lines.append

    start_time = time.time()

    trace_records: list[dict[str, Any]] = []
    final_parts: list[str] = []
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None

    async def _inner() -> None:
        nonlocal num_turns, duration_ms, cost_usd, session_id
        nonlocal input_tokens, output_tokens, cache_tokens

        async for message in query(prompt=prompt, options=options):
            trace_records.append(serialize_message(message))

            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        final_parts.append(block.text)

            if isinstance(message, ResultMessage):
                num_turns = message.num_turns
                duration_ms = message.duration_ms
                cost_usd = message.total_cost_usd
                session_id = message.session_id
                usage = message.usage if isinstance(message.usage, dict) else {}
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                cache_tokens = usage.get("cache_read_input_tokens")

    try:
        _asyncio.run(_asyncio.wait_for(_inner(), timeout=timeout))
    except _asyncio.TimeoutError:
        return _timeout_agent_result(timeout)
    except Exception as exc:
        hook_failures = _extract_cli_hook_errors(stderr_lines)
        trace_jsonl = "\n".join(json.dumps(record) for record in trace_records)
        if trace_jsonl:
            trace_jsonl += "\n"
        all_stderr = f"{type(exc).__name__}: {exc}"
        if hook_failures:
            all_stderr += "\n" + "\n".join(hook_failures)
        return AgentRunResult(
            final_response="\n".join(final_parts),
            trace_jsonl=trace_jsonl,
            stderr=all_stderr,
            exit_code=1,
            hook_failures=hook_failures,
            duration_ms=int((time.time() - start_time) * 1000),
            wall_time_s=time.time() - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
            session_id=session_id,
        )

    hook_failures = _extract_cli_hook_errors(stderr_lines)

    wall_time_s = time.time() - start_time
    trace_jsonl = "\n".join(json.dumps(record) for record in trace_records)
    if trace_jsonl:
        trace_jsonl += "\n"

    return AgentRunResult(
        final_response="\n".join(final_parts),
        trace_jsonl=trace_jsonl,
        stderr="\n".join(hook_failures) if hook_failures else "",
        exit_code=0,
        hook_failures=hook_failures,
        cost_usd=cost_usd,
        num_turns=num_turns,
        duration_ms=duration_ms if duration_ms is not None else int(wall_time_s * 1000),
        wall_time_s=wall_time_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
    )


def _run_runtime_once(
    prompt: str,
    config_dir: str,
    model: str,
    work_dir: Path,
    timeout: int,
    runtime: str,
    approval_policy: str = "never",
    sandbox: str = "workspace-write",
) -> AgentRunResult:
    if runtime == "codex_sdk":
        return _run_codex_sdk_agent(
            prompt, model, work_dir, timeout,
            approval_policy=approval_policy, sandbox=sandbox,
        )
    if runtime == "codex_cli":
        return _run_codex_cli_agent(prompt, model, work_dir, timeout)
    if runtime == "claude_code_cli":
        return _run_claude_code_cli_agent(prompt, model, work_dir, timeout)
    if runtime == "claude_sdk":
        return _run_agent_claude_sdk_sync(prompt, config_dir, model, work_dir, timeout)
    raise ValueError(f"Unsupported runtime: {runtime}")


def run_agent(
    prompt: str,
    config_dir: str,
    model: str,
    work_dir: Path,
    timeout: int,
    runtime: str,
) -> AgentRunResult:
    """Dispatch to the correct runtime and return a uniform result.

    Copies harness files, runs the agent, writes trace artifacts, and
    returns an AgentRunResult. Callers never need to know which runtime
    is active.
    """
    _copy_harness_files(config_dir, work_dir)
    result = _run_runtime_once(prompt, config_dir, model, work_dir, timeout, runtime)
    _persist_agent_run_artifacts(work_dir, result)
    return result
