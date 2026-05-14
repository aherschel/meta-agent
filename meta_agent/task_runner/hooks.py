"""Codex lifecycle hooks: probing native support and emulating when absent.

The Codex CLI/SDK both declare a hook schema in `.codex/hooks.json`. Older
CLIs don't run those hooks natively, so we emulate `SessionStart`,
`UserPromptSubmit`, and `Stop` for them; `PreToolUse`/`PostToolUse` are not
emulable and produce a warning.

The detection result is cached across calls (module-level global) so we
only probe `codex features list` once per process.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .artifacts import _extract_last_agent_message_from_codex_trace

if TYPE_CHECKING:
    from meta_agent.task_runner.codex_sdk import CodexSdkRunResult


_CODEX_HOOK_EVENTS_UNSUPPORTED = {"PreToolUse", "PostToolUse"}
_CODEX_HOOKS_NATIVE_SUPPORT: Optional[bool] = None


def _build_codex_exec_cmd(prompt: str, model: str) -> list[str]:
    cmd = ["codex", "exec", "--full-auto", "--json", "--skip-git-repo-check"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def _codex_native_hooks_supported() -> bool:
    """Probe `codex features list` to see if native hooks exist. Cached."""
    global _CODEX_HOOKS_NATIVE_SUPPORT

    if os.environ.get("META_AGENT_FORCE_CODEX_HOOK_EMULATION", "").strip() == "1":
        return False
    if os.environ.get("META_AGENT_ASSUME_CODEX_NATIVE_HOOKS", "").strip() == "1":
        return True
    if _CODEX_HOOKS_NATIVE_SUPPORT is not None:
        return _CODEX_HOOKS_NATIVE_SUPPORT

    try:
        result = subprocess.run(
            ["codex", "features", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _CODEX_HOOKS_NATIVE_SUPPORT = False
        else:
            _CODEX_HOOKS_NATIVE_SUPPORT = (
                re.search(r"^codex_hooks\s+", result.stdout, re.MULTILINE) is not None
            )
    except (OSError, subprocess.SubprocessError):
        _CODEX_HOOKS_NATIVE_SUPPORT = False

    return _CODEX_HOOKS_NATIVE_SUPPORT


def _load_codex_hooks_config(work_dir: Path) -> dict[str, Any]:
    hooks_path = work_dir / ".codex" / "hooks.json"
    if not hooks_path.is_file():
        return {}
    try:
        payload = json.loads(hooks_path.read_text())
    except json.JSONDecodeError:
        return {}
    hooks = payload.get("hooks")
    return hooks if isinstance(hooks, dict) else {}


def _hook_group_matches(event_name: str, group: dict[str, Any], payload: dict[str, Any]) -> bool:
    matcher = group.get("matcher")
    if not isinstance(matcher, str) or matcher in {"", "*"}:
        return True

    if event_name == "SessionStart":
        target = str(payload.get("source", ""))
    elif event_name in {"PreToolUse", "PostToolUse"}:
        target = str(payload.get("tool_name", ""))
    else:
        # UserPromptSubmit/Stop ignore the matcher entirely.
        return True

    try:
        return re.search(matcher, target) is not None
    except re.error:
        return False


def _run_codex_hook_event(
    hooks_config: dict[str, Any],
    event_name: str,
    work_dir: Path,
    model: str,
    payload: dict[str, Any],
) -> list[str]:
    """Run all hook commands registered for `event_name`. Returns failure strings."""
    groups = hooks_config.get(event_name)
    if not isinstance(groups, list):
        return []

    failures: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        if not _hook_group_matches(event_name, group, payload):
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            continue
        for handler in handlers:
            if not isinstance(handler, dict):
                continue
            handler_type = handler.get("type", "command")
            if handler_type != "command":
                continue
            command = handler.get("command")
            if not isinstance(command, str) or not command.strip():
                continue

            timeout_raw = handler.get("timeout", handler.get("timeoutSec", 600))
            try:
                timeout_sec = max(1, int(timeout_raw))
            except (TypeError, ValueError):
                timeout_sec = 600

            hook_input = {
                "session_id": payload.get("session_id", uuid.uuid4().hex),
                "transcript_path": str(work_dir / "trace.jsonl"),
                "cwd": str(work_dir),
                "hook_event_name": event_name,
                "model": model,
                **payload,
            }
            try:
                hook_result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    input=json.dumps(hook_input),
                    timeout=timeout_sec,
                )
                if hook_result.returncode != 0:
                    stderr = (hook_result.stderr or "").strip()
                    stdout = (hook_result.stdout or "").strip()
                    detail = stderr or stdout or f"exit={hook_result.returncode}"
                    failures.append(
                        f"{event_name}: command `{command}` failed ({detail})"
                    )
            except subprocess.TimeoutExpired:
                failures.append(
                    f"{event_name}: command `{command}` timed out after {timeout_sec}s"
                )

    return failures


def run_codex_cli_with_hooks(
    prompt: str,
    model: str,
    work_dir: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    """Run Codex CLI and emulate hooks when native hooks are unavailable."""
    cmd = _build_codex_exec_cmd(prompt, model)
    hooks_config = _load_codex_hooks_config(work_dir)
    emulate_hooks = bool(hooks_config) and not _codex_native_hooks_supported()
    hook_failures: list[str] = []
    hook_warnings: list[str] = []

    if emulate_hooks:
        unsupported = sorted(
            event for event in _CODEX_HOOK_EVENTS_UNSUPPORTED if event in hooks_config
        )
        if unsupported:
            hook_warnings.append(
                "Codex hook emulation does not support events: "
                + ", ".join(unsupported)
            )

        pre_payload = {"source": "startup", "prompt": prompt, "turn_id": uuid.uuid4().hex}
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "SessionStart", work_dir, model, pre_payload)
        )
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "UserPromptSubmit", work_dir, model, pre_payload)
        )

        if hook_failures:
            return (
                subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="\n".join(hook_failures),
                ),
                hook_failures,
                hook_warnings,
            )

    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if emulate_hooks:
        last_message = _extract_last_agent_message_from_codex_trace(result.stdout)
        stop_payload = {
            "turn_id": uuid.uuid4().hex,
            "stop_hook_active": False,
            "last_assistant_message": last_message,
        }
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "Stop", work_dir, model, stop_payload)
        )

    return result, hook_failures, hook_warnings


def _run_codex_sdk_with_hooks(
    prompt: str,
    model: str,
    work_dir: Path,
    timeout: int,
    approval_policy: str = "never",
    sandbox: str = "workspace-write",
) -> "CodexSdkRunResult":
    """Run Codex SDK via the shared Python runner with hook emulation."""
    from meta_agent.task_runner.codex_sdk import CodexSdkRunResult, run_codex_sdk_turn

    hooks_config = _load_codex_hooks_config(work_dir)
    emulate_hooks = bool(hooks_config) and not _codex_native_hooks_supported()
    hook_failures: list[str] = []
    hook_warnings: list[str] = []

    if emulate_hooks:
        unsupported = sorted(
            event for event in _CODEX_HOOK_EVENTS_UNSUPPORTED if event in hooks_config
        )
        if unsupported:
            hook_warnings.append(
                "Codex hook emulation does not support events: "
                + ", ".join(unsupported)
            )

        pre_payload = {"source": "startup", "prompt": prompt, "turn_id": uuid.uuid4().hex}
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "SessionStart", work_dir, model, pre_payload)
        )
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "UserPromptSubmit", work_dir, model, pre_payload)
        )

        if hook_failures:
            return CodexSdkRunResult(
                exit_code=1,
                stderr="\n".join(hook_failures),
                hook_failures=list(hook_failures),
                hook_warnings=list(hook_warnings),
            )

    sdk_result = run_codex_sdk_turn(
        prompt=prompt,
        model=model,
        cwd=str(work_dir),
        timeout_sec=timeout,
        approval_policy=approval_policy,
        sandbox=sandbox,
    )

    if emulate_hooks:
        stop_payload = {
            "turn_id": uuid.uuid4().hex,
            "stop_hook_active": False,
            "last_assistant_message": sdk_result.final_response,
        }
        hook_failures.extend(
            _run_codex_hook_event(hooks_config, "Stop", work_dir, model, stop_payload)
        )

    sdk_result.hook_failures = hook_failures
    sdk_result.hook_warnings = hook_warnings
    return sdk_result
