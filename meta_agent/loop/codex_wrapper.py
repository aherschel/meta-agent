"""Small, constrained wrapper around `codex exec` for proposer sessions.

This mirrors the role of Stanford meta-harness's `claude_wrapper.py`: keep the
outer optimizer's proposer subprocess protocol boring and inspectable.  The
wrapper owns command construction, streaming JSONL capture, timeout handling,
and telemetry extraction; higher-level loop code decides whether the files the
proposer wrote are usable.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class CodexWrapperResult:
    exit_code: int
    command: list[str]
    stderr: str = ""
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None


def _toml_string(value: str) -> str:
    """Return a minimal TOML string literal for Codex `-c key=value` overrides."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _maybe_configure_azure_provider(model: Optional[str]) -> tuple[Optional[str], list[str]]:
    """Map Azure deployment names onto Codex's OpenAI-compatible provider config."""
    if not model:
        return model, []

    requested = model
    deployment = model
    if model.startswith("azure:"):
        deployment = model.split(":", 1)[1]
    elif model.startswith("azure/"):
        deployment = model.split("/", 1)[1]
    else:
        azure_deployment = os.environ.get("AZURE_GPT55_DEPLOYMENT", "").strip()
        if not azure_deployment or model != azure_deployment:
            return model, []

    base_url = (
        os.environ.get("AZURE_OPENAI_V1_BASE", "").strip()
        or os.environ.get("AZURE_FOUNDRY_OPENAI_BASE", "").strip()
    )
    if not base_url:
        api_base = os.environ.get("AZURE_API_BASE", "").strip()
        if api_base:
            base_url = api_base.rstrip("/") + "/openai/v1"
    if not base_url:
        raise RuntimeError(
            f"Azure Codex proposer requested ({requested}) but no Azure OpenAI "
            "v1 base URL is configured. Set AZURE_OPENAI_V1_BASE or "
            "AZURE_FOUNDRY_OPENAI_BASE in the azure-openai secret."
        )

    env_key = "AZURE_API_KEY" if os.environ.get("AZURE_API_KEY") else "AZURE_FOUNDRY_API_KEY"
    return deployment, [
        "-c", 'model_provider="azure"',
        "-c", "model_providers.azure.name=" + _toml_string("Azure OpenAI"),
        "-c", "model_providers.azure.base_url=" + _toml_string(base_url.rstrip("/")),
        "-c", 'model_providers.azure.wire_api="responses"',
        "-c", "model_providers.azure.env_key=" + _toml_string(env_key),
        "-c", "model_providers.azure.requires_openai_auth=false",
        "-c", "model_providers.azure.supports_websockets=false",
    ]


def _reasoning_effort_args(model: Optional[str]) -> list[str]:
    """Return explicit Codex reasoning-effort overrides for known GPT-5.5 runs."""
    effort = os.environ.get("CODEX_MODEL_REASONING_EFFORT", "").strip()
    if effort.lower() in {"none", "unset", "off"}:
        return []
    if not effort and model and model.lower().startswith("gpt-5.5"):
        effort = "xhigh"
    if not effort:
        return []
    return ["-c", "model_reasoning_effort=" + _toml_string(effort)]


def build_command(model: Optional[str], *, cwd: Path) -> list[str]:
    """Build the `codex exec` command.

    The prompt is intentionally supplied on stdin via `-`, so long proposer
    briefs do not become enormous argv strings and logs do not duplicate the
    entire prompt.
    """
    codex_danger = os.environ.get("CODEX_DANGEROUS_BYPASS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    model, provider_args = _maybe_configure_azure_provider(model)
    cmd = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--cd",
        str(cwd),
        "--ephemeral",
    ]
    cmd.extend(provider_args)
    cmd.extend(_reasoning_effort_args(model))
    tool_output_token_limit = os.environ.get(
        "CODEX_TOOL_OUTPUT_TOKEN_LIMIT", "4000",
    ).strip()
    if tool_output_token_limit.lower() in {"none", "unset", "off"}:
        tool_output_token_limit = ""
    if tool_output_token_limit:
        cmd.extend(["-c", f"tool_output_token_limit={tool_output_token_limit}"])
    if not codex_danger:
        cmd.append("--full-auto")
    if model:
        cmd.extend(["--model", model])
    codex_sandbox = os.environ.get("CODEX_SANDBOX_MODE", "").strip()
    if codex_sandbox and not codex_danger:
        cmd.extend(["--sandbox", codex_sandbox])
    if codex_danger:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    cmd.append("-")
    return cmd


def _update_captured(event: dict[str, Any], captured: dict[str, Any]) -> None:
    if event.get("type") != "result":
        return
    turns = event.get("num_turns")
    if isinstance(turns, int):
        captured["num_turns"] = turns
    cost = event.get("total_cost_usd")
    if isinstance(cost, (int, float)):
        captured["cost_usd"] = float(cost)
    usage = event.get("usage") or {}
    if isinstance(usage, dict):
        captured["input_tokens"] = usage.get("input_tokens")
        captured["output_tokens"] = usage.get("output_tokens")
        captured["cache_read_tokens"] = usage.get("cache_read_input_tokens")


def run(
    *,
    prompt: str,
    model: Optional[str],
    cwd: Path,
    trace_path: Optional[Path] = None,
    stall_timeout_seconds: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> CodexWrapperResult:
    """Run one non-interactive Codex proposer session."""
    captured: dict[str, Any] = {
        "cost_usd": None,
        "num_turns": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
    }
    stderr_lines: list[str] = []
    try:
        cmd = build_command(model, cwd=cwd)
    except Exception as exc:  # noqa: BLE001 - convert config issues into proposer failure
        return CodexWrapperResult(
            exit_code=1,
            command=[],
            stderr=f"Failed to build codex command: {type(exc).__name__}: {exc}",
            **captured,
        )

    trace_file = None
    if trace_path:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = open(trace_path, "w")
        except OSError as exc:
            stderr_lines.append(
                f"Could not open codex trace {trace_path}: {type(exc).__name__}: {exc}"
            )
    trace_write_failed = False
    last_activity = time.time()
    started_at = last_activity

    try:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr_lines.append(
                f"Failed to start codex process: {type(exc).__name__}: {exc}"
            )
            return CodexWrapperResult(
                exit_code=1,
                command=cmd,
                stderr="\n".join(stderr_lines),
                **captured,
            )
        assert proc.stdin is not None
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            stderr_lines.append(
                f"Failed to write prompt to codex stdin: {type(exc).__name__}: {exc}"
            )

        q: queue.Queue[tuple[str, str]] = queue.Queue()

        def enqueue(stream: Any, name: str) -> None:
            try:
                for line in stream:
                    q.put((name, line))
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        threads = [
            threading.Thread(target=enqueue, args=(proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=enqueue, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        while True:
            now = time.time()
            if timeout_seconds and now - started_at > timeout_seconds:
                proc.kill()
                stderr_lines.append(f"Process timed out after {timeout_seconds:.0f}s.")
                break
            if stall_timeout_seconds and now - last_activity > stall_timeout_seconds:
                proc.kill()
                stderr_lines.append(
                    f"Process stalled for {stall_timeout_seconds:.0f}s without output."
                )
                break
            try:
                stream_name, raw = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    if all(not thread.is_alive() for thread in threads) and q.empty():
                        break
                continue

            line = raw.rstrip("\n")
            if not line:
                continue
            last_activity = time.time()
            if stream_name == "stderr":
                stderr_lines.append(line)
                continue

            if trace_file and not trace_file.closed and not trace_write_failed:
                try:
                    trace_file.write(line + "\n")
                    trace_file.flush()
                except OSError as exc:
                    trace_write_failed = True
                    stderr_lines.append(
                        f"Could not write codex trace {trace_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            _update_captured(event, captured)
            if on_event:
                on_event(event)

        for thread in threads:
            thread.join(timeout=2)
        try:
            exit_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait(timeout=5)
    finally:
        if trace_file:
            trace_file.close()

    return CodexWrapperResult(
        exit_code=exit_code,
        command=cmd,
        stderr="\n".join(stderr_lines),
        **captured,
    )
