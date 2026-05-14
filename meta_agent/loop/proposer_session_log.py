"""Structured proposer session logs derived from CLI stream JSONL traces.

The raw `proposer_trace.jsonl` remains the canonical event stream. This module
adds a Stanford Meta-Harness-style navigation layer: compact metadata, response
text, raw events, extracted JSON artifacts, and one readable file per tool-like
event.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class _ToolEvent:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    is_error: bool = False


_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rate_limit", re.compile(r"\b(429|too many requests|rate[\s_-]*limit|throttl)", re.I)),
    ("quota", re.compile(r"\b(quota|insufficient_quota|billing)", re.I)),
    ("stream_disconnect", re.compile(r"\b(response\.failed|stream disconnected|connection reset|eof)", re.I)),
    ("timeout", re.compile(r"\b(timeout|timed out|stalled)", re.I)),
    ("provider_error", re.compile(r"\b(azure|openai|responses?|server error|5\d\d)", re.I)),
)

_SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|authorization|bearer|token|secret|password)", re.I)


def _safe_slug(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return out.strip("_")[:80] or "proposer"


def _read_events(trace_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not trace_path.is_file():
        return events
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "raw_text", "text": line})
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _redact_value(value: Any, key_hint: str = "") -> Any:
    if _SENSITIVE_KEY_RE.search(key_hint):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, key_hint) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(api[_-]?key|authorization|bearer|token|secret|password)"
            r"([\"'\s:=]+)[^\"'\s,)}]+",
            r"\1\2[REDACTED]",
            value,
        )
    return value


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _event_message(event: dict[str, Any]) -> str:
    parts: list[str] = []
    message = event.get("message")
    if isinstance(message, str):
        parts.append(message)
    error = event.get("error")
    if isinstance(error, dict):
        for key in ("type", "code", "message", "param"):
            value = error.get(key)
            if value:
                parts.append(f"{key}={value}")
    elif isinstance(error, str):
        parts.append(error)
    if not parts:
        try:
            parts.append(json.dumps(event, default=str)[:1000])
        except TypeError:
            parts.append(str(event)[:1000])
    return " | ".join(parts)


def _classify_failure_text(text: str) -> list[str]:
    labels: list[str] = []
    for label, pattern in _FAILURE_PATTERNS:
        if pattern.search(text):
            labels.append(label)
    return labels


def _error_diagnostics(
    events: list[dict[str, Any]],
    stderr: Optional[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build compact failure diagnostics without replacing the raw trace."""
    interesting_records: list[dict[str, Any]] = []
    messages: list[str] = []
    labels: dict[str, int] = {}
    error_event_count = 0
    turn_failed_count = 0
    reconnect_count = 0
    response_failed_count = 0

    def add_label_counts(text: str) -> None:
        for label in _classify_failure_text(text):
            labels[label] = labels.get(label, 0) + 1

    for idx, event in enumerate(events):
        etype = str(event.get("type") or "")
        message = _event_message(event)
        is_error_event = etype in {"error", "turn.failed"}
        has_failure_text = bool(_classify_failure_text(message))
        if not is_error_event and not has_failure_text:
            continue
        if etype == "error":
            error_event_count += 1
        if etype == "turn.failed":
            turn_failed_count += 1
        if "Reconnecting..." in message:
            reconnect_count += 1
        if "response.failed" in message:
            response_failed_count += 1
        add_label_counts(message)
        if message and message not in messages:
            messages.append(message)
        interesting_records.append({
            "source": "event",
            "index": idx,
            "type": etype,
            "message": message,
            "event": _redact_value(event),
        })

    stderr_lines = stderr.splitlines() if stderr else []
    interesting_stderr: list[str] = []
    for idx, line in enumerate(stderr_lines):
        if _classify_failure_text(line) or "error" in line.lower() or "failed" in line.lower():
            add_label_counts(line)
            redacted = str(_redact_value(line))
            interesting_stderr.append(redacted)
            interesting_records.append({
                "source": "stderr",
                "index": idx,
                "message": redacted,
            })

    diagnostics = {
        "error_event_count": error_event_count,
        "turn_failed_count": turn_failed_count,
        "reconnect_count": reconnect_count,
        "response_failed_count": response_failed_count,
        "stderr_line_count": len(stderr_lines),
        "interesting_stderr_count": len(interesting_stderr),
        "failure_label_counts": labels,
        "messages": messages[:25],
        "interesting_stderr": interesting_stderr[:25],
        "stderr_tail": _tail_text(str(_redact_value(stderr or ""))),
    }
    return diagnostics, interesting_records


def _extract_json_blocks(text: str) -> list[tuple[Optional[str], Any]]:
    blocks: list[tuple[Optional[str], Any]] = []
    pattern = re.compile(
        r"(?:\*\*`?([^`*\n]+\.json)`?\*\*[: \t]*\n)?"
        r"```json\s*\n(.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        name_hint = match.group(1)
        body = match.group(2).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        blocks.append((Path(name_hint).name if name_hint else None, parsed))
    return blocks


def _tool_arg_summary(tool: _ToolEvent) -> str:
    for key in ("file_path", "path", "pattern", "command", "description", "prompt"):
        value = tool.input.get(key)
        if isinstance(value, str) and value:
            return value[:160]
    return ""


def _classify_file_access(path: str) -> str:
    p = path.replace("\\", "/")
    name = Path(p).name
    if (
        name in {"harness.py", "AGENTS.md", "CLAUDE.md"}
        or "/.codex/" in p
        or "/.claude/" in p
        or "/harnesses/" in p
    ):
        return "harness_source"
    if (
        name.endswith("_trace.jsonl")
        or name.endswith("_trace.raw.jsonl")
        or name in {"trace.jsonl", "trace.raw.jsonl", "log.jsonl"}
        or "/per_task/" in p
    ):
        return "execution_trace"
    if name in {
        "scores.json",
        "summary.md",
        "category_scores.json",
        "candidate_index.json",
        "frontier.json",
        "history.json",
        "evolution_summary.jsonl",
        "epoch_meta.json",
        "proposal_notes.json",
        "proposal_manifest.json",
    }:
        return "score_summary"
    return "other"


def _file_access_stats(
    files_read: dict[str, dict[str, int]],
    files_written: dict[str, dict[str, int]],
) -> dict[str, Any]:
    read_breakdown: dict[str, int] = {
        "harness_source": 0,
        "execution_trace": 0,
        "score_summary": 0,
        "other": 0,
    }
    for path in files_read:
        read_breakdown[_classify_file_access(path)] += 1
    total_reads = sum(read_breakdown.values())
    return {
        "files_read_count": len(files_read),
        "files_written_count": len(files_written),
        "read_breakdown_counts": read_breakdown,
        "read_breakdown_fraction": {
            key: (value / total_reads if total_reads else 0.0)
            for key, value in read_breakdown.items()
        },
    }


def _parse_events(events: list[dict[str, Any]]) -> tuple[str, list[_ToolEvent], dict[str, Any], dict[str, Any], dict[str, int]]:
    text_parts: list[str] = []
    tools: list[_ToolEvent] = []
    tool_by_id: dict[str, _ToolEvent] = {}
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    usage: dict[str, int] = {}

    def add_usage(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = payload.get(key)
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value

    def note_read(path: str, output: str = "") -> None:
        entry = files_read.setdefault(path, {"reads": 0, "lines": 0})
        entry["reads"] += 1
        entry["lines"] += len(output.splitlines()) if output else 0

    def note_write(path: str, content: str = "") -> None:
        entry = files_written.setdefault(path, {"writes": 0, "lines": 0})
        entry["writes"] += 1
        entry["lines"] += content.count("\n") + (1 if content else 0)

    for event in events:
        etype = event.get("type")
        if etype == "assistant":
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            add_usage(message.get("usage"))
            for block in message.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool = _ToolEvent(
                        name=str(block.get("name") or "tool"),
                        input=block.get("input") if isinstance(block.get("input"), dict) else {},
                    )
                    tools.append(tool)
                    tool_id = block.get("id")
                    if isinstance(tool_id, str):
                        tool_by_id[tool_id] = tool
        elif etype == "user":
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            for block in message.get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool = tool_by_id.get(str(block.get("tool_use_id") or ""))
                if tool is None:
                    continue
                tool.output = str(block.get("content") or "")
                tool.is_error = bool(block.get("is_error"))
        elif etype == "item.completed":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            item_type = item.get("type")
            if item_type == "agent_message" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif item_type in {"command_execution", "file_change"}:
                tool = _ToolEvent(name=str(item_type), input=item)
                for key in ("output", "stdout", "stderr", "text"):
                    if isinstance(item.get(key), str):
                        tool.output += item[key]
                exit_code = item.get("exit_code")
                tool.is_error = isinstance(exit_code, int) and exit_code != 0
                tools.append(tool)
                if item_type == "file_change":
                    for change in item.get("changes", []) or []:
                        if isinstance(change, dict) and isinstance(change.get("path"), str):
                            note_write(change["path"])
        elif etype in {"turn.completed", "result"}:
            add_usage(event.get("usage"))

    for tool in tools:
        path = tool.input.get("file_path")
        if isinstance(path, str) and path:
            if tool.name == "Read":
                note_read(path, tool.output)
            elif tool.name in {"Write", "Edit"}:
                note_write(path, str(tool.input.get("content") or tool.input.get("new_string") or ""))

    return "".join(text_parts), tools, files_read, files_written, usage


def write_proposer_session_log(
    *,
    trace_path: Path,
    session_dir: Path,
    prompt: str,
    cli: str,
    model: Optional[str],
    cwd: Path,
    exit_code: int,
    cost_usd: Optional[float],
    num_turns: Optional[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_read_tokens: Optional[int],
    command: list[str],
    stderr: Optional[str] = None,
) -> None:
    """Write a structured proposer-session directory from a raw trace."""
    events = _read_events(trace_path)
    text, tools, files_read, files_written, usage = _parse_events(events)
    error_diagnostics, error_records = _error_diagnostics(events, stderr)
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if cache_read_tokens is not None:
        usage["cache_read_input_tokens"] = cache_read_tokens

    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "cli": cli,
        "model": model,
        "exit_code": exit_code,
        "cost_usd": cost_usd,
        "num_turns": num_turns,
        "token_usage": usage,
        "cwd": str(cwd),
        "trace_path": str(trace_path),
        "command": command,
        "files_read": files_read,
        "files_written": files_written,
        "file_access_stats": _file_access_stats(files_read, files_written),
        "error_diagnostics": error_diagnostics,
        "tool_summary": [
            f"{tool.name}({'ERR ' if tool.is_error else ''}{_tool_arg_summary(tool)})"
            for tool in tools
        ],
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if text:
        (session_dir / "response.md").write_text(text)

    if events:
        (session_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, default=str) + "\n" for event in events)
        )

    if stderr:
        (session_dir / "stderr.txt").write_text(str(_redact_value(stderr)) + "\n")

    if error_records:
        (session_dir / "errors.jsonl").write_text(
            "".join(json.dumps(record, default=str) + "\n" for record in error_records)
        )

    json_blocks = _extract_json_blocks(text)
    if json_blocks:
        artifacts_dir = session_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for idx, (name, value) in enumerate(json_blocks, 1):
            filename = _safe_slug(name or f"{idx:03d}.json")
            (artifacts_dir / filename).write_text(json.dumps(value, indent=2) + "\n")

    if tools:
        tools_dir = session_dir / "tools"
        tools_dir.mkdir(exist_ok=True)
        for idx, tool in enumerate(tools, 1):
            parts = [f"{tool.name}: {_tool_arg_summary(tool)}".rstrip(), ""]
            for key, value in tool.input.items():
                rendered = json.dumps(value, indent=2, default=str) if isinstance(value, (dict, list)) else str(value)
                parts.append(f"{key}:")
                parts.append(rendered)
                parts.append("")
            if tool.output:
                parts.append("--- output ---")
                parts.append(tool.output)
            filename = f"{idx:03d}_{_safe_slug(tool.name)}.txt"
            (tools_dir / filename).write_text("\n".join(parts))
