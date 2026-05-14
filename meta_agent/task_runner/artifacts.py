"""On-disk artifacts: harness copy-in, trace serialization, result persistence.

One place for every "read from or write to the task workspace" operation.
The runtime adapters don't touch files directly — they call these helpers so
the disk layout stays consistent across runtimes.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from meta_agent.core.targets import TARGETS

from .results import AgentRunResult


def _union_across_file_based(attr: str) -> set[str]:
    """Collect a tuple-attr across all file-based targets."""
    out: set[str] = set()
    for target in TARGETS.values():
        if target.is_file_based:
            out.update(getattr(target, attr))
    return out


_HARNESS_FILES = _union_across_file_based("harness_files")
_HARNESS_GLOBS = _union_across_file_based("harness_globs")
_HARNESS_DIRS = _union_across_file_based("harness_dirs")


def _copy_harness_files(config_dir: str, work_dir: Path) -> None:
    """Copy harness files into the task work directory."""
    src = Path(config_dir)
    if not src.is_dir():
        return

    for name in _HARNESS_FILES:
        f = src / name
        if f.is_file():
            shutil.copy2(f, work_dir / name)

    for pattern in _HARNESS_GLOBS:
        for f in src.glob(pattern):
            if f.is_file():
                shutil.copy2(f, work_dir / f.name)

    for name in _HARNESS_DIRS:
        d = src / name
        if d.is_dir():
            shutil.copytree(d, work_dir / name, dirs_exist_ok=True)


def _ensure_claude_md(work_dir: Path) -> None:
    """Claude Code reads CLAUDE.md, not AGENTS.md.

    If only AGENTS.md exists, synthesize CLAUDE.md that imports it.
    """
    claude_md = work_dir / "CLAUDE.md"
    agents_md = work_dir / "AGENTS.md"
    if claude_md.exists():
        return
    if agents_md.exists():
        claude_md.write_text("@AGENTS.md\n")


def _extract_last_agent_message_from_codex_trace(raw_trace: str) -> Optional[str]:
    """Walk a codex JSONL trace and return the last agent message text.

    Tolerates both legacy `type:"message"` events and newer
    `item.completed` events with nested agent_message items.
    """
    last_message: Optional[str] = None
    for line in raw_trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "message":
            content = event.get("content")
            if isinstance(content, str) and content.strip():
                last_message = content
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            last_message = text
    return last_message


def serialize_block(block: Any) -> dict[str, Any]:
    from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock

    if isinstance(block, TextBlock):
        return {"type": "TextBlock", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "ThinkingBlock", "thinking": block.thinking}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "ToolUseBlock",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        content = block.content
        if isinstance(content, list):
            content = [str(c) if not isinstance(c, (str, dict)) else c for c in content]
        return {
            "type": "ToolResultBlock",
            "tool_use_id": block.tool_use_id,
            "content": content,
            "is_error": block.is_error,
        }
    return {"type": type(block).__name__, "raw": str(block)[:500]}


def serialize_message(message: Any) -> dict[str, Any]:
    from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage, SystemMessage

    msg_type = type(message).__name__
    record: dict[str, Any] = {"type": msg_type, "timestamp": time.time()}

    if isinstance(message, AssistantMessage):
        record["content"] = [serialize_block(b) for b in message.content]
        record["model"] = message.model
        if message.usage:
            record["usage"] = message.usage
    elif isinstance(message, ResultMessage):
        record["subtype"] = message.subtype
        record["is_error"] = message.is_error
        record["num_turns"] = message.num_turns
        record["duration_ms"] = message.duration_ms
        record["total_cost_usd"] = message.total_cost_usd
        record["session_id"] = message.session_id
        record["usage"] = message.usage
        record["result"] = message.result
    elif isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, str):
            record["content"] = content
        elif isinstance(content, list):
            record["content"] = [serialize_block(b) for b in content]
        else:
            record["content"] = str(content)[:500]
    elif isinstance(message, SystemMessage):
        record["subtype"] = message.subtype
    else:
        record["raw"] = str(message)[:500]

    return record


def _persist_agent_run_artifacts(work_dir: Path, result: AgentRunResult) -> None:
    """Write the runtime artifacts produced by an agent run."""
    (work_dir / "trace.jsonl").write_text(result.trace_jsonl)
    (work_dir / "final_response.txt").write_text(result.final_response)
    if result.raw_trace_jsonl:
        (work_dir / "trace.raw.jsonl").write_text(result.raw_trace_jsonl)
    if result.events_jsonl:
        (work_dir / "events.jsonl").write_text(result.events_jsonl)


def _write_agent_result_metadata(work_dir: Path, result: AgentRunResult) -> None:
    """Persist runtime metadata for downstream artifact collection."""
    payload = {
        "exit_code": result.exit_code,
        "stderr": result.stderr,
        "num_turns": result.num_turns,
        "duration_ms": result.duration_ms,
        "total_cost_usd": result.cost_usd,
        "session_id": result.session_id,
        "wall_time_s": result.wall_time_s,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_tokens": result.cache_tokens,
        "hook_failures": result.hook_failures,
        "hook_warnings": result.hook_warnings,
        "metadata": result.metadata,
    }
    (work_dir / "result.json").write_text(json.dumps(payload, indent=2))
