from __future__ import annotations

import json
from typing import Any


def flatten_conversation(conversation: list[dict[str, Any]]) -> str:
    """Render tau2 messages as a newline-joined role-tagged transcript.

    Message schema per ``tau2.data_model.message``:

    * assistant: ``{role, content, tool_calls}`` — content may be empty when
      tool_calls populated.
    * user:      ``{role, content}`` — from the user simulator.
    * tool:      ``{role, content, id, error}`` — content is a JSON string.

    We keep only the fields the judge needs (role / content / tool_calls /
    error) and drop every other metadata key (timestamp, usage, raw_data,
    audio_*, speech_*, etc.). Tool-call dicts are serialized inline as
    ``name(args)`` so the judge can see what the agent did, not just what
    it said.
    """
    lines: list[str] = []
    for msg in conversation:
        role = str(msg.get("role", "?")).lower()
        content = msg.get("content")
        if content is None:
            content_str = ""
        elif isinstance(content, list):
            # Some model backends return list-of-blocks; flatten to text.
            chunks: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    if isinstance(c.get("text"), str):
                        chunks.append(c["text"])
                    elif isinstance(c.get("content"), str):
                        chunks.append(c["content"])
            content_str = "\n".join(chunks)
        else:
            content_str = str(content)

        tool_calls = msg.get("tool_calls") or []
        tool_call_strs: list[str] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = str(args)
            tool_call_strs.append(f"{name}({args_str})")

        if role == "tool":
            # Surface tool errors explicitly so the judge can spot failed calls.
            error_tag = "  [ERROR]" if msg.get("error") else ""
            lines.append(f"[TOOL]{error_tag} {content_str}")
            continue

        pieces = [f"[{role.upper()}]"]
        if content_str:
            pieces.append(content_str)
        if tool_call_strs:
            pieces.append("tool_calls: " + " ; ".join(tool_call_strs))
        lines.append(" ".join(pieces))

    return "\n".join(lines)
