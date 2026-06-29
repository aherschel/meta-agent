"""In-process proposer: write a candidate harness via a direct provider call.

The historical proposers exec an external agent CLI (`claude` or `codex`) and
route it through Bedrock/Azure. Neither CLI is available on every host (e.g. the
ASP fleet), so this module provides a CLI-free proposer that drives the
configured LLM provider directly through `meta_agent.services.llm.invoke_model`.

It works identically on OpenRouter (Chat Completions) and Anthropic (Messages)
because `invoke_model` normalizes both into the same `{"content": [...]}` shape.
A single forced-tool call asks the model to emit the candidate file(s); we write
them into the staging dir and let the outer optimizer validate + smoke-test the
result exactly as it does for the CLI proposers.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from meta_agent.core.targets import AgentTarget
from meta_agent.utils.logging import get_logger

logger = get_logger("loop")


# The forced tool the model must call to deliver its candidate. Anthropic-shaped
# (name + input_schema); `invoke_model` translates it to OpenAI function tools
# for the OpenRouter path.
_WRITE_TOOL_NAME = "write_candidate"


def _write_candidate_tool() -> dict[str, Any]:
    return {
        "name": _WRITE_TOOL_NAME,
        "description": (
            "Deliver the proposed candidate harness. Provide every file the "
            "candidate needs, each with a path relative to the staging dir and "
            "its full text content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path relative to the staging dir, e.g. 'harness.py'.",
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
                "notes": {
                    "type": "object",
                    "description": "Optional proposal_notes.json payload (hypothesis, lever, rationale, risks).",
                },
            },
            "required": ["files"],
        },
    }


def _forced_tool_extra_body() -> dict[str, Any]:
    return {
        "tools": [_write_candidate_tool()],
        "tool_choice": {"type": "tool", "name": _WRITE_TOOL_NAME},
    }


def _seed_excerpt(staging_dir: Path, target: AgentTarget, max_chars: int = 12000) -> str:
    """Return the current (seeded) required-file content, if any, as a base."""
    for filename in target.required_written_files:
        candidate = staging_dir / filename
        if candidate.is_file():
            text = candidate.read_text()
            if len(text) > max_chars:
                text = text[:max_chars] + "\n# … (truncated for the proposer prompt) …\n"
            return f"Current `{filename}` (your starting point):\n```\n{text}\n```\n"
    return "No baseline file is staged; write the candidate from scratch.\n"


def _tool_input_from_response(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == _WRITE_TOOL_NAME:
            value = block.get("input")
            if isinstance(value, dict):
                return value
    return None


def _files_from_text_fallback(response: dict[str, Any], target: AgentTarget) -> list[dict[str, str]]:
    """Best-effort recovery when the model emitted a code block instead of a tool call."""
    from meta_agent.services.llm import extract_text

    text = extract_text(response)
    if not text.strip():
        return []
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL)
    body = match.group(1).strip() if match else text.strip()
    if not body:
        return []
    primary = target.required_written_files[0] if target.required_written_files else "harness.py"
    return [{"path": primary, "content": body + "\n"}]


def _safe_staging_path(staging_dir: Path, rel_path: str) -> Optional[Path]:
    """Resolve `rel_path` strictly inside `staging_dir`; reject traversal/abs paths."""
    cleaned = rel_path.strip().lstrip("/")
    if not cleaned:
        return None
    dest = (staging_dir / cleaned).resolve()
    try:
        dest.relative_to(staging_dir.resolve())
    except ValueError:
        return None
    return dest


def _write_files(staging_dir: Path, files: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel_path = entry.get("path")
        content = entry.get("content")
        if not isinstance(rel_path, str) or not isinstance(content, str):
            continue
        dest = _safe_staging_path(staging_dir, rel_path)
        if dest is None:
            logger.warning(f"in-process proposer: rejecting unsafe path {rel_path!r}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        written.append(rel_path)
    return written


def _usage_int(usage: dict[str, Any], key: str) -> Optional[int]:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def run_inprocess_proposer(
    *,
    prompt: str,
    system_append: str,
    staging_dir: Path,
    target: AgentTarget,
    model: Optional[str],
    trace_path: Optional[Path] = None,
):
    """Run one CLI-free proposer call and stage the resulting candidate file(s).

    Returns a `ProposerRunResult` (imported lazily to avoid a circular import).
    """
    from meta_agent.loop.proposer import ProposerRunResult
    from meta_agent.services.llm import invoke_model, selected_llm_provider

    eval_model = (
        model
        or os.environ.get("META_AGENT_PROPOSER_MODEL", "").strip()
        or os.environ.get("META_AGENT_MODEL", "").strip()
    )
    if not eval_model:
        return ProposerRunResult(
            exit_code=1,
            cli="inprocess",
            model=model,
            stderr=(
                "in-process proposer needs a model: pass --proposer-model or set "
                "META_AGENT_PROPOSER_MODEL / META_AGENT_MODEL."
            ),
        )

    system = (
        "You are the meta-agent harness proposer. Read the contract below and "
        "deliver the candidate by calling the `write_candidate` tool exactly "
        "once. Do not narrate; the tool call is your entire response.\n\n"
        f"{system_append}"
    )
    required = ", ".join(target.required_written_files) or "harness.py"
    user = (
        f"{prompt}\n\n"
        f"## Required output\n"
        f"Call `write_candidate` with a `files` array. At minimum include "
        f"`{target.required_written_files[0] if target.required_written_files else 'harness.py'}` "
        f"(one of: {required}). Each file needs its full text content — no diffs, "
        f"no placeholders. Optionally include `notes` for proposal_notes.json.\n\n"
        f"{_seed_excerpt(staging_dir, target)}"
    )

    logger.info(
        f"Invoking in-process proposer (provider={selected_llm_provider()} model={eval_model})..."
    )

    async def _call() -> dict[str, Any]:
        return await invoke_model(
            model=eval_model,
            messages=[{"role": "user", "content": user}],
            system=system,
            max_tokens=8192,
            temperature=0,
            extra_body=_forced_tool_extra_body(),
        )

    try:
        response = asyncio.run(_call())
    except Exception as exc:  # noqa: BLE001 - proposer infra failure must not crash the loop
        logger.warning(f"in-process proposer call failed: {type(exc).__name__}: {exc}")
        return ProposerRunResult(
            exit_code=1, cli="inprocess", model=eval_model,
            stderr=f"{type(exc).__name__}: {exc}",
        )

    if trace_path is not None:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(response.get("provider_raw", response), indent=2, default=str))
        except OSError as exc:
            logger.warning(f"could not write in-process proposer trace: {type(exc).__name__}: {exc}")

    tool_input = _tool_input_from_response(response)
    files: list[dict[str, Any]] = []
    notes: Any = None
    if tool_input is not None:
        raw_files = tool_input.get("files")
        if isinstance(raw_files, list):
            files = [f for f in raw_files if isinstance(f, dict)]
        notes = tool_input.get("notes")
    if not files:
        files = _files_from_text_fallback(response, target)

    written = _write_files(staging_dir, files)
    if notes is not None and not any(p == "proposal_notes.json" for p in written):
        try:
            (staging_dir / "proposal_notes.json").write_text(json.dumps(notes, indent=2, default=str))
        except OSError:
            pass

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    exit_code = 0 if written else 1
    if not written:
        logger.warning("in-process proposer produced no candidate files")
    return ProposerRunResult(
        exit_code=exit_code,
        cli="inprocess",
        model=eval_model,
        num_turns=1,
        input_tokens=_usage_int(usage, "input_tokens"),
        output_tokens=_usage_int(usage, "output_tokens"),
    )
