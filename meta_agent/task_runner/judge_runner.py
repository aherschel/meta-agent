"""Shared judge-via-SDK runner for pairwise + best-of-N judge benchmarks.

Runs a proposer's `build_options(ctx) -> ClaudeAgentOptions` harness as a
preference judge. Two public drivers share one underlying primitive
(`_run_one_ordering`):

* `run_judge_benchmark` — pairwise with optional position-swap.
* `run_bestofn_benchmark` — best-of-N tournament scoring: for each
  sample, judge picks chosen vs each rejected in N-1 pairwise comparisons
  and the sample passes iff chosen wins every one.

This module is the judge-benchmark's *exit contract*: it injects the
required `submit_verdict` MCP server, extends `allowed_tools`, prepends a
minimal "call `submit_verdict` when done" system prompt, and collects the
result. The proposer owns all other behavior (system prompt, additional
tools, hooks, subagents, permission mode, thinking, max_turns).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, cast

from claude_agent_sdk import (
    HookMatcher,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from meta_agent.harness_contracts.claude_agent_sdk import (
    append_hooks,
    build_claude_agent_options,
    extend_allowed_tools,
    merge_mcp_server,
    prepend_system_prompt,
    set_default_max_turns,
)
from meta_agent.services.llm import (
    ensure_agent_sdk_env as ensure_bedrock_env,
    resolve_model_for_provider as resolve_bedrock_model,
)
from meta_agent.harness_contracts.program import (
    HarnessContext,
    run_program_harness,
)
from meta_agent.core.run_context import RunContext
from meta_agent.task_runner import TaskResult
from meta_agent.task_runner.artifacts import serialize_message


JudgeDecision = Literal["A>B", "B>A"]
"""The two allowed verdicts. Judge benchmarks have no ties."""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime diagnostics — preserve stderr + classify failures
# ---------------------------------------------------------------------------

_HOOK_ERROR_RE = re.compile(r"hook_stderr|Error in hook callback|ZodError", re.IGNORECASE)
_TRANSIENT_RUNTIME_RE = re.compile(
    r"Fatal error in message reader|API Error:|unexpected error during processing",
    re.IGNORECASE,
)
_TRANSIENT_RUNTIME_MAX_RETRIES = 1
_DIAGNOSTIC_LINE_CHAR_LIMIT = 240
_DIAGNOSTIC_SUMMARY_LINE_LIMIT = 4


def _normalize_diagnostic_line(line: str) -> Optional[str]:
    """Collapse whitespace and cap noisy stderr/result lines for traces."""
    text = " ".join(line.strip().split())
    if not text:
        return None
    if len(text) <= _DIAGNOSTIC_LINE_CHAR_LIMIT:
        return text
    return text[: _DIAGNOSTIC_LINE_CHAR_LIMIT - 3] + "..."


def _dedupe_lines(lines: list[str]) -> list[str]:
    """Preserve order while removing duplicate diagnostic lines."""
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _extract_event_error_lines(events: list[dict[str, Any]]) -> list[str]:
    """Surface synthetic API/runtime errors that may not land in stderr."""
    out: list[str] = []
    for event in events:
        result = event.get("result")
        if isinstance(result, str):
            normalized = _normalize_diagnostic_line(result)
            if normalized is not None and _TRANSIENT_RUNTIME_RE.search(normalized):
                out.append(normalized)

        if event.get("type") != "AssistantMessage":
            continue
        content = event.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "TextBlock":
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            normalized = _normalize_diagnostic_line(text)
            if normalized is not None and _TRANSIENT_RUNTIME_RE.search(normalized):
                out.append(normalized)
    return _dedupe_lines(out)


def _classify_runtime_diagnostics(
    err: Optional[str],
    stderr_lines: list[str],
    events: list[dict[str, Any]],
) -> tuple[Optional[str], list[str], list[str]]:
    """Classify captured runtime evidence for trace/debug purposes."""
    normalized_stderr = _dedupe_lines([
        normalized
        for line in stderr_lines
        if (normalized := _normalize_diagnostic_line(line)) is not None
    ])
    event_error_lines = _extract_event_error_lines(events)

    context_lines: list[str] = []
    if err:
        normalized_err = _normalize_diagnostic_line(err)
        if normalized_err is not None:
            context_lines.append(normalized_err)
    context_lines.extend(normalized_stderr)
    context_lines.extend(event_error_lines)

    if any(_HOOK_ERROR_RE.search(line) for line in context_lines):
        return "hook_error", normalized_stderr, event_error_lines
    if any(_TRANSIENT_RUNTIME_RE.search(line) for line in context_lines):
        return "transient_runtime", normalized_stderr, event_error_lines
    if normalized_stderr:
        return "stderr_other", normalized_stderr, event_error_lines
    return None, normalized_stderr, event_error_lines


def _select_diagnostic_summary_lines(
    classification: str,
    err: Optional[str],
    stderr_lines: list[str],
    event_error_lines: list[str],
) -> list[str]:
    """Pick the most informative subset of lines for the compact error string."""
    candidates: list[str] = []
    if err:
        normalized_err = _normalize_diagnostic_line(err)
        if normalized_err is not None:
            candidates.append(normalized_err)
    candidates.extend(stderr_lines)
    candidates.extend(event_error_lines)
    deduped = _dedupe_lines(candidates)

    matcher = None
    if classification == "hook_error":
        matcher = _HOOK_ERROR_RE
    elif classification == "transient_runtime":
        matcher = _TRANSIENT_RUNTIME_RE

    if matcher is not None:
        matched = [line for line in deduped if matcher.search(line)]
        if matched:
            return matched[:_DIAGNOSTIC_SUMMARY_LINE_LIMIT]
    return deduped[:_DIAGNOSTIC_SUMMARY_LINE_LIMIT]


def _merge_runtime_diagnostics(
    err: Optional[str],
    stderr_lines: list[str],
    events: list[dict[str, Any]],
) -> Optional[str]:
    """Attach classified stderr/runtime context to the recorded error string."""
    classification, normalized_stderr, event_error_lines = _classify_runtime_diagnostics(
        err, stderr_lines, events
    )
    if not normalized_stderr and not event_error_lines and classification is None:
        return err

    events.append({
        "type": "runtime_diagnostics",
        "classification": classification or "stderr_other",
        "stderr_lines": normalized_stderr,
        "event_error_lines": event_error_lines,
    })

    summary_lines = _select_diagnostic_summary_lines(
        classification or "stderr_other",
        err,
        normalized_stderr,
        event_error_lines,
    )
    if not summary_lines:
        return err

    summary = f"{classification or 'stderr_other'}: " + " | ".join(summary_lines)
    if err:
        return err if summary in err else f"{err}; {summary}"
    if classification == "hook_error":
        return summary
    return None


# ---------------------------------------------------------------------------
# Input + per-ordering / per-pair outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgePair:
    """One preference pair the judge must score."""

    pair_id: str
    question: str
    response_a: str
    response_b: str
    gold: JudgeDecision
    source: str = ""
    category: str = "Other"


@dataclass(frozen=True)
class ProgramJudgeTask:
    """Safe task object exposed to program-harness judges.

    Deliberately excludes the gold label. The adapter still owns position
    swaps, scoring, and trace persistence; the candidate-owned program owns
    how to render evidence, call models/tools, and choose a final decision.
    """

    name: str
    pair_id: str
    question: str
    response_a: str
    response_b: str
    source: str = ""
    category: str = "Other"
    ordering_label: str = "original"
    response_a_ref: str = "response_a_original"
    response_b_ref: str = "response_b_original"

    def as_prompt(self) -> str:
        return _JUDGE_PAIR_PROMPT_TEMPLATE.format(
            question=self.question,
            response_a=self.response_a,
            response_b=self.response_b,
        )


@dataclass
class _OrderingOutcome:
    label: str                               # "original" or "swapped"
    decision_raw: Optional[JudgeDecision]    # what submit_verdict reported
    decision_final: Optional[JudgeDecision]  # raw for original; flipped for swapped
    wall_time_s: float
    prompt: str = ""
    response_a_ref: str = ""
    response_b_ref: str = ""
    options_snapshot: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PairOutcome:
    pair_id: str
    gold: JudgeDecision
    source: str
    category: str
    orderings: List[_OrderingOutcome]
    wall_time_s: float
    question: str = ""
    response_a_original: str = ""
    response_b_original: str = ""

    @property
    def decisions(self) -> List[JudgeDecision]:
        return [o.decision_final for o in self.orderings if o.decision_final is not None]

    @property
    def error(self) -> Optional[str]:
        errs = [f"{o.label}: {o.error}" for o in self.orderings if o.error]
        return "; ".join(errs) if errs else None


def _ordering_runtime_classification(outcome: _OrderingOutcome) -> Optional[str]:
    """Return the classified runtime failure for an ordering, if any."""
    for event in reversed(outcome.events):
        if event.get("type") != "runtime_diagnostics":
            continue
        classification = event.get("classification")
        return classification if isinstance(classification, str) else None
    return None


def _should_retry_ordering(outcome: _OrderingOutcome) -> bool:
    """Retry only SDK/backend crashes that were classified as transient."""
    return (
        outcome.error is not None
        and _ordering_runtime_classification(outcome) == "transient_runtime"
    )


def _merge_ordering_attempts(attempts: list[_OrderingOutcome]) -> _OrderingOutcome:
    """Collapse retry attempts into one ordering outcome for scoring/traces."""
    if len(attempts) == 1:
        return attempts[0]

    final_attempt = attempts[-1]
    combined_events: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        if index > 1:
            combined_events.append({
                "type": "retry",
                "attempt": index,
                "reason": "transient_runtime",
                "previous_error": attempts[index - 2].error,
            })
        for event in attempt.events:
            tagged_event = dict(event)
            tagged_event["attempt"] = index
            combined_events.append(tagged_event)

    total_cost = sum(
        (attempt.cost_usd or 0.0) for attempt in attempts if attempt.cost_usd is not None
    ) or None
    total_input = sum(
        (attempt.input_tokens or 0) for attempt in attempts if attempt.input_tokens is not None
    ) or None
    total_output = sum(
        (attempt.output_tokens or 0) for attempt in attempts if attempt.output_tokens is not None
    ) or None
    total_cache = sum(
        (attempt.cache_tokens or 0) for attempt in attempts if attempt.cache_tokens is not None
    ) or None
    total_turns = sum(
        (attempt.num_turns or 0) for attempt in attempts if attempt.num_turns is not None
    ) or None

    return _OrderingOutcome(
        label=final_attempt.label,
        decision_raw=final_attempt.decision_raw,
        decision_final=final_attempt.decision_final,
        wall_time_s=sum(attempt.wall_time_s for attempt in attempts),
        prompt=final_attempt.prompt,
        response_a_ref=final_attempt.response_a_ref,
        response_b_ref=final_attempt.response_b_ref,
        options_snapshot=final_attempt.options_snapshot,
        error=final_attempt.error,
        num_turns=total_turns,
        cost_usd=total_cost,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_tokens=total_cache,
        session_id=final_attempt.session_id,
        events=combined_events,
    )


def _flip(decision: JudgeDecision) -> JudgeDecision:
    return "B>A" if decision == "A>B" else "A>B"


def _normalize_judge_decision(value: Any) -> Optional[JudgeDecision]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("decision", "choice", "verdict", "preference"):
            if key in value:
                normalized = _normalize_judge_decision(value.get(key))
                if normalized is not None:
                    return normalized
        return None
    text = str(value).strip().upper().replace(" ", "")
    if text in ("A>B", "B>A"):
        return cast(JudgeDecision, text)
    if text in ("A", "[[A]]"):
        return "A>B"
    if text in ("B", "[[B]]"):
        return "B>A"
    for token in ("A>B", "B>A", "[[A]]", "[[B]]"):
        if token in text:
            return "A>B" if token in ("A>B", "[[A]]") else "B>A"
    return None


def _decision_from_program_result(result: Any) -> Optional[JudgeDecision]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        normalized = _normalize_judge_decision(metadata)
        if normalized is not None:
            return normalized
    return _normalize_judge_decision(getattr(result, "final_output", result))


def _position_consistent_correct(outcome: _PairOutcome) -> bool:
    if outcome.error is not None or not outcome.decisions:
        return False
    return all(d == outcome.gold for d in outcome.decisions)


# ---------------------------------------------------------------------------
# Benchmark-injected submit_verdict MCP tool
# ---------------------------------------------------------------------------


@dataclass
class _VerdictState:
    """Mutable slot the submit_verdict tool writes into once per call.

    `stop_hook_fired` distinguishes "agent stopped naturally after the tool
    call" from "the benchmark's PostToolUse hook forced the stop". Surfacing
    it in the per-pair verdict event lets us verify, across a run, that the
    stop contract is enforced by the hook rather than by model politeness —
    if this ever goes False on healthy pairs, the hook isn't wired correctly
    and the only thing capping turns is `max_turns`.
    """

    decision: Optional[JudgeDecision] = None
    rationale: str = ""
    call_count: int = 0
    stop_hook_fired: bool = False


def _make_submit_verdict_server(state: _VerdictState) -> Any:
    """Create a single-tool MCP server that the agent uses to finalize its verdict."""

    @tool(
        "submit_verdict",
        (
            "Submit your final pairwise preference. Call this exactly once when "
            "you've decided which response better answers the user. 'A>B' means "
            "Response A is preferred; 'B>A' means Response B is preferred."
        ),
        {"choice": str, "rationale": str},
    )
    async def submit_verdict(args: dict[str, Any]) -> dict[str, Any]:
        state.call_count += 1
        raw = str(args.get("choice", "")).strip().upper().replace(" ", "")
        if raw not in ("A>B", "B>A"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Invalid choice {raw!r}. Must be exactly 'A>B' or 'B>A'. "
                            "Call submit_verdict again with a valid choice."
                        ),
                    }
                ]
            }
        state.decision = cast(JudgeDecision, raw)
        state.rationale = str(args.get("rationale") or "")
        return {"content": [{"type": "text", "text": f"Verdict recorded: {raw}."}]}

    return create_sdk_mcp_server(name="judge", tools=[submit_verdict])


_SUBMIT_VERDICT_TOOL_NAME = "mcp__judge__submit_verdict"

# Benchmark-enforced default. The submit_verdict stop-hook terminates the
# agent as soon as it fires, so this is a safety ceiling, not a tight bound.
# Proposers can set a lower max_turns in `build_options` to experiment; they
# can also raise it to explore agentic judges that reason across multiple
# tool calls before submitting.
_JUDGE_DEFAULT_MAX_TURNS = 8


_JUDGE_TASK_PROMPT_TEMPLATE = (
    "You are an impartial pairwise preference judge.\n\n"
    "Read the user's question and the two candidate responses. Decide which "
    "response a careful human evaluator would prefer. When you've decided, call "
    "the `submit_verdict` tool with choice='A>B' or choice='B>A' and a brief "
    "rationale. Calling `submit_verdict` ends the task — the runtime stops the "
    "agent as soon as the tool fires, so do not plan any work after the call."
)


_JUDGE_PAIR_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Decide which response is preferred, then call `submit_verdict`."
)


def _trace_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Best-effort JSON representation for SDK option values.

    Trace snapshots should explain the runtime envelope without requiring the
    SDK's rich objects (MCP servers, hooks, plugins) to be JSON serializable.
    """
    if depth > 4:
        return {"type": type(value).__name__, "repr": repr(value)[:500]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_trace_jsonable(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(k): _trace_jsonable(v, depth=depth + 1)
            for k, v in value.items()
        }
    if callable(value):
        return {"type": type(value).__name__, "callable": getattr(value, "__name__", repr(value))}
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"type": type(value).__name__}
        for f in fields(value):
            try:
                out[f.name] = _trace_jsonable(getattr(value, f.name), depth=depth + 1)
            except Exception as exc:  # noqa: BLE001 - trace serialization must not break eval
                out[f.name] = {"error": f"{type(exc).__name__}: {exc}"}
        return out
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def _snapshot_mcp_servers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(name): {
                "type": type(server).__name__,
                "repr": repr(server)[:300],
            }
            for name, server in value.items()
        }
    return _trace_jsonable(value)


def _snapshot_hooks(value: Any) -> Any:
    if not isinstance(value, dict):
        return _trace_jsonable(value)
    return {
        str(event): [
            {
                "type": type(matcher).__name__,
                "repr": repr(matcher)[:300],
            }
            for matcher in matchers
        ]
        for event, matchers in value.items()
    }


def _snapshot_agents(value: Any) -> Any:
    if not isinstance(value, dict):
        return _trace_jsonable(value)
    return {
        str(name): _trace_jsonable(agent)
        for name, agent in value.items()
    }


def _snapshot_claude_options(options: Any) -> dict[str, Any]:
    """Persist the effective ClaudeAgentOptions envelope used for query()."""
    env = getattr(options, "env", None)
    return {
        "cwd": _trace_jsonable(getattr(options, "cwd", None)),
        "model": getattr(options, "model", None),
        "fallback_model": getattr(options, "fallback_model", None),
        "system_prompt": _trace_jsonable(getattr(options, "system_prompt", None)),
        "tools": _trace_jsonable(getattr(options, "tools", None)),
        "allowed_tools": _trace_jsonable(getattr(options, "allowed_tools", None)),
        "disallowed_tools": _trace_jsonable(getattr(options, "disallowed_tools", None)),
        "mcp_servers": _snapshot_mcp_servers(getattr(options, "mcp_servers", None)),
        "permission_mode": getattr(options, "permission_mode", None),
        "permission_prompt_tool_name": getattr(options, "permission_prompt_tool_name", None),
        "max_turns": getattr(options, "max_turns", None),
        "max_budget_usd": getattr(options, "max_budget_usd", None),
        "thinking": _trace_jsonable(getattr(options, "thinking", None)),
        "max_thinking_tokens": getattr(options, "max_thinking_tokens", None),
        "effort": getattr(options, "effort", None),
        "hooks": _snapshot_hooks(getattr(options, "hooks", None)),
        "agents": _snapshot_agents(getattr(options, "agents", None)),
        "betas": _trace_jsonable(getattr(options, "betas", None)),
        "settings": getattr(options, "settings", None),
        "add_dirs": _trace_jsonable(getattr(options, "add_dirs", None)),
        "extra_args": _trace_jsonable(getattr(options, "extra_args", None)),
        "env_keys": sorted(env.keys()) if isinstance(env, dict) else [],
        "include_partial_messages": getattr(options, "include_partial_messages", None),
        "fork_session": getattr(options, "fork_session", None),
        "setting_sources": _trace_jsonable(getattr(options, "setting_sources", None)),
        "sandbox": _trace_jsonable(getattr(options, "sandbox", None)),
        "plugins": _trace_jsonable(getattr(options, "plugins", None)),
        "output_format": _trace_jsonable(getattr(options, "output_format", None)),
        "task_budget": _trace_jsonable(getattr(options, "task_budget", None)),
        "enable_file_checkpointing": getattr(options, "enable_file_checkpointing", None),
        "can_use_tool_configured": getattr(options, "can_use_tool", None) is not None,
        "stderr_configured": getattr(options, "stderr", None) is not None,
    }


# ---------------------------------------------------------------------------
# Best-of-N rankings exit contract (free-text [[A]]..[[D]] extraction)
# ---------------------------------------------------------------------------
#
# Matches `rewardbench/generative_v2.py::prompt_v2` and `MTBENCH_V2[
# "prompt_template"]` *verbatim*, and the `process_judgement` substring
# priority rule (first match in order A > B > C > D wins). This is the
# leaderboard-native protocol for every generative LM-as-judge entry on
# RewardBench 2, including Gemini-2.5-Flash, Claude-Opus-4, GPT-4o.
#
# No MCP tool. No PostToolUse hook. The judge generates free text
# ending in "[[X]]"; we extract X via substring check in priority order.


def _letters_for(n: int) -> tuple[str, ...]:
    """Return the first n uppercase letters ('A',..,chr(64+n))."""
    if n < 2 or n > 26:
        raise ValueError(f"rankings mode supports 2..26 choices; got {n}")
    return tuple(chr(ord("A") + i) for i in range(n))


def _extract_rankings_verdict(
    text: str, valid_letters: tuple[str, ...]
) -> Optional[str]:
    """Extract the judge's verdict letter from free-form output.

    Matches `rewardbench/generative_v2.py::process_judgement` semantics
    exactly: substring check in priority order (A > B > C > D). If the
    judge output contains multiple bracket tokens, the *lowest-letter*
    one wins — this is the leaderboard rule. Format error returns None.
    """
    for letter in valid_letters:
        if f"[[{letter}]]" in text:
            return letter
    return None


# Verbatim `prompt_v2` from `rewardbench/generative_v2.py`. Do NOT edit
# this string — matching the leaderboard byte-for-byte is the point.
_REWARDBENCH2_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the four responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is best.'
)


# Verbatim `MTBENCH_V2["prompt_template"]` from the same file.
_REWARDBENCH2_USER_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]\n\n"
    "[The Start of Assistant C's Answer]\n{answer_c}\n[The End of Assistant C's Answer]\n\n"
    "[The Start of Assistant D's Answer]\n{answer_d}\n[The End of Assistant D's Answer]"
)


# Max turns for rankings mode: 2 is enough (one assistant response, plus
# a safety buffer). Rankings judgment is a single-call protocol — no tool
# use, no multi-turn reasoning required. Proposers can override upward if
# they add lifecycle hooks or CoT that needs more turns.
_RANKINGS_DEFAULT_MAX_TURNS = 2


def _make_stop_after_verdict_hook(state: "_VerdictState") -> HookMatcher:
    """PostToolUse hook that terminates the agent when submit_verdict fires.

    Returns `continue_=False` to halt the SDK query loop once a valid verdict
    has been recorded. If the agent called the tool with a malformed choice,
    we let it retry (the handler returned an error message, decision stays
    None), so we only stop when `state.decision` is populated.

    `matcher=None` so the hook fires for every PostToolUse event. The
    SDK's HookMatcher.matcher matches against tool names — with MCP tools
    the exact name format (prefixed, unprefixed, or regex-anchored) is not
    portable across SDK versions, so we guard with an explicit tool-name
    check inside the hook instead.
    """

    async def _on_post_tool_use(
        input_data: dict[str, Any],
        tool_use_id: Optional[str],
        context: Any,
    ) -> dict[str, Any]:
        tool_name = str(input_data.get("tool_name") or "")
        if tool_name != _SUBMIT_VERDICT_TOOL_NAME:
            return {}
        if state.decision is None:
            return {}
        state.stop_hook_fired = True
        return {"continue_": False, "stopReason": "verdict submitted"}

    return HookMatcher(matcher=None, hooks=[_on_post_tool_use])


# ---------------------------------------------------------------------------
# Per-ordering execution
# ---------------------------------------------------------------------------


async def _run_one_ordering_once(
    *,
    harness_path: Path,
    label: str,
    pair: JudgePair,
    flip_responses: bool,
    model: str,
    timeout: int,
) -> _OrderingOutcome:
    """Run one ordering of one pair through the proposer's harness once."""
    ensure_bedrock_env()
    start = time.time()
    state = _VerdictState()
    events: list[dict[str, Any]] = []

    response_a = pair.response_b if flip_responses else pair.response_a
    response_b = pair.response_a if flip_responses else pair.response_b
    response_a_ref = "response_b_original" if flip_responses else "response_a_original"
    response_b_ref = "response_a_original" if flip_responses else "response_b_original"

    pair_prompt = _JUDGE_PAIR_PROMPT_TEMPLATE.format(
        question=pair.question,
        response_a=response_a,
        response_b=response_b,
    )

    # Fresh cwd per ordering so any filesystem tool the proposer enables can't
    # leak state across pairs. The SDK requires cwd to exist as a real dir.
    cwd = Path(tempfile.mkdtemp(prefix=f"judge_{pair.pair_id}_{label}_"))
    resolved_model = resolve_bedrock_model(model)
    ctx = RunContext(cwd=str(cwd), model=resolved_model, task_instruction=pair_prompt)

    err: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    stderr_lines: list[str] = []
    options_snapshot: dict[str, Any] = {}

    try:
        options = build_claude_agent_options(harness_path, ctx)
        options.stderr = stderr_lines.append
        server = _make_submit_verdict_server(state)

        # Benchmark exit contract: inject the required tool + MCP server,
        # pick a sensible max_turns default if the proposer left it unset,
        # prepend the task contract to system_prompt, and force-terminate
        # the agent loop as soon as a valid verdict is submitted.
        merge_mcp_server(options, "judge", server)
        extend_allowed_tools(options, [_SUBMIT_VERDICT_TOOL_NAME])
        prepend_system_prompt(options, _JUDGE_TASK_PROMPT_TEMPLATE)
        set_default_max_turns(options, _JUDGE_DEFAULT_MAX_TURNS)
        append_hooks(options, "PostToolUse", [_make_stop_after_verdict_hook(state)])
        options_snapshot = _snapshot_claude_options(options)

        async def _drive() -> None:
            nonlocal num_turns, cost_usd, session_id
            nonlocal input_tokens, output_tokens, cache_tokens
            async for msg in query(prompt=pair_prompt, options=options):
                events.append(serialize_message(msg))
                if isinstance(msg, ResultMessage):
                    num_turns = msg.num_turns
                    cost_usd = msg.total_cost_usd
                    session_id = msg.session_id
                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    cache_tokens = usage.get("cache_read_input_tokens")

        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001 — surface anything as an outcome
        err = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    raw_decision = state.decision
    if raw_decision is None and err is None:
        err = (
            f"agent stopped without calling submit_verdict "
            f"(call_count={state.call_count})"
        )
    err = _merge_runtime_diagnostics(err, stderr_lines, events)

    final = _flip(raw_decision) if (flip_responses and raw_decision is not None) else raw_decision

    events.append({
        "type": "verdict",
        "label": label,
        "decision_raw": raw_decision,
        "decision_final": final,
        "submit_verdict_call_count": state.call_count,
        "stop_hook_fired": state.stop_hook_fired,
        "rationale": state.rationale,
    })

    return _OrderingOutcome(
        label=label,
        decision_raw=raw_decision,
        decision_final=final,
        wall_time_s=time.time() - start,
        prompt=pair_prompt,
        response_a_ref=response_a_ref,
        response_b_ref=response_b_ref,
        options_snapshot=options_snapshot,
        error=err,
        num_turns=num_turns,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
        events=events,
    )


async def _run_one_ordering(
    *,
    harness_path: Path,
    label: str,
    pair: JudgePair,
    flip_responses: bool,
    model: str,
    timeout: int,
) -> _OrderingOutcome:
    """Run one ordering and retry one transient SDK/API crash once."""
    attempts = [
        await _run_one_ordering_once(
            harness_path=harness_path,
            label=label,
            pair=pair,
            flip_responses=flip_responses,
            model=model,
            timeout=timeout,
        )
    ]
    while len(attempts) <= _TRANSIENT_RUNTIME_MAX_RETRIES and _should_retry_ordering(attempts[-1]):
        attempts.append(
            await _run_one_ordering_once(
                harness_path=harness_path,
                label=label,
                pair=pair,
                flip_responses=flip_responses,
                model=model,
                timeout=timeout,
            )
        )
    return _merge_ordering_attempts(attempts)


async def _run_program_one_ordering(
    *,
    harness_path: Path,
    label: str,
    pair: JudgePair,
    flip_responses: bool,
    model: str,
    timeout: int,
) -> _OrderingOutcome:
    """Run one ordering of one pair through a candidate-owned program harness."""
    start = time.time()
    response_a = pair.response_b if flip_responses else pair.response_a
    response_b = pair.response_a if flip_responses else pair.response_b
    response_a_ref = "response_b_original" if flip_responses else "response_a_original"
    response_b_ref = "response_a_original" if flip_responses else "response_b_original"
    task = ProgramJudgeTask(
        name=f"{pair.pair_id}::{label}",
        pair_id=pair.pair_id,
        question=pair.question,
        response_a=response_a,
        response_b=response_b,
        source=pair.source,
        category=pair.category,
        ordering_label=label,
        response_a_ref=response_a_ref,
        response_b_ref=response_b_ref,
    )

    cwd = Path(tempfile.mkdtemp(prefix=f"program_judge_{pair.pair_id}_{label}_"))
    ctx = HarnessContext(
        task=task,
        model=resolve_bedrock_model(model),
        cwd=cwd,
        timeout=timeout,
        metadata={
            "pair_id": pair.pair_id,
            "category": pair.category,
            "source": pair.source,
            "ordering_label": label,
            "position_swap": flip_responses,
        },
    )

    err: Optional[str] = None
    decision_raw: Optional[JudgeDecision] = None
    events: list[dict[str, Any]] = []
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    final_output: Any = None
    result_metadata: dict[str, Any] = {}
    try:
        result = await run_program_harness(harness_path, ctx, timeout=timeout)
        final_output = result.final_output
        result_metadata = dict(result.metadata)
        decision_raw = _decision_from_program_result(result)
        events = list(result.events)
        num_turns = result.num_turns
        cost_usd = result.cost_usd
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        cache_tokens = result.cache_tokens
        session_id = result.session_id
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
        events = ctx.events
    except Exception as exc:  # noqa: BLE001 - candidate failure is an outcome
        err = f"{type(exc).__name__}: {exc}"
        events = ctx.events
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    if decision_raw is None and err is None:
        err = (
            "program harness did not return a valid decision; "
            "return ctx.finish(..., decision='A>B') or decision='B>A'"
        )

    final = _flip(decision_raw) if (flip_responses and decision_raw is not None) else decision_raw
    events.append({
        "type": "program_verdict",
        "label": label,
        "decision_raw": decision_raw,
        "decision_final": final,
        "final_output": _trace_jsonable(final_output),
        "metadata": _trace_jsonable(result_metadata),
        "error": err,
    })

    return _OrderingOutcome(
        label=label,
        decision_raw=decision_raw,
        decision_final=final,
        wall_time_s=time.time() - start,
        prompt=task.as_prompt(),
        response_a_ref=response_a_ref,
        response_b_ref=response_b_ref,
        options_snapshot={
            "target": "program_harness",
            "task": {
                "pair_id": pair.pair_id,
                "category": pair.category,
                "source": pair.source,
                "ordering_label": label,
                "response_a_ref": response_a_ref,
                "response_b_ref": response_b_ref,
            },
        },
        error=err,
        num_turns=num_turns,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
        events=events,
    )


# ---------------------------------------------------------------------------
# Per-pair orchestration + TaskResult
# ---------------------------------------------------------------------------


async def judge_pair(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
    position_swap: bool,
) -> _PairOutcome:
    """Score one pair. Runs two orderings when `position_swap` is True."""
    start = time.time()
    orderings = [
        await _run_one_ordering(
            harness_path=harness_path,
            label="original",
            pair=pair,
            flip_responses=False,
            model=model,
            timeout=timeout,
        )
    ]
    if position_swap:
        orderings.append(
            await _run_one_ordering(
                harness_path=harness_path,
                label="swapped",
                pair=pair,
                flip_responses=True,
                model=model,
                timeout=timeout,
            )
        )
    return _PairOutcome(
        pair_id=pair.pair_id,
        gold=pair.gold,
        source=pair.source,
        category=pair.category,
        orderings=orderings,
        wall_time_s=time.time() - start,
        question=pair.question,
        response_a_original=pair.response_a,
        response_b_original=pair.response_b,
    )


async def judge_pair_program(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
    position_swap: bool,
) -> _PairOutcome:
    """Score one pair with a candidate-owned program harness."""
    start = time.time()
    orderings = [
        await _run_program_one_ordering(
            harness_path=harness_path,
            label="original",
            pair=pair,
            flip_responses=False,
            model=model,
            timeout=timeout,
        )
    ]
    if position_swap:
        orderings.append(
            await _run_program_one_ordering(
                harness_path=harness_path,
                label="swapped",
                pair=pair,
                flip_responses=True,
                model=model,
                timeout=timeout,
            )
        )
    return _PairOutcome(
        pair_id=pair.pair_id,
        gold=pair.gold,
        source=pair.source,
        category=pair.category,
        orderings=orderings,
        wall_time_s=time.time() - start,
        question=pair.question,
        response_a_original=pair.response_a,
        response_b_original=pair.response_b,
    )


def write_pair_trace(outcome: _PairOutcome, pair_dir: Path, *, trace_type: str) -> None:
    """Write the full per-pair trace.jsonl the proposer reads on the next iteration."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "pair_id": outcome.pair_id,
                "category": outcome.category,
                "source": outcome.source,
                "gold": outcome.gold,
                "input": {
                    "question": outcome.question,
                    "response_a_original": outcome.response_a_original,
                    "response_b_original": outcome.response_b_original,
                    "system_task_prompt": _JUDGE_TASK_PROMPT_TEMPLATE,
                    "pair_prompt_template": _JUDGE_PAIR_PROMPT_TEMPLATE,
                },
                "decisions": list(outcome.decisions),
                "passed": _position_consistent_correct(outcome),
                "orderings": [
                    {
                        "label": o.label,
                        "input": {
                            "response_a_ref": o.response_a_ref,
                            "response_b_ref": o.response_b_ref,
                            "user_prompt": o.prompt,
                            "options_snapshot": o.options_snapshot,
                        },
                        "decision_raw": o.decision_raw,
                        "decision_final": o.decision_final,
                        "wall_time_s": o.wall_time_s,
                        "error": o.error,
                        "num_turns": o.num_turns,
                        "cost_usd": o.cost_usd,
                        "events": o.events,
                    }
                    for o in outcome.orderings
                ],
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
            }
        )
        + "\n"
    )


def task_result_from_outcome(
    outcome: _PairOutcome,
    work_dir: Path,
    position_swap: bool,
) -> TaskResult:
    passed = _position_consistent_correct(outcome)

    if outcome.error is not None:
        verify_exit_code = 1
        verify_output = f"error: {outcome.error}"
    elif not passed:
        verify_exit_code = 1
        decisions_str = ", ".join(outcome.decisions) if outcome.decisions else "(none)"
        verify_output = (
            f"gold={outcome.gold}  decisions=[{decisions_str}]"
            + ("  position_swap=on" if position_swap else "")
        )
    else:
        verify_exit_code = 0
        verify_output = f"gold={outcome.gold}  all orderings correct"

    total_cost = sum(
        (o.cost_usd or 0.0) for o in outcome.orderings if o.cost_usd is not None
    ) or None
    total_input = sum((o.input_tokens or 0) for o in outcome.orderings) or None
    total_output = sum((o.output_tokens or 0) for o in outcome.orderings) or None
    total_cache = sum((o.cache_tokens or 0) for o in outcome.orderings) or None
    total_turns = sum((o.num_turns or 0) for o in outcome.orderings) or None

    return TaskResult(
        task_name=outcome.pair_id,
        passed=passed,
        reward=1.0 if passed else 0.0,
        cost_usd=total_cost,
        num_turns=total_turns,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_tokens=total_cache,
        session_id=outcome.orderings[0].session_id if outcome.orderings else None,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


# ---------------------------------------------------------------------------
# Shared driver used by both adapters
# ---------------------------------------------------------------------------


def resolve_harness_path(config_path: str) -> Path:
    """Accept either a dir containing harness.py or the harness.py path itself."""
    p = Path(config_path)
    if p.is_file() and p.name == "harness.py":
        return p
    if p.is_dir():
        candidate = p / "harness.py"
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"config_path must be a directory containing harness.py or the harness.py "
        f"file itself; got {config_path!r}"
    )


async def run_judge_benchmark(
    *,
    pairs: List[JudgePair],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    position_swap: bool,
    trace_type: str,
    trace_root: Optional[Path] = None,
    logger: Any = None,
) -> List[TaskResult]:
    """Drive one judge benchmark: validate harness, run all pairs, collect TaskResults."""
    harness_path = resolve_harness_path(config_path)

    n_total = len(pairs)
    tmp_root = trace_root or Path(tempfile.mkdtemp(prefix=f"judge_{trace_type}_"))

    if logger is not None:
        logger.info(
            f"Running {n_total} pairs  concurrency={concurrency}  "
            f"position_swap={position_swap}  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    lock = asyncio.Lock()

    async def _run_one(pair: JudgePair) -> TaskResult:
        nonlocal completed, running_passed
        async with sem:
            outcome = await judge_pair(
                harness_path=harness_path,
                pair=pair,
                model=model,
                timeout=timeout,
                position_swap=position_swap,
            )

        pair_dir = tmp_root / pair.pair_id
        write_pair_trace(outcome, pair_dir, trace_type=trace_type)
        result = task_result_from_outcome(outcome, pair_dir, position_swap)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            rate = running_passed / completed if completed else 0.0
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {pair.pair_id[:8]}..  "
                    f"{dur}  pass_rate={rate:.0%}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(p) for p in pairs])
    return list(results)


async def run_program_judge_benchmark(
    *,
    pairs: List[JudgePair],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    position_swap: bool,
    trace_type: str,
    trace_root: Optional[Path] = None,
    logger: Any = None,
) -> List[TaskResult]:
    """Drive a judge benchmark whose candidate is a full Python program harness."""
    harness_path = resolve_harness_path(config_path)

    n_total = len(pairs)
    tmp_root = trace_root or Path(tempfile.mkdtemp(prefix=f"program_judge_{trace_type}_"))

    if logger is not None:
        logger.info(
            f"Running {n_total} program-harness pairs  concurrency={concurrency}  "
            f"position_swap={position_swap}  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    lock = asyncio.Lock()

    async def _run_one(pair: JudgePair) -> TaskResult:
        nonlocal completed, running_passed
        async with sem:
            outcome = await judge_pair_program(
                harness_path=harness_path,
                pair=pair,
                model=model,
                timeout=timeout,
                position_swap=position_swap,
            )

        pair_dir = tmp_root / pair.pair_id
        write_pair_trace(outcome, pair_dir, trace_type=trace_type)
        result = task_result_from_outcome(outcome, pair_dir, position_swap)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            rate = running_passed / completed if completed else 0.0
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {pair.pair_id[:8]}..  "
                    f"{dur}  pass_rate={rate:.0%}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(p) for p in pairs])
    return list(results)


# ---------------------------------------------------------------------------
# Best-of-N sample abstraction and driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BestOfNSample:
    """One sample for judge scoring (any mode).

    The default single-chosen shape covers pairwise tournament + rankings:
    judge must pick `chosen` over each element of `rejected`. Binary
    accuracy = judge wins every matchup. Graded reward = (wins / len(rejected)).

    For RewardBench 2 Ties (ratings mode), a sample may have multiple
    equally-correct responses. In that case `all_chosen` holds the full
    correct list (`chosen` is always `all_chosen[0]` for back-compat with
    pairwise/rankings callers that assume a single chosen). `num_correct`
    = `len(all_chosen)`. For all other benchmarks `all_chosen` is None
    and `num_correct` is 1.
    """

    sample_id: str
    question: str
    chosen: str
    rejected: List[str]
    category: str = "Other"
    source: str = ""
    all_chosen: Optional[List[str]] = None
    num_correct: int = 1


@dataclass
class _BestOfNOutcome:
    sample_id: str
    category: str
    source: str
    n_rejected: int
    pair_outcomes: List[_PairOutcome]  # one per rejected response
    wall_time_s: float

    @property
    def n_correct(self) -> int:
        """Number of pairwise matchups where judge correctly picked chosen."""
        return sum(
            1
            for p in self.pair_outcomes
            if p.error is None
            and p.decisions
            and all(d == "A>B" for d in p.decisions)
        )

    @property
    def binary_passed(self) -> bool:
        """Strict best-of-N: judge picks chosen over every rejected."""
        return self.n_correct == self.n_rejected and self.n_rejected > 0

    @property
    def graded_reward(self) -> float:
        """Partial-credit score: (correct matchups) / (total matchups)."""
        if self.n_rejected == 0:
            return 0.0
        return self.n_correct / self.n_rejected

    @property
    def error(self) -> Optional[str]:
        errs = [
            f"rej[{i}]: {p.error}" for i, p in enumerate(self.pair_outcomes) if p.error
        ]
        return "; ".join(errs) if errs else None


async def _judge_bestofn_sample(
    *,
    harness_path: Path,
    sample: BestOfNSample,
    model: str,
    timeout: int,
) -> _BestOfNOutcome:
    """Score one best-of-N sample. Runs N-1 sequential pair comparisons.

    Pairs run sequentially within a sample so we don't multiply the global
    concurrency budget by N. Position is fixed at A=chosen, B=rejected[i];
    best-of-N benchmarks don't use position swap (each rejected is a
    distinct response, not a re-ordering).
    """
    start = time.time()
    pair_outcomes: List[_PairOutcome] = []

    for i, rej in enumerate(sample.rejected):
        pair = JudgePair(
            pair_id=f"{sample.sample_id}#rej{i}",
            question=sample.question,
            response_a=sample.chosen,
            response_b=rej,
            gold=cast(JudgeDecision, "A>B"),
            source=sample.source,
            category=sample.category,
        )
        ordering = await _run_one_ordering(
            harness_path=harness_path,
            label=f"rej{i}",
            pair=pair,
            flip_responses=False,
            model=model,
            timeout=timeout,
        )
        pair_outcomes.append(
            _PairOutcome(
                pair_id=pair.pair_id,
                gold=pair.gold,
                source=pair.source,
                category=pair.category,
                orderings=[ordering],
                wall_time_s=ordering.wall_time_s,
            )
        )

    return _BestOfNOutcome(
        sample_id=sample.sample_id,
        category=sample.category,
        source=sample.source,
        n_rejected=len(sample.rejected),
        pair_outcomes=pair_outcomes,
        wall_time_s=time.time() - start,
    )


def write_bestofn_trace(
    outcome: _BestOfNOutcome, sample_dir: Path, *, trace_type: str
) -> None:
    """Write the full per-sample trace.jsonl (one JSON object, `category` +
    `passed` fields so the generic category_scores stats pick it up)."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "sample_id": outcome.sample_id,
                "category": outcome.category,
                "source": outcome.source,
                "n_rejected": outcome.n_rejected,
                "n_correct": outcome.n_correct,
                "passed": outcome.binary_passed,
                "graded_reward": outcome.graded_reward,
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
                "pairs": [
                    {
                        "pair_id": p.pair_id,
                        "decisions": list(p.decisions),
                        "error": p.error,
                        "orderings": [
                            {
                                "label": o.label,
                                "decision_raw": o.decision_raw,
                                "decision_final": o.decision_final,
                                "wall_time_s": o.wall_time_s,
                                "error": o.error,
                                "num_turns": o.num_turns,
                                "cost_usd": o.cost_usd,
                                "events": o.events,
                            }
                            for o in p.orderings
                        ],
                    }
                    for p in outcome.pair_outcomes
                ],
            }
        )
        + "\n"
    )


def _task_result_from_bestofn(
    outcome: _BestOfNOutcome, work_dir: Path
) -> TaskResult:
    """Build a sample-level TaskResult from N pair outcomes.

    `passed` is strict best-of-N (binary). `reward` is the graded
    (partial-credit) pair-accuracy, used as the meta-agent search signal.
    """
    passed = outcome.binary_passed

    if outcome.error is not None:
        verify_exit_code = 1
        verify_output = f"error: {outcome.error}"
    elif not passed:
        verify_exit_code = 1
        verify_output = (
            f"best-of-{outcome.n_rejected}: chosen won "
            f"{outcome.n_correct}/{outcome.n_rejected} matchups"
        )
    else:
        verify_exit_code = 0
        verify_output = (
            f"best-of-{outcome.n_rejected}: chosen won every matchup"
        )

    # Flatten stats across all orderings of all pairs in the sample.
    total_cost = 0.0
    total_input = total_output = total_cache = 0
    total_turns = 0
    first_session: Optional[str] = None
    for p in outcome.pair_outcomes:
        for o in p.orderings:
            if o.cost_usd is not None:
                total_cost += o.cost_usd
            if o.input_tokens:
                total_input += o.input_tokens
            if o.output_tokens:
                total_output += o.output_tokens
            if o.cache_tokens:
                total_cache += o.cache_tokens
            if o.num_turns:
                total_turns += o.num_turns
            if first_session is None:
                first_session = o.session_id

    return TaskResult(
        task_name=outcome.sample_id,
        passed=passed,
        reward=outcome.graded_reward,
        cost_usd=total_cost or None,
        num_turns=total_turns or None,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=total_input or None,
        output_tokens=total_output or None,
        cache_tokens=total_cache or None,
        session_id=first_session,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


async def run_bestofn_benchmark(
    *,
    samples: List[BestOfNSample],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    trace_type: str,
    trace_root: Optional[Path] = None,
    logger: Any = None,
) -> List[TaskResult]:
    """Drive one best-of-N judge benchmark: validate harness, run all samples.

    Concurrency is across samples; pairs within a sample run sequentially
    (matching the pattern used by `run_judge_benchmark`'s two orderings),
    so the effective SDK-call concurrency stays at `concurrency` regardless
    of N.
    """
    harness_path = resolve_harness_path(config_path)

    n_total = len(samples)
    tmp_root = trace_root or Path(tempfile.mkdtemp(prefix=f"bestofn_{trace_type}_"))

    if logger is not None:
        avg_n = (
            sum(len(s.rejected) for s in samples) / n_total if n_total else 0.0
        )
        logger.info(
            f"Running {n_total} samples  concurrency={concurrency}  "
            f"avg_rejected={avg_n:.1f}  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    running_graded = 0.0
    lock = asyncio.Lock()

    async def _run_one(sample: BestOfNSample) -> TaskResult:
        nonlocal completed, running_passed, running_graded
        async with sem:
            outcome = await _judge_bestofn_sample(
                harness_path=harness_path,
                sample=sample,
                model=model,
                timeout=timeout,
            )

        sample_dir = tmp_root / sample.sample_id
        write_bestofn_trace(outcome, sample_dir, trace_type=trace_type)
        result = _task_result_from_bestofn(outcome, sample_dir)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            running_graded += outcome.graded_reward
            binary_rate = running_passed / completed if completed else 0.0
            graded_rate = running_graded / completed if completed else 0.0
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {sample.sample_id[:12]}..  "
                    f"{dur}  binary={binary_rate:.0%}  graded={graded_rate:.0%}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(s) for s in samples])
    return list(results)


# ---------------------------------------------------------------------------
# Best-of-N rankings driver (single call per sample, shuffled positions)
# ---------------------------------------------------------------------------


@dataclass
class _RankingsOutcome:
    """One sample's outcome under the best-of-N rankings exit contract.

    Matches the leaderboard's `run_generative_v2.py` per-sample signal:
    the judge produces free text, we extract `[[A-D]]` via substring
    priority (A > B > C > D), sample passes iff the extracted letter
    equals the chosen response's letter (after shuffle unwind).
    """

    sample_id: str
    category: str
    source: str
    n_choices: int
    # The letter that corresponded to `chosen` after random shuffling
    # (e.g. "C" if the chosen response was shown in the 3rd slot).
    chosen_letter: Optional[str]
    decision_letter: Optional[str]  # what the judge picked (or None on format error)
    passed: bool                    # decision_letter == chosen_letter
    wall_time_s: float
    error: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    events: List[dict[str, Any]] = field(default_factory=list)
    # The judge's raw assistant text output. Stored for diagnostics (letting
    # the proposer see *why* a format error happened, or why a wrong letter
    # was picked). Matches what the leaderboard's `judgement` variable holds.
    judgment_text: str = ""
    # Record the shuffle so traces are reproducible/debuggable.
    response_order: List[int] = field(default_factory=list)  # original indices in shuffled order
    question: str = ""
    responses_original: List[str] = field(default_factory=list)
    user_prompt: str = ""
    system_prompt: str = _REWARDBENCH2_SYSTEM_PROMPT
    options_snapshot: dict[str, Any] = field(default_factory=dict)


def _shuffle_responses(
    sample: BestOfNSample, rng: random.Random
) -> Tuple[List[str], int, List[int]]:
    """Shuffle [chosen, *rejected] into a random order.

    Matches `rewardbench/scripts/run_generative_v2.py` exactly: the chosen
    response is placed uniformly at random into one of the N slots via a
    single pairwise swap with slot 0, while the rejected responses keep
    their original relative order. This is equivalent to
    `shuffle_option = randint(0, N)` and swapping A with the chosen slot.

    Concretely, over many calls:
      - `chosen_new_index` is uniform over 0..N-1
      - Position 0 is either chosen (no swap) or a rejected that was
        displaced there by the swap.

    Returns (responses_in_shuffled_order, chosen_new_index, original_indices).
    `original_indices[i]` is the *original* index (0 = chosen, 1..N-1 = rejected)
    of the response placed at position `i` after shuffling.
    """
    n = len(sample.rejected) + 1
    responses = [sample.chosen, *sample.rejected]
    original_indices = list(range(n))

    chosen_new_index = rng.randrange(n)
    if chosen_new_index != 0:
        original_indices[0], original_indices[chosen_new_index] = (
            original_indices[chosen_new_index],
            original_indices[0],
        )

    shuffled = [responses[i] for i in original_indices]
    return shuffled, chosen_new_index, original_indices


def _collect_assistant_text(events: List[dict[str, Any]]) -> str:
    """Concatenate all TextBlock content from AssistantMessage events.

    The judge may stream multiple text blocks in its turn; we concatenate
    them in order and run the leaderboard substring-priority extractor
    on the full text. This matches `run_generative_v2.py`'s behavior of
    feeding the model's complete response into `process_judgement`.

    Event types come from `serialize_message` in
    `meta_agent/task_runner/artifacts.py`, which uses
    `type(msg).__name__` → "AssistantMessage", "TextBlock", etc.
    (PascalCase). ThinkingBlock content is intentionally excluded —
    the leaderboard parses only the assistant's visible output.
    """
    chunks: List[str] = []
    for ev in events:
        if ev.get("type") != "AssistantMessage":
            continue
        content = ev.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "TextBlock":
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


async def _run_one_rankings_call(
    *,
    harness_path: Path,
    sample: BestOfNSample,
    model: str,
    timeout: int,
    rng: random.Random,
) -> _RankingsOutcome:
    """Run one best-of-N rankings call, matching the leaderboard protocol.

    Protocol (exact match to `rewardbench/generative_v2.py`):
      1. Shuffle positions (random permutation of N responses)
      2. Format with `MTBENCH_V2["prompt_template"]` verbatim
      3. System prompt = `prompt_v2` verbatim (prepended via benchmark
         exit contract; the proposer's system_prompt, if any, follows)
      4. One generative call with NO tools (`allowed_tools=[]`)
      5. Collect the assistant's free text
      6. Extract verdict via substring priority (`[[A]]` > `[[B]]` > ...)
      7. Compare to chosen letter (after shuffle unwind) → pass/fail

    No MCP tool, no PostToolUse hook. The agent naturally terminates at
    end of turn after generating one assistant message.
    """

    ensure_bedrock_env()
    start = time.time()
    n = len(sample.rejected) + 1
    letters = _letters_for(n)
    events: List[dict[str, Any]] = []

    shuffled_responses, chosen_new_index, original_indices = _shuffle_responses(
        sample, rng
    )
    chosen_letter = letters[chosen_new_index]

    # RewardBench 2 only has `prompt_v2` defined for N=4. For other N we'd
    # need a different template; the adapter's only caller (RewardBench 2)
    # emits N=4 samples, so this is correct-by-construction. Guard anyway.
    if n != 4:
        raise ValueError(
            f"rankings-mode user template is defined for N=4 only; got N={n}. "
            "For pairwise (N=2) use scoring_mode='pairwise_tournament' instead."
        )

    user_prompt = _REWARDBENCH2_USER_TEMPLATE.format(
        question=sample.question,
        answer_a=shuffled_responses[0],
        answer_b=shuffled_responses[1],
        answer_c=shuffled_responses[2],
        answer_d=shuffled_responses[3],
    )

    cwd = Path(tempfile.mkdtemp(prefix=f"rankings_{sample.sample_id}_"))
    resolved_model = resolve_bedrock_model(model)
    ctx = RunContext(cwd=str(cwd), model=resolved_model, task_instruction=user_prompt)

    err: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    stderr_lines: list[str] = []
    options_snapshot: dict[str, Any] = {}

    try:
        options = build_claude_agent_options(harness_path, ctx)
        options.stderr = stderr_lines.append

        # Benchmark exit contract for rankings mode = just the leaderboard's
        # system prompt. No MCP server, no tools, no hooks. The proposer
        # can still add their own system_prompt (it follows the leaderboard
        # prompt after prepend_system_prompt).
        prepend_system_prompt(options, _REWARDBENCH2_SYSTEM_PROMPT)
        set_default_max_turns(options, _RANKINGS_DEFAULT_MAX_TURNS)
        options_snapshot = _snapshot_claude_options(options)

        async def _drive() -> None:
            nonlocal num_turns, cost_usd, session_id
            nonlocal input_tokens, output_tokens, cache_tokens
            async for msg in query(prompt=user_prompt, options=options):
                events.append(serialize_message(msg))
                if isinstance(msg, ResultMessage):
                    num_turns = msg.num_turns
                    cost_usd = msg.total_cost_usd
                    session_id = msg.session_id
                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    cache_tokens = usage.get("cache_read_input_tokens")

        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    judgment_text = _collect_assistant_text(events)
    decision_letter = _extract_rankings_verdict(judgment_text, letters)

    if decision_letter is None and err is None:
        err = "format_error: no [[A-D]] token found in judge output"
    err = _merge_runtime_diagnostics(err, stderr_lines, events)

    passed = (
        decision_letter is not None
        and decision_letter == chosen_letter
    )

    events.append({
        "type": "verdict",
        "decision_letter": decision_letter,
        "chosen_letter": chosen_letter,
        "judgment_text_chars": len(judgment_text),
        "response_order": original_indices,
    })

    return _RankingsOutcome(
        sample_id=sample.sample_id,
        category=sample.category,
        source=sample.source,
        n_choices=n,
        chosen_letter=chosen_letter,
        decision_letter=decision_letter,
        passed=passed,
        wall_time_s=time.time() - start,
        error=err,
        num_turns=num_turns,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
        events=events,
        judgment_text=judgment_text,
        response_order=original_indices,
        question=sample.question,
        responses_original=[sample.chosen, *sample.rejected],
        user_prompt=user_prompt,
        system_prompt=_REWARDBENCH2_SYSTEM_PROMPT,
        options_snapshot=options_snapshot,
    )


def write_rankings_trace(
    outcome: _RankingsOutcome, sample_dir: Path, *, trace_type: str
) -> None:
    """Write the per-sample trace.jsonl (one line, `category` + `passed`
    fields so the generic category_scores stats pick it up).

    Includes the judge's raw `judgment_text` so the proposer can diagnose
    format errors and wrong-letter decisions directly from the trace.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "sample_id": outcome.sample_id,
                "category": outcome.category,
                "source": outcome.source,
                "n_choices": outcome.n_choices,
                "chosen_letter": outcome.chosen_letter,
                "decision_letter": outcome.decision_letter,
                "passed": outcome.passed,
                "input": {
                    "question": outcome.question,
                    "responses_original": outcome.responses_original,
                    "response_order": outcome.response_order,
                    "user_prompt": outcome.user_prompt,
                    "system_prompt": outcome.system_prompt,
                    "options_snapshot": outcome.options_snapshot,
                },
                "judgment_text": outcome.judgment_text,
                "response_order": outcome.response_order,
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
                "num_turns": outcome.num_turns,
                "cost_usd": outcome.cost_usd,
                "events": outcome.events,
            }
        )
        + "\n"
    )


def _task_result_from_rankings(
    outcome: _RankingsOutcome, work_dir: Path
) -> TaskResult:
    """Build a TaskResult from a rankings outcome.

    Rankings is a single-call protocol. Reward matches the leaderboard's
    `process_shuffled` return values (`run_generative_v2.py::process_shuffled`):

    - correct letter    -> reward = 1.0,  passed = True
    - wrong letter      -> reward = 0.0,  passed = False
    - format error      -> reward = 0.25, passed = False  ("effectively a tie")

    The 0.25 on format error is the leaderboard's parity rule — it keeps
    the headline number comparable to published Avg-6 scores even when a
    judge occasionally fails to emit `[[A-D]]`.
    """
    is_format_error = (
        outcome.decision_letter is None
        and outcome.error is not None
        and str(outcome.error).startswith("format_error")
    )

    if outcome.passed:
        reward = 1.0
        verify_exit_code = 0
        verify_output = (
            f"best-of-{outcome.n_choices}: chose {outcome.chosen_letter} correctly"
        )
    elif is_format_error:
        reward = 0.25
        verify_exit_code = 1
        verify_output = f"format_error (reward=0.25 per leaderboard parity): {outcome.error}"
    elif outcome.error is not None:
        reward = 0.0
        verify_exit_code = 1
        verify_output = f"error: {outcome.error}"
    else:
        reward = 0.0
        verify_exit_code = 1
        verify_output = (
            f"best-of-{outcome.n_choices}: decision={outcome.decision_letter} "
            f"chosen={outcome.chosen_letter}"
        )

    return TaskResult(
        task_name=outcome.sample_id,
        passed=outcome.passed,
        reward=reward,
        cost_usd=outcome.cost_usd,
        num_turns=outcome.num_turns,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cache_tokens=outcome.cache_tokens,
        session_id=outcome.session_id,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


async def run_bestofn_rankings_benchmark(
    *,
    samples: List[BestOfNSample],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    trace_type: str,
    trace_root: Optional[Path] = None,
    logger: Any = None,
    shuffle_seed: int = 42,
) -> List[TaskResult]:
    """Drive a best-of-N rankings benchmark.

    One SDK call per sample. Positions shuffled with a per-sample RNG
    derived from `shuffle_seed` and the sample_id, so the shuffle is
    deterministic across runs of the same sample but different across
    samples. Matches the protocol of
    `rewardbench/generative_v2.py::prompt_v2` (which uses
    `np.random.rand()` per sample to shuffle).
    """
    harness_path = resolve_harness_path(config_path)

    n_total = len(samples)
    tmp_root = trace_root or Path(
        tempfile.mkdtemp(prefix=f"rankings_{trace_type}_")
    )

    if logger is not None:
        ns = sorted({len(s.rejected) + 1 for s in samples})
        logger.info(
            f"Running {n_total} samples (rankings mode)  concurrency={concurrency}  "
            f"n_choices={ns}  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    lock = asyncio.Lock()

    async def _run_one(sample: BestOfNSample) -> TaskResult:
        nonlocal completed, running_passed
        # Deterministic shuffle per-sample so repeat runs give same shuffle.
        sample_rng = random.Random(f"{shuffle_seed}::{sample.sample_id}")
        async with sem:
            outcome = await _run_one_rankings_call(
                harness_path=harness_path,
                sample=sample,
                model=model,
                timeout=timeout,
                rng=sample_rng,
            )

        sample_dir = tmp_root / sample.sample_id
        write_rankings_trace(outcome, sample_dir, trace_type=trace_type)
        result = _task_result_from_rankings(outcome, sample_dir)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            rate = running_passed / completed if completed else 0.0
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {sample.sample_id[:12]}..  "
                    f"{dur}  rate={rate:.0%}  chose={outcome.decision_letter}/"
                    f"gold={outcome.chosen_letter}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(s) for s in samples])
    return list(results)


# ---------------------------------------------------------------------------
# Ratings-mode driver (individual rating of each response, 1-10 scale)
# ---------------------------------------------------------------------------
#
# Matches `rewardbench/generative_v2.py::get_single_rating` + `run_judge_ratings_multi`
# and `run_generative_v2.py::get_judgement`'s ratings branch. Used for:
#   - RewardBench 2 Ties subset (mandatory; the leaderboard's Ties scorer is
#     ratings-native because ties have multiple equally-correct responses)
#   - Optional `--score-w-ratings` mode for non-Ties subsets (appendix only).
#
# Per-sample protocol:
#   1. For each response in chosen + rejected:
#      - Prompt with `ratings_prompt` (or `ratings_prompt_ties`)
#      - One generative call, no tools, free text
#      - Extract trailing 1-10 integer via regex
#   2. Emit per-completion ratings list; `num_correct` tells downstream which
#      indices are "correct" (leaderboard's `i < num_correct` rule).
#   3. Per-sample `accurate` = min(correct scores) > max(incorrect scores).
#
# Subset-level Ties aggregation (process_single_model) runs post-hoc from
# traces.


# Verbatim `ratings_prompt` from `rewardbench/generative_v2.py`. Do NOT edit.
_REWARDBENCH2_RATINGS_PROMPT = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


# Verbatim `ratings_prompt_ties` from `rewardbench/generative_v2.py`. Do NOT edit.
# Note "but need not consider depth or level of detail" — this is what makes the
# Ties prompt tolerant of equally-good-but-different responses.
_REWARDBENCH2_RATINGS_PROMPT_TIES = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


# Matches `get_single_rating`'s trailing-integer regex. Must be on the *last*
# number in the judgment so a judge's "out of 10 possible" intro doesn't get
# scooped up as the rating. Returns -1 on no-match, identical to leaderboard.
_RATING_RE = re.compile(r"\b([1-9]|10)\b\s*$")


def _extract_rating(text: str) -> int:
    """Extract 1-10 rating from free-form text.

    Matches `rewardbench/generative_v2.py::get_single_rating`: regex for
    `\\b([1-9]|10)\\b\\s*$` on the stripped judgment. Returns -1 on
    no-match, identical to leaderboard.
    """
    if not text:
        return -1
    m = _RATING_RE.search(text.strip())
    if not m:
        return -1
    val = int(m.group(1))
    return val if 1 <= val <= 10 else -1


@dataclass
class _RatingsOutcome:
    """One sample's outcome under the ratings exit contract.

    `ratings` is the list of per-completion 1-10 ratings (or -1 on failure),
    parallel to `completions = all_chosen + rejected`. The first
    `num_correct` indices are the "correct" responses per leaderboard
    convention. `accurate` = min(correct scores) > max(incorrect scores).
    """

    sample_id: str
    category: str
    source: str
    num_correct: int
    ratings: List[int]
    judgments: List[str]
    per_completion_errors: List[Optional[str]]
    wall_time_s: float
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    accurate: bool = False
    error: Optional[str] = None
    question: str = ""
    completions: List[str] = field(default_factory=list)
    user_prompts: List[str] = field(default_factory=list)
    options_snapshots: List[dict[str, Any]] = field(default_factory=list)


async def _run_one_ratings_call(
    *,
    harness_path: Path,
    question: str,
    completion: str,
    model: str,
    timeout: int,
    is_ties: bool,
) -> Tuple[int, str, Optional[str], dict[str, Any]]:
    """Run one rating call (one completion).

    Returns (rating, judgment_text, error, metadata_dict). Metadata has
    per-call num_turns, cost_usd, input_tokens, etc. Rating is -1 on
    failure per leaderboard semantics.
    """
    ensure_bedrock_env()
    start = time.time()

    template = _REWARDBENCH2_RATINGS_PROMPT_TIES if is_ties else _REWARDBENCH2_RATINGS_PROMPT
    user_prompt = template.format(prompt=question, completion=completion)

    cwd = Path(tempfile.mkdtemp(prefix=f"ratings_{'ties' if is_ties else 'plain'}_"))
    resolved_model = resolve_bedrock_model(model)
    ctx = RunContext(cwd=str(cwd), model=resolved_model, task_instruction=user_prompt)

    events: List[dict[str, Any]] = []
    err: Optional[str] = None
    stderr_lines: list[str] = []
    meta: dict[str, Any] = {
        "num_turns": None,
        "cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_tokens": None,
        "session_id": None,
        "user_prompt": user_prompt,
        "options_snapshot": {},
    }

    try:
        options = build_claude_agent_options(harness_path, ctx)
        options.stderr = stderr_lines.append
        # Ratings mode has NO benchmark system prompt (matches leaderboard's
        # `system_prompt = ""` in get_single_rating). Everything is in the
        # user prompt. The proposer's system_prompt is preserved as-is.
        set_default_max_turns(options, _RANKINGS_DEFAULT_MAX_TURNS)
        meta["options_snapshot"] = _snapshot_claude_options(options)

        async def _drive() -> None:
            async for msg in query(prompt=user_prompt, options=options):
                events.append(serialize_message(msg))
                if isinstance(msg, ResultMessage):
                    meta["num_turns"] = msg.num_turns
                    meta["cost_usd"] = msg.total_cost_usd
                    meta["session_id"] = msg.session_id
                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                    meta["input_tokens"] = usage.get("input_tokens")
                    meta["output_tokens"] = usage.get("output_tokens")
                    meta["cache_tokens"] = usage.get("cache_read_input_tokens")

        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    judgment_text = _collect_assistant_text(events)
    rating = _extract_rating(judgment_text)
    if rating == -1 and err is None:
        err = "format_error: no trailing 1-10 rating found in judge output"
    err = _merge_runtime_diagnostics(err, stderr_lines, events)

    meta["wall_time_s"] = time.time() - start
    return rating, judgment_text, err, meta


def _compute_ratings_accurate(ratings: List[int], num_correct: int) -> bool:
    """Match `_compute_prompt_stats` accurate semantics: min(correct) > max(incorrect).

    Ignores -1 failures (they don't count as either correct or incorrect).
    """
    correct = [r for i, r in enumerate(ratings) if i < num_correct and r != -1]
    incorrect = [r for i, r in enumerate(ratings) if i >= num_correct and r != -1]
    if not correct or not incorrect:
        return False
    return min(correct) > max(incorrect)


async def _run_one_ratings_sample(
    *,
    harness_path: Path,
    sample: BestOfNSample,
    model: str,
    timeout: int,
    is_ties: bool,
    per_completion_concurrency: int,
) -> _RatingsOutcome:
    """Run ratings for every completion in a sample; aggregate.

    Completions = (all_chosen or [chosen]) + rejected; first `num_correct`
    are correct. Matches `run_judge_ratings_multi`'s loop over `all_answers`,
    one rating per completion.
    """
    start = time.time()

    chosen_list = sample.all_chosen if sample.all_chosen is not None else [sample.chosen]
    completions = [*chosen_list, *sample.rejected]
    num_correct = sample.num_correct if sample.all_chosen is not None else 1
    if num_correct != len(chosen_list):
        num_correct = len(chosen_list)

    sem = asyncio.Semaphore(max(1, per_completion_concurrency))

    async def _one(completion: str) -> Tuple[int, str, Optional[str], dict[str, Any]]:
        async with sem:
            return await _run_one_ratings_call(
                harness_path=harness_path,
                question=sample.question,
                completion=completion,
                model=model,
                timeout=timeout,
                is_ties=is_ties,
            )

    results = await asyncio.gather(*[_one(c) for c in completions])
    ratings = [r[0] for r in results]
    judgments = [r[1] for r in results]
    errors = [r[2] for r in results]
    metas = [r[3] for r in results]

    def _sum_opt(key: str) -> Optional[float]:
        vals = [m.get(key) for m in metas if m.get(key) is not None]
        return sum(vals) if vals else None

    valid = [r for r in ratings if r != -1]
    sample_error: Optional[str] = None
    if not valid:
        sample_error = "All ratings invalid."
        accurate = False
    else:
        accurate = _compute_ratings_accurate(ratings, num_correct)

    return _RatingsOutcome(
        sample_id=sample.sample_id,
        category=sample.category,
        source=sample.source,
        num_correct=num_correct,
        ratings=ratings,
        judgments=judgments,
        per_completion_errors=errors,
        wall_time_s=time.time() - start,
        num_turns=cast(Optional[int], _sum_opt("num_turns")),
        cost_usd=_sum_opt("cost_usd"),
        input_tokens=cast(Optional[int], _sum_opt("input_tokens")),
        output_tokens=cast(Optional[int], _sum_opt("output_tokens")),
        cache_tokens=cast(Optional[int], _sum_opt("cache_tokens")),
        accurate=accurate,
        error=sample_error,
        question=sample.question,
        completions=completions,
        user_prompts=[
            m.get("user_prompt") for m in metas if isinstance(m.get("user_prompt"), str)
        ],
        options_snapshots=[
            cast(dict[str, Any], m.get("options_snapshot"))
            for m in metas
            if isinstance(m.get("options_snapshot"), dict)
        ],
    )


def write_ratings_trace(
    outcome: _RatingsOutcome, sample_dir: Path, *, trace_type: str
) -> None:
    """Write the per-sample ratings trace. Includes full `ratings` + `num_correct`
    so the subset-level Ties scorer reconstructs `process_single_model` exactly.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "sample_id": outcome.sample_id,
                "category": outcome.category,
                "source": outcome.source,
                "num_correct": outcome.num_correct,
                "ratings": outcome.ratings,
                "accurate": outcome.accurate,
                "passed": outcome.accurate,
                "input": {
                    "question": outcome.question,
                    "completions": outcome.completions,
                    "num_correct": outcome.num_correct,
                    "user_prompts": outcome.user_prompts,
                    "options_snapshots": outcome.options_snapshots,
                },
                "judgments": outcome.judgments,
                "per_completion_errors": outcome.per_completion_errors,
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
                "num_turns": outcome.num_turns,
                "cost_usd": outcome.cost_usd,
            }
        )
        + "\n"
    )


def _task_result_from_ratings(
    outcome: _RatingsOutcome, work_dir: Path
) -> TaskResult:
    """Build a TaskResult from a ratings outcome.

    Per-sample `passed` = `accurate` (all correct scored above all incorrect).
    `reward` = 1.0 if accurate else 0.0 if any valid ratings else 0.25
    (matching `get_vllm_judgement`'s "all invalid -> 0.25" parity).

    The full leaderboard Ties score (60% accuracy + 40% margin components)
    is computed post-hoc at the subset level from traces. This per-sample
    reward is an approximation suitable for optimization signal only.
    """
    valid = [r for r in outcome.ratings if r != -1]

    if outcome.accurate:
        reward = 1.0
        verify_exit_code = 0
        verify_output = f"ratings: min_correct > max_incorrect  ratings={outcome.ratings}"
    elif not valid:
        reward = 0.25
        verify_exit_code = 1
        verify_output = "ratings: all invalid (format_error); reward=0.25 per leaderboard parity"
    else:
        reward = 0.0
        verify_exit_code = 1
        verify_output = (
            f"ratings: not accurate  num_correct={outcome.num_correct}  "
            f"ratings={outcome.ratings}"
        )

    return TaskResult(
        task_name=outcome.sample_id,
        passed=outcome.accurate,
        reward=reward,
        cost_usd=outcome.cost_usd,
        num_turns=outcome.num_turns,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cache_tokens=outcome.cache_tokens,
        session_id=None,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


async def run_ratings_benchmark(
    *,
    samples: List[BestOfNSample],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    trace_type: str,
    is_ties: bool,
    trace_root: Optional[Path] = None,
    logger: Any = None,
    per_completion_concurrency: Optional[int] = None,
) -> List[TaskResult]:
    """Drive a ratings-mode benchmark.

    Each sample fans out to N calls (one per completion). `concurrency`
    caps concurrent samples; `per_completion_concurrency` caps concurrent
    calls within a sample (default = 4, reasonable for Ties which average
    ~20 completions per sample).

    Sample-level trace JSONL includes raw ratings + num_correct so the
    subset-level scorer can compute
    the full leaderboard Ties score post-hoc.
    """
    harness_path = resolve_harness_path(config_path)

    n_total = len(samples)
    tmp_root = trace_root or Path(
        tempfile.mkdtemp(prefix=f"ratings_{trace_type}_")
    )

    if logger is not None:
        total_calls = sum(
            (len(s.all_chosen) if s.all_chosen is not None else 1) + len(s.rejected)
            for s in samples
        )
        logger.info(
            f"Running {n_total} samples (ratings mode, is_ties={is_ties})  "
            f"samples_concurrency={concurrency}  per_completion_concurrency="
            f"{per_completion_concurrency or 4}  total_calls={total_calls}  "
            f"timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_accurate = 0
    lock = asyncio.Lock()

    async def _run_one(sample: BestOfNSample) -> TaskResult:
        nonlocal completed, running_accurate
        async with sem:
            outcome = await _run_one_ratings_sample(
                harness_path=harness_path,
                sample=sample,
                model=model,
                timeout=timeout,
                is_ties=is_ties,
                per_completion_concurrency=per_completion_concurrency or 4,
            )

        sample_dir = tmp_root / sample.sample_id
        write_ratings_trace(outcome, sample_dir, trace_type=trace_type)
        result = _task_result_from_ratings(outcome, sample_dir)

        async with lock:
            completed += 1
            if outcome.accurate:
                running_accurate += 1
            rate = running_accurate / completed if completed else 0.0
            mark = "PASS" if outcome.accurate else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {sample.sample_id[:18]}..  "
                    f"{dur}  rate={rate:.0%}  nc={outcome.num_correct}  "
                    f"ratings={outcome.ratings}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(s) for s in samples])
    return list(results)


# ---------------------------------------------------------------------------
# Arena-Hard judge protocol (PPE-HP leaderboard-exact)
# ---------------------------------------------------------------------------
#
# Matches `lmarena/PPE/llm_judge/judges/arena_hard.py` + `prompts/arena_hard_default`
# verbatim. The PPE Arena-Hard judge is the canonical LLM-judge for the
# Human-Preference-v1 benchmark. Protocol:
#
#   1. System prompt = `arena_hard_default` file contents, verbatim
#      (NOTE: the file uses literal `\n` sequences, not actual newlines.
#       That's how PPE sends it to the model — we match byte-for-byte.)
#   2. User prompt = `<|User Prompt|>\n{q}\n\n<|The Start of Assistant A's Answer|>\n{a}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{b}\n<|The End of Assistant B's Answer|>`
#   3. One generative call (no tools, no hooks), free text
#   4. Extract last `[[<verdict>]]` match via regex `\[\[([AB<>=]+)\]\]`
#   5. Map via {A>>B:1, A>B:1, B>A:0, B>>A:0}; anything else (ties, parse
#      failure) returns None — treated as "dropped" to match PPE's scorer
#      `dropna()` behavior when computing mean accuracy.
#   6. No position swap. PPE runs each row exactly once with the
#      dataset-provided ordering (response_1 → A, response_2 → B).
#
# This enables direct comparison to PPE's published Arena-Hard leaderboard
# numbers. Train-mode (Skywork pools) retains the pairwise_swap protocol
# via `submit_verdict` — position-swap is useful as a training signal,
# even though PPE's canonical eval doesn't use it.


# Verbatim contents of `llm_judge/prompts/arena_hard_default`. Do NOT edit
# this string — matching PPE byte-for-byte is the point. The file
# contains 15 literal `\n` escape sequences (not actual newlines); that's
# how PPE sends it to the judge API.
_ARENA_HARD_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
    "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better."
    "\\n\\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers."
    "\\n\\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information."
    "\\n\\nThen consider if the assistant's answers are helpful, relevant, and concise. "
    "Helpful means the answer correctly responds to the prompt or follows the instructions. "
    "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. "
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. "
    "Concise means the response is clear and not verbose or excessive."
    "\\n\\nThen consider the creativity and novelty of the assistant's answers when needed. "
    "Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt."
    "\\n\\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:"
    "\\n\\n1. Assistant A is significantly better: [[A>>B]]"
    "\\n2. Assistant A is slightly better: [[A>B]]"
    "\\n3. Assistant B is slightly better: [[B>A]]"
    "\\n4. Assistant B is significantly better: [[B>>A]]."
)


# Verbatim `ArenaHardJudge.prompt_format` from `llm_judge/judges/arena_hard.py`.
# Uses actual newlines because PPE's Python string uses `\n` escape sequences
# inside a regular string literal (which evaluate to real newlines at runtime).
# Do NOT edit.
_ARENA_HARD_USER_TEMPLATE = (
    "<|User Prompt|>\n{question_1}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"
)


# Regex + score_map matches `ArenaHardJudge` byte-for-byte.
_ARENA_HARD_PATTERN = re.compile(r"\[\[([AB<>=]+)\]\]")

# Arena-Hard is a 2-way forced choice (no tie option). Verdicts that don't
# appear here (most notably `[[A=B]]` if the judge rebels) map to None,
# same as PPE's KeyError → decision=None → dropped by scorer.
_ARENA_HARD_SCORE_MAP: dict[str, int] = {
    "A>>B": 1,  # Assistant A significantly better
    "A>B": 1,   # Assistant A slightly better
    "B>A": 0,   # Assistant B slightly better
    "B>>A": 0,  # Assistant B significantly better
}


# Arena-Hard's default prompt instructs the judge to "Begin your evaluation
# by generating your own answer to the prompt" before judging. That can
# produce a long first turn. 2 turns is enough (one assistant response +
# safety margin); the judge naturally terminates at end of turn once the
# verdict is emitted.
_ARENA_HARD_DEFAULT_MAX_TURNS = 2


def _extract_arena_hard_verdict(text: str) -> Optional[int]:
    """Extract the 0/1 verdict from an Arena-Hard judgment.

    Matches `ArenaHardJudge._parse_judgment` exactly:
      - Find all `[[...]]` matches via the Arena-Hard regex
      - Take the LAST match (`output[-1]`)
      - Look up in score_map
      - Return None on no match or unknown verdict (including ties)

    Returns:
        1 → Assistant A wins (A>>B or A>B)
        0 → Assistant B wins (B>A or B>>A)
        None → parse failure, tie verdict, or anything off-script
    """
    if not text:
        return None
    matches = _ARENA_HARD_PATTERN.findall(text)
    if not matches:
        return None
    choice = matches[-1].strip()
    return _ARENA_HARD_SCORE_MAP.get(choice)


@dataclass
class _ArenaHardOutcome:
    """One pair's outcome under the Arena-Hard judge protocol.

    Matches the per-row output structure of `llm_judge/evaluate.py`:
      - decision ∈ {0, 1, None}
      - judgment (raw free-form text)
    Plus metadata for cost/trace bookkeeping.

    `passed` = (decision == gold_01). A None decision is NOT passed but
    doesn't count as wrong either — the subset accuracy sidecar will
    drop None rows to match PPE's `dropna()`.
    """

    pair_id: str
    category: str
    source: str
    gold_01: int                      # 1 = A preferred, 0 = B preferred
    decision: Optional[int]           # judge's extracted decision (0/1/None)
    passed: bool
    judgment_text: str
    wall_time_s: float
    error: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    events: List[dict[str, Any]] = field(default_factory=list)
    question: str = ""
    response_a: str = ""
    response_b: str = ""
    user_prompt: str = ""
    options_snapshot: dict[str, Any] = field(default_factory=dict)


def _gold_to_01(gold: JudgeDecision) -> int:
    """Map our `JudgePair.gold` ("A>B"/"B>A") to Arena-Hard's 0/1 convention."""
    return 1 if gold == "A>B" else 0


async def _run_one_arena_hard_call(
    *,
    harness_path: Path,
    pair: JudgePair,
    model: str,
    timeout: int,
) -> _ArenaHardOutcome:
    """Run one Arena-Hard judgment under the PPE extraction protocol.

    Protocol (no position swap, single call):
      1. System prompt = whatever the harness sets. The proposer owns
         this lever; it must include verdict-format instructions
         (`[[A>>B]]`/`[[A>B]]`/`[[B>A]]`/`[[B>>A]]`) for extraction to
         succeed.
      2. User prompt = `_ARENA_HARD_USER_TEMPLATE` formatted with
         question + response_a + response_b (ordered as given — PPE
         never swaps)
      3. One generative call, no tools (`allowed_tools=[]`), no hooks
      4. Collect assistant text, extract verdict via
         `_extract_arena_hard_verdict`
      5. Compare to gold → passed/failed
    """
    ensure_bedrock_env()
    start = time.time()
    events: List[dict[str, Any]] = []

    user_prompt = _ARENA_HARD_USER_TEMPLATE.format(
        question_1=pair.question,
        answer_1=pair.response_a,
        answer_2=pair.response_b,
    )

    cwd = Path(tempfile.mkdtemp(prefix=f"arena_hard_{pair.pair_id}_"))
    resolved_model = resolve_bedrock_model(model)
    ctx = RunContext(cwd=str(cwd), model=resolved_model, task_instruction=user_prompt)

    err: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None
    stderr_lines: list[str] = []
    options_snapshot: dict[str, Any] = {}

    try:
        options = build_claude_agent_options(harness_path, ctx)
        options.stderr = stderr_lines.append
        # No adapter-injected system prompt. The harness owns the full
        # system_prompt lever; it must include verdict-format instructions
        # for the regex extractor to succeed. See
        # Preserve runtime-owned fields from the harness context.
        set_default_max_turns(options, _ARENA_HARD_DEFAULT_MAX_TURNS)
        options_snapshot = _snapshot_claude_options(options)

        async def _drive() -> None:
            nonlocal num_turns, cost_usd, session_id
            nonlocal input_tokens, output_tokens, cache_tokens
            async for msg in query(prompt=user_prompt, options=options):
                events.append(serialize_message(msg))
                if isinstance(msg, ResultMessage):
                    num_turns = msg.num_turns
                    cost_usd = msg.total_cost_usd
                    session_id = msg.session_id
                    usage = msg.usage if isinstance(msg.usage, dict) else {}
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    cache_tokens = usage.get("cache_read_input_tokens")

        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

    judgment_text = _collect_assistant_text(events)
    decision = _extract_arena_hard_verdict(judgment_text)

    if decision is None and err is None:
        err = (
            "format_error: no parseable "
            "[[A>>B]]/[[A>B]]/[[B>A]]/[[B>>A]] verdict in judge output"
        )
    err = _merge_runtime_diagnostics(err, stderr_lines, events)

    gold_01 = _gold_to_01(pair.gold)
    passed = decision is not None and decision == gold_01

    events.append({
        "type": "verdict",
        "decision_01": decision,
        "gold_01": gold_01,
        "judgment_text_chars": len(judgment_text),
    })

    return _ArenaHardOutcome(
        pair_id=pair.pair_id,
        category=pair.category,
        source=pair.source,
        gold_01=gold_01,
        decision=decision,
        passed=passed,
        judgment_text=judgment_text,
        wall_time_s=time.time() - start,
        error=err,
        num_turns=num_turns,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        session_id=session_id,
        events=events,
        question=pair.question,
        response_a=pair.response_a,
        response_b=pair.response_b,
        user_prompt=user_prompt,
        options_snapshot=options_snapshot,
    )


def _write_arena_hard_trace(
    outcome: _ArenaHardOutcome, pair_dir: Path, *, trace_type: str
) -> None:
    """Write the per-pair trace. Includes judgment_text + decision so the
    proposer can diagnose parse failures / wrong decisions directly.
    Matches the structure of our other judge-trace writers.
    """
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": trace_type,
                "pair_id": outcome.pair_id,
                "category": outcome.category,
                "source": outcome.source,
                "gold_01": outcome.gold_01,
                "decision_01": outcome.decision,
                "passed": outcome.passed,
                "input": {
                    "question": outcome.question,
                    "response_a": outcome.response_a,
                    "response_b": outcome.response_b,
                    "user_prompt": outcome.user_prompt,
                    "arena_hard_user_template": _ARENA_HARD_USER_TEMPLATE,
                    "arena_hard_reference_system_prompt": _ARENA_HARD_SYSTEM_PROMPT,
                    "options_snapshot": outcome.options_snapshot,
                },
                "judgment_text": outcome.judgment_text,
                "error": outcome.error,
                "wall_time_s": outcome.wall_time_s,
                "num_turns": outcome.num_turns,
                "cost_usd": outcome.cost_usd,
                "events": outcome.events,
            }
        )
        + "\n"
    )


def _task_result_from_arena_hard(
    outcome: _ArenaHardOutcome, work_dir: Path
) -> TaskResult:
    """Build a TaskResult from an Arena-Hard outcome.

    Reward semantics:
      - decision == gold (passed) → reward = 1.0
      - decision != gold (wrong)  → reward = 0.0
      - decision is None (parse fail / tie verdict) → reward = 0.0

    Note on parse failures: PPE's scorer drops rows where decision is None
    via `dropna()`, effectively excluding them from the denominator. Here
    we keep them in the denominator as reward=0.0 (strict). A parallel
    "accuracy_no_na" metric can be computed at the subset level to match
    PPE's `dropna` behavior exactly — `passed=False` on these rows plus
    `decision_01=None` in the trace is enough data for that post-process.
    """
    is_parse_fail = (
        outcome.decision is None
        and outcome.error is not None
        and str(outcome.error).startswith("format_error")
    )

    if outcome.passed:
        reward = 1.0
        verify_exit_code = 0
        verify_output = (
            f"arena_hard: decision={outcome.decision} "
            f"gold={outcome.gold_01} (correct)"
        )
    elif is_parse_fail:
        reward = 0.0
        verify_exit_code = 1
        verify_output = "arena_hard: parse_failure (PPE drops this row from accuracy_no_na)"
    elif outcome.error is not None:
        reward = 0.0
        verify_exit_code = 1
        verify_output = f"arena_hard: error={outcome.error}"
    else:
        reward = 0.0
        verify_exit_code = 1
        verify_output = (
            f"arena_hard: decision={outcome.decision} "
            f"gold={outcome.gold_01} (wrong)"
        )

    return TaskResult(
        task_name=outcome.pair_id,
        passed=outcome.passed,
        reward=reward,
        cost_usd=outcome.cost_usd,
        num_turns=outcome.num_turns,
        duration_ms=int(outcome.wall_time_s * 1000),
        wall_time_s=outcome.wall_time_s,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cache_tokens=outcome.cache_tokens,
        session_id=outcome.session_id,
        work_dir=str(work_dir),
        verify_exit_code=verify_exit_code,
        verify_output=verify_output,
    )


async def run_arena_hard_benchmark(
    *,
    pairs: List[JudgePair],
    config_path: str,
    model: str,
    concurrency: int,
    timeout: int,
    trace_type: str,
    trace_root: Optional[Path] = None,
    logger: Any = None,
) -> List[TaskResult]:
    """Drive a PPE Arena-Hard evaluation over `pairs`.

    One generative call per pair (no position swap). Each call uses the
    verbatim Arena-Hard prompt + format. Results are directly comparable
    to PPE's published leaderboard numbers when evaluated on the same
    dataset (e.g. `lmarena-ai/PPE-Human-Preference-V1`).
    """
    harness_path = resolve_harness_path(config_path)

    n_total = len(pairs)
    tmp_root = trace_root or Path(
        tempfile.mkdtemp(prefix=f"arena_hard_{trace_type}_")
    )

    if logger is not None:
        logger.info(
            f"Running {n_total} pairs (arena_hard mode)  concurrency={concurrency}  "
            f"position_swap=False (PPE-exact)  timeout={timeout}s"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    running_parseable = 0
    running_correct_given_parseable = 0
    lock = asyncio.Lock()

    async def _run_one(pair: JudgePair) -> TaskResult:
        nonlocal completed, running_passed, running_parseable, running_correct_given_parseable
        async with sem:
            outcome = await _run_one_arena_hard_call(
                harness_path=harness_path,
                pair=pair,
                model=model,
                timeout=timeout,
            )

        pair_dir = tmp_root / pair.pair_id
        _write_arena_hard_trace(outcome, pair_dir, trace_type=trace_type)
        result = _task_result_from_arena_hard(outcome, pair_dir)

        async with lock:
            completed += 1
            if result.passed:
                running_passed += 1
            if outcome.decision is not None:
                running_parseable += 1
                if outcome.passed:
                    running_correct_given_parseable += 1
            strict_rate = running_passed / completed if completed else 0.0
            ppe_rate = (
                running_correct_given_parseable / running_parseable
                if running_parseable else 0.0
            )
            mark = "PASS" if result.passed else "FAIL"
            if logger is not None:
                dur = f"{outcome.wall_time_s:.1f}s"
                tail = f" err={outcome.error}" if outcome.error else ""
                logger.info(
                    f"[{completed:>3}/{n_total}] {mark}  {pair.pair_id[:10]}..  "
                    f"{dur}  strict={strict_rate:.0%}  ppe={ppe_rate:.0%}  "
                    f"dec={outcome.decision}/gold={outcome.gold_01}{tail}"
                )
        return result

    results = await asyncio.gather(*[_run_one(p) for p in pairs])
    return list(results)
