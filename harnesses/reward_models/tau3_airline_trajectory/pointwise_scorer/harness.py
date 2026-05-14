"""Pointwise latent reward baseline for tau3 trajectory success.

The benchmark adapter calls this harness once per trajectory. This harness
returns only a scalar reward for the single trajectory it receives; the adapter
owns the later A/B score comparison needed for pairwise tau3 accuracy.
"""

from __future__ import annotations

import json
import re
from typing import Any


MIN_SCORE = 1
MAX_SCORE = 100
MAX_CRITIQUE_CHARS = 240


RUBRIC_ISSUES = {
    "task_completion",
    "planning_state_tracking",
    "tool_grounding",
    "tool_hallucination",
    "safety_refusal",
    "policy_or_constraint_violation",
    "irrelevance_unavailable",
    "robustness_recovery",
    "stale_constraint",
    "redundant_or_inefficient",
    "final_answer_quality",
    "unclear",
}


SEVERITIES = {"none", "minor", "major", "critical"}


SCORER_SYSTEM = """You are a strict tau-airline trajectory reward model.

You will receive:
- one customer-service agent trajectory
- the customer's messages, assistant messages, tool calls, and tool results

Assign a latent task-success reward score from 1 to 100.

This score is not a percentage or globally calibrated grade. It will only be
compared against scores for other trajectories on the same customer task.
Higher means the trajectory is more likely to have correctly resolved the
customer's issue according to airline policy.

Score exactly one trajectory. Do not compare against any unseen alternative.
Do not mention Response A, Response B, winner, pass/fail, or pairwise labels.

Use this success rubric:
- 1-20: severe failure. Wrong or missing resolution, unsafe or unauthorized
  database reads/mutations, policy violation, fabricated tool state, or final
  answer contradicts tool results.
- 21-40: mostly failed. Some relevant steps, but the core customer request is
  unresolved or a key policy/tool-grounding error remains.
- 41-60: partial success. Understands the goal and uses relevant tools, but
  leaves important constraints, mutations, recovery, or final communication
  incomplete or uncertain.
- 61-80: mostly successful. Correct goal and mostly correct tool/policy use,
  with minor inefficiency, incomplete explanation, or low-risk omissions.
- 81-100: successful. Correctly identifies the user goal, verifies required
  identity and policy constraints, grounds decisions in tool outputs, performs
  the right mutations if needed, recovers from errors, and accurately states the
  final outcome.

Tool success alone is not enough: the action must have been allowed and must
match the user's request. Surface quality, confidence, politeness, and length
should not dominate task success.

Call the `record_score` tool exactly once. Do not write prose outside the tool.
The score is the only reward signal. The other fields are diagnostic notes for
future proposers reading search traces; they must not compare this trajectory
against any unseen alternative.

Tool fields:
- score: integer 1-100.
- critique: one concise sentence, at most 240 characters.
- rubric_issue: one of task_completion, planning_state_tracking,
  tool_grounding, tool_hallucination, safety_refusal,
  policy_or_constraint_violation, irrelevance_unavailable,
  robustness_recovery, stale_constraint, redundant_or_inefficient,
  final_answer_quality, unclear.
- severity: one of none, minor, major, critical; severity names the strongest
  issue noted in the critique.
"""


SCORE_TOOL_NAME = "record_score"


SCORE_TOOL = {
    "name": SCORE_TOOL_NAME,
    "description": (
        "Record the scalar latent reward score and proposer-facing diagnostic "
        "notes for exactly one trajectory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": MIN_SCORE,
                "maximum": MAX_SCORE,
                "description": "Latent task-success reward score for the single trajectory.",
            },
            "critique": {
                "type": "string",
                "maxLength": MAX_CRITIQUE_CHARS,
                "description": (
                    "One concise proposer-facing sentence explaining the main "
                    "reason for the score."
                ),
            },
            "rubric_issue": {
                "type": "string",
                "enum": sorted(RUBRIC_ISSUES),
                "description": "Diagnostic rubric issue behind the score.",
            },
            "severity": {
                "type": "string",
                "enum": sorted(SEVERITIES),
                "description": "Severity of the main issue named in the critique.",
            },
        },
        "required": ["score", "critique", "rubric_issue", "severity"],
        "additionalProperties": False,
    },
}


FORCED_SCORE_TOOL = {
    "tools": [SCORE_TOOL],
    "tool_choice": {"type": "tool", "name": SCORE_TOOL_NAME},
}


def render_task_context(task: Any) -> str:
    question = str(getattr(task, "question", "") or "")
    category = str(getattr(task, "category", "") or "")
    return (
        f"{question.strip()}\n\n"
        f"Task bucket: {category}\n"
        "The bucket is for telemetry only. Do not memorize or branch on task IDs."
    ).strip()


def score_prompt(task_context: str, trajectory: str) -> str:
    return (
        f"{task_context}\n\n"
        "Trajectory to score:\n"
        f"{trajectory}\n\n"
        "Call record_score with exactly these fields: "
        f"score=<integer {MIN_SCORE}-{MAX_SCORE}>, critique=<one short sentence>, "
        "rubric_issue=<diagnostic issue>, severity=<none|minor|major|critical>."
    )


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
    return objects


def _coerce_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if MIN_SCORE <= score <= MAX_SCORE:
        return score
    return None


def _severity_from_score(score: int | None) -> str:
    if score is None:
        return "major"
    if score <= 20:
        return "critical"
    if score <= 60:
        return "major"
    if score <= 80:
        return "minor"
    return "none"


def _normalize_issue(value: Any) -> str:
    issue = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return issue if issue in RUBRIC_ISSUES else "unclear"


def _normalize_severity(value: Any, score: int | None) -> str:
    severity = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return severity if severity in SEVERITIES else _severity_from_score(score)


def _clean_critique(value: Any) -> str:
    critique = "No concise critique was recorded."
    if isinstance(value, str) and value.strip():
        critique = " ".join(value.strip().split())
    if len(critique) > MAX_CRITIQUE_CHARS:
        critique = critique[:MAX_CRITIQUE_CHARS].rstrip()
    return critique


def _record_from_parts(
    *,
    score: int | None,
    critique: Any = None,
    rubric_issue: Any = None,
    severity: Any = None,
) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "score": score,
        "critique": _clean_critique(critique),
        "rubric_issue": _normalize_issue(rubric_issue),
        "severity": _normalize_severity(severity, score),
    }


def parse_score(text: str) -> int | None:
    stripped = text.strip()
    score = _coerce_score(stripped)
    if score is not None:
        return score

    for obj in _json_objects(text):
        if isinstance(obj, dict) and "score" in obj:
            score = _coerce_score(obj.get("score"))
            if score is not None:
                return score

    matches = re.findall(r"\b(?:score|reward)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    for raw in reversed(matches):
        score = _coerce_score(raw)
        if score is not None:
            return score
    return None


def parse_text_record(text: str) -> dict[str, Any] | None:
    for obj in _json_objects(text):
        if not isinstance(obj, dict):
            continue
        score = _coerce_score(obj.get("score"))
        record = _record_from_parts(
            score=score,
            critique=obj.get("critique"),
            rubric_issue=obj.get("rubric_issue"),
            severity=obj.get("severity"),
        )
        if record is not None:
            return record
    return _record_from_parts(score=parse_score(text))


def parse_tool_record(raw_response: Any) -> dict[str, Any] | None:
    if not isinstance(raw_response, dict):
        return None
    content = raw_response.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        if block.get("name") != SCORE_TOOL_NAME:
            continue
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            continue
        record = _record_from_parts(
            score=_coerce_score(tool_input.get("score")),
            critique=tool_input.get("critique"),
            rubric_issue=tool_input.get("rubric_issue"),
            severity=tool_input.get("severity"),
        )
        if record is not None:
            return record
    return None


async def score_trace(ctx: Any, *, task_context: str, trajectory: str, label: str) -> dict[str, Any]:
    prompt = score_prompt(task_context, trajectory)
    response = await ctx.call_model(
        system=SCORER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0,
        extra_body=FORCED_SCORE_TOOL,
    )
    record = parse_tool_record(response.raw)
    if record is None:
        record = parse_text_record(response.text)

    repair_text = ""
    repair_raw = None
    repaired = False
    if record is None:
        repaired = True
        repair = await ctx.call_model(
            system=(
                SCORER_SYSTEM
                + "\nYou must call the record_score tool with score, critique, "
                "rubric_issue, and severity. Do not write prose."
            ),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
            extra_body=FORCED_SCORE_TOOL,
        )
        repair_text = repair.text
        repair_raw = repair.raw
        record = parse_tool_record(repair.raw)
        if record is None:
            record = parse_text_record(repair.text)
    if record is None:
        record = {
            "score": None,
            "critique": "No valid scalar score or critique was recorded.",
            "rubric_issue": "unclear",
            "severity": "major",
        }

    ctx.log_event(
        "score_trace",
        label=label,
        score=record["score"],
        critique=record["critique"],
        rubric_issue=record["rubric_issue"],
        severity=record["severity"],
        raw_text=response.text,
        raw_response=response.raw,
        repaired=repaired,
        repair_text=repair_text,
        repair_raw=repair_raw,
    )
    return {
        "score": record["score"],
        "critique": record["critique"],
        "rubric_issue": record["rubric_issue"],
        "severity": record["severity"],
        "raw_text": response.text,
        "raw_response": response.raw,
        "repair_text": repair_text,
        "repair_raw": repair_raw,
        "repaired": repaired,
        "usage": response.usage,
    }


async def run(ctx: Any):
    task = ctx.task
    task_context = render_task_context(task)

    ctx.log_event(
        "start",
        mechanism="tau3_pointwise_latent_reward",
        category=str(getattr(task, "category", "unknown") or "unknown"),
        ordering_label=str(getattr(task, "ordering_label", "unknown") or "unknown"),
        trajectory_label=str(getattr(task, "trajectory_label", "unknown") or "unknown"),
    )

    result = await score_trace(
        ctx,
        task_context=task_context,
        trajectory=str(getattr(task, "trajectory", "") or ""),
        label=str(getattr(task, "trajectory_label", "trajectory") or "trajectory"),
    )

    ctx.log_event(
        "score_output",
        score=result["score"],
        critique=result["critique"],
        rubric_issue=result["rubric_issue"],
        severity=result["severity"],
        valid_score=result["score"] is not None,
    )

    return ctx.finish(
        result["score"],
        score=result["score"],
        mechanism="pointwise_latent_reward",
        category=str(getattr(task, "category", "unknown") or "unknown"),
        output_mode="forced_tool_score",
        critique=result["critique"],
        rubric_issue=result["rubric_issue"],
        severity=result["severity"],
        model_text=result["raw_text"],
        model_raw=result["raw_response"],
        repair_text=result["repair_text"],
        repair_raw=result["repair_raw"],
        repaired=result["repaired"],
    )
