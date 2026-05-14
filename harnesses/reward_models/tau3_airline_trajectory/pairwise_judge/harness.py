"""Program-harness baseline for the tau3 trajectory judge.

The adapter owns pair construction, position swaps, scoring, and trace
persistence. This candidate owns the judge procedure: render the safe pair,
call the evaluation model, parse a normalized pairwise decision, optionally
repair malformed output, and return the final verdict.
"""

from __future__ import annotations

import json
import re
from typing import Any


# --- PROMPTS / RUBRICS -------------------------------------------------------

MAX_STEPS = 4
VALID_DECISIONS = frozenset({"A>B", "B>A"})

SYSTEM_PROMPT = """You are a strict evaluator for tau-airline customer-service trajectories.

You compare two transcripts for the same customer task. One trajectory may sound
helpful while still failing the task. Prefer the trajectory that more correctly
follows airline policy, interprets tool results, performs required database
mutations, and resolves the customer's actual request.

Use concrete transcript evidence. Do not prefer a trajectory because it is
longer, more polite, more confident, or listed first. Tool success alone is not
enough: the agent must have been allowed to take the action and must have taken
the right action for the user's request.

Check, in order:
1. User goal: what did the customer actually ask for?
2. Eligibility and policy: did the agent verify identity and policy constraints
   before sensitive reads or mutations?
3. Tool grounding: did the agent read the right records and correctly interpret
   tool results?
4. Mutation correctness: did booking, cancellation, refund, exchange, passenger,
   seat, or baggage changes match the policy and requested outcome?
5. Recovery and communication: did the agent recover from errors and accurately
   communicate the final state?

Return a compact JSON object followed by one verdict line:
{"winner":"A>B"|"B>A","confidence":"low"|"medium"|"high","key_reason":"..."}
VERDICT: A>B

or

VERDICT: B>A
"""


# --- INPUT RENDERING ---------------------------------------------------------

def render_task(task: Any) -> str:
    return task.as_prompt() if hasattr(task, "as_prompt") else str(task)


def render_judge_prompt(task: Any) -> str:
    rendered = render_task(task)
    return (
        f"{rendered}\n\n"
        "Decide which trajectory is better for the current A/B ordering. "
        "There are no ties. Use A>B if Response A is better; use B>A if "
        "Response B is better."
    )


# --- ROUTING / EVIDENCE ------------------------------------------------------

def choose_route(task: Any) -> str:
    return str(getattr(task, "category", "default") or "default")


def extract_evidence(rendered_task: str) -> str:
    # Keep the complete adapter-rendered transcript. Tau3 failures often hinge
    # on a small tool result or mutation detail that crude truncation can erase.
    return rendered_task


# --- PARSING / FINALIZATION --------------------------------------------------

def normalize_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    decision = value.upper().replace(" ", "")
    return decision if decision in VALID_DECISIONS else None


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_verdict(text: str) -> str | None:
    for obj in reversed(_extract_json_objects(text)):
        decision = normalize_decision(obj.get("winner"))
        if decision is not None:
            return decision

    upper = text.upper()
    matches = re.findall(r"VERDICT\s*:\s*(A\s*>\s*B|B\s*>\s*A)", upper)
    if matches:
        return matches[-1].replace(" ", "")
    if "A>B" in upper and "B>A" not in upper:
        return "A>B"
    if "B>A" in upper and "A>B" not in upper:
        return "B>A"
    return None


def extract_key_reason(text: str) -> str:
    for obj in reversed(_extract_json_objects(text)):
        reason = obj.get("key_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()[:800]
    return text.strip()[:800]


# --- VERIFICATION / REPAIR ---------------------------------------------------

async def repair_verdict(ctx: Any, text: str) -> str | None:
    response = await ctx.call_model(
        system=(
            "Extract exactly one pairwise verdict from the judge text. "
            "Return only A>B or B>A."
        ),
        messages=[{"role": "user", "content": text}],
        max_tokens=8,
        temperature=0,
    )
    return normalize_decision(response.text) or parse_verdict(response.text)


# --- STATE / ACTIONS ----------------------------------------------------------

def init_state(task: Any) -> dict[str, Any]:
    route = choose_route(task)
    rendered = render_judge_prompt(task)
    return {
        "route": route,
        "rendered": rendered,
        "evidence": extract_evidence(rendered),
        "model_text": "",
        "decision": None,
        "key_reason": "",
        "repaired": False,
        "repair_attempted": False,
        "steps": [],
        "finish_reason": None,
    }


async def choose_next_action(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    if not state["model_text"]:
        return {"type": "judge"}
    if state["decision"] is None and not state["repair_attempted"]:
        return {"type": "repair"}
    return {"type": "finish"}


async def execute_action(ctx: Any, state: dict[str, Any], action: dict[str, Any]) -> Any:
    action_type = action.get("type")
    if action_type == "judge":
        return await ctx.call_model(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["evidence"]}],
            max_tokens=1800,
            temperature=0,
        )
    if action_type == "repair":
        return await repair_verdict(ctx, state["model_text"])
    if action_type == "finish":
        return None
    raise ValueError(f"unknown action: {action_type}")


def update_state(state: dict[str, Any], action: dict[str, Any], obs: Any) -> dict[str, Any]:
    action_type = action.get("type")
    state["steps"].append(action_type)
    if action_type == "judge":
        state["model_text"] = obs.text
        state["decision"] = parse_verdict(obs.text)
        state["key_reason"] = extract_key_reason(obs.text)
    elif action_type == "repair":
        state["repair_attempted"] = True
        state["decision"] = normalize_decision(obs)
        state["repaired"] = state["decision"] is not None
    return state


def finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    decision = normalize_decision(state["decision"])
    state["finish_reason"] = (
        "valid_decision" if decision is not None else "invalid_or_missing_decision"
    )
    state["decision"] = decision
    return state


# --- RUN LOOP ----------------------------------------------------------------

async def run(ctx: Any):
    state = init_state(ctx.task)
    ctx.log_event("start", mechanism="tau3_program_baseline", route=state["route"])

    for step in range(MAX_STEPS):
        action = await choose_next_action(ctx, state)
        ctx.log_event("action", step=step, action=action.get("type"))
        if action.get("type") == "finish":
            break
        obs = await execute_action(ctx, state, action)
        state = update_state(state, action, obs)

    state = finalize_state(state)
    ctx.log_event(
        "verdict",
        decision=state["decision"],
        repaired=state["repaired"],
        repair_attempted=state["repair_attempted"],
        finish_reason=state["finish_reason"],
    )

    return ctx.finish(
        state["decision"],
        decision=state["decision"],
        route=state["route"],
        repaired=state["repaired"],
        repair_attempted=state["repair_attempted"],
        key_reason=state["key_reason"],
        model_text=state["model_text"],
        steps=state["steps"],
        finish_reason=state["finish_reason"],
    )
