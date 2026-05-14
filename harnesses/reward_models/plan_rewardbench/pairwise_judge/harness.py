"""Single-file program-harness baseline for Plan-RewardBench.

The benchmark adapter owns pair construction, position swaps, scoring, and trace
persistence. This candidate owns only the judge procedure: render the safe task,
call the evaluation model, parse a pairwise verdict, and return it.
"""

from __future__ import annotations

import re
from typing import Any

from benchmarks.plan_rewardbench.rubric_prompts import APPENDIX_C_RUBRICS


# --- PROMPTS / RUBRICS -------------------------------------------------------

MAX_STEPS = 4
VALID_DECISIONS = frozenset({"A>B", "B>A"})

SYSTEM_PROMPT = f"""You are an expert evaluator for tool-augmented agent trajectories.

Apply the fixed Plan-RewardBench Appendix C rubrics exactly. Do not invent new
criteria. Do not prefer a trajectory because it is longer, more confident, more
verbose, or listed first. Trust tool calls and tool responses over assistant
claims.

Compare Response A and Response B independently against the same rubric. Then
return exactly one final verdict token on its own line:

VERDICT: A>B

or

VERDICT: B>A

Use these exact rubrics as the fixed target criteria:

{APPENDIX_C_RUBRICS}
"""


# --- INPUT RENDERING ---------------------------------------------------------

def render_task(task: Any) -> str:
    return task.as_prompt() if hasattr(task, "as_prompt") else str(task)


# --- ROUTING -----------------------------------------------------------------

def choose_route(task: Any) -> str:
    return str(getattr(task, "category", "default") or "default")


# --- EVIDENCE EXTRACTION -----------------------------------------------------

def extract_evidence(rendered_task: str) -> str:
    # Baseline keeps the full benchmark-owned rendering.
    return rendered_task


# --- PARSING / FINALIZATION --------------------------------------------------

def parse_verdict(text: str) -> str | None:
    upper = text.upper()
    matches = re.findall(r"VERDICT\s*:\s*(A\s*>\s*B|B\s*>\s*A)", upper)
    if matches:
        return matches[-1].replace(" ", "")
    if "A>B" in upper and "B>A" not in upper:
        return "A>B"
    if "B>A" in upper and "A>B" not in upper:
        return "B>A"
    return None


def normalize_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    decision = value.upper().replace(" ", "")
    return decision if decision in VALID_DECISIONS else None


# --- VERIFICATION / REPAIR ---------------------------------------------------

async def repair_verdict(ctx, text: str) -> str | None:
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
    rendered = render_task(task)
    return {
        "route": route,
        "rendered": rendered,
        "evidence": extract_evidence(rendered),
        "model_text": "",
        "decision": None,
        "repaired": False,
        "repair_attempted": False,
        "steps": [],
        "finish_reason": None,
    }


async def choose_next_action(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Choose the next judge-control step."""
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
            max_tokens=1600,
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
    elif action_type == "repair":
        state["repair_attempted"] = True
        state["decision"] = normalize_decision(obs)
        state["repaired"] = state["decision"] is not None
    return state


def finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Code-owned finalization: only emit normalized pairwise decisions."""
    decision = normalize_decision(state["decision"])
    if decision is not None:
        state["finish_reason"] = "valid_decision"
    else:
        state["finish_reason"] = "invalid_or_missing_decision"
    state["decision"] = decision
    return state


# --- RUN LOOP ----------------------------------------------------------------

async def run(ctx):
    state = init_state(ctx.task)
    ctx.log_event("start", mechanism="plan_rb_program_baseline", route=state["route"])

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
        model_text=state["model_text"],
        steps=state["steps"],
        finish_reason=state["finish_reason"],
    )
