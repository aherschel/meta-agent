"""Generic starter for the `program_harness` target.

This file is intentionally plain Python. The proposer may modify this entire
file. Keep the candidate in this file by default; expose different harness
surfaces as named constants, functions, classes, or sections. The benchmark
adapter and scorer live outside the candidate directory and must not be edited.
"""

from __future__ import annotations

from typing import Any


# ============================================================================
# CANDIDATE-OWNED HARNESS
# ============================================================================
#
# Required public contract:
#
#   async def run(ctx): ...
#
# The outer runner supplies:
#
#   ctx.task          safe benchmark-specific task object
#   ctx.model         model name chosen by the outer eval
#   ctx.call_model    repo-owned model-call helper
#   ctx.run_command   local command helper, when appropriate
#   ctx.log_event     structured telemetry for traces
#   ctx.finish        return helper
#
# You may change any section below:
#
#   PROMPTS / RUBRICS
#   INPUT RENDERING
#   ROUTING
#   EVIDENCE EXTRACTION
#   TOOLS / SUBROUTINES
#   MODEL CALLS
#   VERIFICATION / REPAIR
#   PARSING / FINALIZATION
#   RUN LOOP
#
# Do not read labels, scorer files, split manifests, hidden holdout data,
# benchmark adapters, Modal/runtime files, or _internal state.
# ============================================================================


# --- PROMPTS / RUBRICS -------------------------------------------------------

MAX_STEPS = 4

SYSTEM_PROMPT = """You are a careful task solver.

Read the task input, identify the relevant evidence, and return the best final
answer according to the benchmark-specific instructions.
"""


# --- INPUT RENDERING ---------------------------------------------------------

def render_task(task: Any) -> str:
    """Render a safe, generic task object for the model."""
    if isinstance(task, dict):
        lines = []
        for key, value in task.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    return str(task)


# --- ROUTING -----------------------------------------------------------------

def choose_route(task: Any) -> str:
    """Choose a candidate-local procedure from safe observable task properties."""
    if isinstance(task, dict):
        return str(task.get("category") or task.get("type") or "default")
    return "default"


# --- EVIDENCE EXTRACTION -----------------------------------------------------

def extract_evidence(rendered_task: str) -> str:
    """Extract or compress salient evidence before the model call."""
    return rendered_task


# --- VERIFICATION / REPAIR ---------------------------------------------------

def verify_output(final: str) -> tuple[bool, str]:
    """Return whether the parsed final answer is acceptable and why."""
    if final:
        return True, "non_empty"
    return False, "empty_output"


# --- PARSING / FINALIZATION --------------------------------------------------

def parse_model_output(text: str) -> str:
    """Default parser: return the model text as the final answer."""
    return text.strip()


# --- STATE / ACTIONS ----------------------------------------------------------

def init_state(task: Any) -> dict[str, Any]:
    rendered = render_task(task)
    return {
        "route": choose_route(task),
        "rendered": rendered,
        "evidence": extract_evidence(rendered),
        "model_text": "",
        "final": "",
        "verified": False,
        "verification_attempted": False,
        "verification_reason": "not_checked",
        "steps": [],
    }


async def choose_next_action(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Choose the next candidate-owned control step."""
    if not state["model_text"]:
        return {"type": "call_model"}
    if not state["verified"] and not state["verification_attempted"]:
        return {"type": "verify"}
    return {"type": "finish"}


async def execute_action(ctx: Any, state: dict[str, Any], action: dict[str, Any]) -> Any:
    """Execute one control action and return an observation."""
    action_type = action.get("type")
    if action_type == "call_model":
        return await ctx.call_model(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["evidence"]}],
            max_tokens=1024,
        )
    if action_type == "verify":
        return verify_output(state["final"])
    if action_type == "finish":
        return None
    raise ValueError(f"unknown action: {action_type}")


def update_state(state: dict[str, Any], action: dict[str, Any], obs: Any) -> dict[str, Any]:
    """Update candidate-local state after an action."""
    action_type = action.get("type")
    state["steps"].append(action_type)
    if action_type == "call_model":
        state["model_text"] = obs.text
        state["final"] = parse_model_output(obs.text)
    elif action_type == "verify":
        verified, reason = obs
        state["verification_attempted"] = True
        state["verified"] = verified
        state["verification_reason"] = reason
    return state


def fallback_answer(state: dict[str, Any]) -> str:
    """Return the best available answer after the step budget is exhausted."""
    return state.get("final") or state.get("model_text") or ""


# --- RUN LOOP ----------------------------------------------------------------

async def run(ctx):
    """Run the candidate harness for one task with explicit control flow."""
    state = init_state(ctx.task)
    ctx.log_event("start", mechanism="generic_program_harness", route=state["route"])

    for step in range(MAX_STEPS):
        action = await choose_next_action(ctx, state)
        ctx.log_event("action", step=step, action=action.get("type"))

        if action.get("type") == "finish":
            return ctx.finish(
                state["final"],
                model_text=state["model_text"],
                verified=state["verified"],
                verification_reason=state["verification_reason"],
                route=state["route"],
                steps=state["steps"],
            )

        obs = await execute_action(ctx, state, action)
        state = update_state(state, action, obs)

    final = fallback_answer(state)
    return ctx.finish(
        final,
        model_text=state["model_text"],
        verified=state["verified"],
        verification_reason=state["verification_reason"],
        route=state["route"],
        steps=state["steps"],
        exhausted=True,
    )
