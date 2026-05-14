"""Reference tau3 customer-service agent harness.

This harness captures the main mechanism recovered from the original
tau-bench v3 agent-harness experiments: make tool use explicit, ground every
claim in tool results, handle the full customer request, and prevent premature
stops before the customer has been told the outcome.

It is a reference starting point for tau3-style customer-service agents, not a
byte-for-byte copy of a historical optimized candidate.
"""

from __future__ import annotations

import os
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from meta_agent.core.run_context import RunContext


CUSTOMER_SERVICE_GUIDELINES = """\
You are a customer-service agent operating through tools.

Core operating rules:

1. Use tools instead of narrating tool use.
   When you need account, reservation, flight, policy, or state information,
   call the relevant tool first. Do not claim that you looked something up
   unless a tool call actually returned the evidence.

2. Look up facts before stating them.
   Before making claims about reservations, flights, passengers, payments,
   membership, status, fees, or available options, inspect the relevant records.
   Tool output is the source of truth.

3. Attempt the requested action through the system.
   If a customer asks to cancel, modify, book, upgrade, remove baggage, send a
   certificate, or transfer, use the relevant tool path. Do not invent policy
   restrictions. If the system rejects the action, explain that result clearly.

4. Follow the full service cycle.
   Identify the customer or reservation, inspect the relevant records, explain
   the plan, get confirmation before changes, execute the tool call, and report
   the result back to the customer.

5. Handle every part of the request.
   If the customer asks for multiple things, resolve each one or explain why it
   cannot be completed. If the customer changes direction mid-conversation,
   address the new request instead of stopping early.

6. Respect customer conditions.
   If the customer gives a condition such as "only cancel if I get a refund,"
   preserve that condition while acting. Do not override explicit customer
   preferences unless the tool or policy result requires it.

7. Transfer with context.
   When transferring to a human agent, include the customer's request, what you
   tried, tool results or blockers, and the customer's latest message.
"""


async def require_customer_update_before_stop(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Reject the first stop attempt so the agent reports the outcome."""
    if input_data.get("stop_hook_active", False):
        return {}

    return {
        "reason": (
            "Before stopping, tell the customer what happened using the customer "
            "communication tool. If you performed a database action, summarize "
            "the final state. If the customer has not ended the conversation, "
            "ask whether they need anything else."
        ),
        "continue_": True,
    }


def build_options(ctx: RunContext) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=ctx.cwd,
        model=ctx.model,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": CUSTOMER_SERVICE_GUIDELINES,
        },
        tools={"type": "preset", "preset": "claude_code"},
        permission_mode=os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions"),
        max_turns=200,
        max_budget_usd=10.0,
        thinking={"type": "adaptive"},
        hooks={
            "Stop": [
                HookMatcher(hooks=[require_customer_update_before_stop]),
            ],
        },
    )
