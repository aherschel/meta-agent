# Tau3 Airline Actor - Program Harness Proposer Context

## Objective

You are optimizing a tau3 airline customer-service agent harness.

The harness is a `program_harness`: it owns the agent's control procedure for a
live multi-turn customer-service task. The benchmark owns the customer
simulator, airline database tools, search reward, and final official grader.

Your job is not to solve one task. Your job is to improve the reusable harness
so the same model follows airline policy, uses tools correctly, updates
external state correctly, and resolves customer requests more reliably.

## What The Program Harness Sees

The benchmark passes a safe task object as `ctx.task`.

Available fields and methods:

- `ctx.task.domain`: domain name, usually `airline`.
- `ctx.task.task_id`: stable task identifier. Use only for telemetry/debugging,
  never for branching.
- `ctx.task.policy`: airline policy text.
- `ctx.task.available_tools`: assistant-side airline tool schemas.
- `ctx.task.tools_description`: compact tool descriptions.
- `ctx.task.conversation`: customer/assistant message history so far.
- `ctx.task.tool_calls`: assistant tool-call history so far.
- `ctx.task.customer_has_ended`: whether the user simulator has ended.
- `ctx.task.last_customer_message`: latest customer message text.
- `await ctx.task.talk_to_customer(message)`: send one assistant message and
  receive the next customer message.
- `await ctx.task.call_tool(tool_name, arguments)`: call one airline database or
  environment tool.

The task object does not expose the hidden user scenario, evaluation criteria,
trusted labels, official reward, test split, or reward-model internals.

## Required Contract

The harness must define:

```python
async def run(ctx): ...
```

The harness should drive the full conversation until the customer ends, a
transfer is appropriate, or the step budget is exhausted.

Return via:

```python
return ctx.finish(
    final_output,
    mechanism="...",
    steps=...,
    customer_has_ended=ctx.task.customer_has_ended,
)
```

The final output is for logging only. Task success comes from the actual
conversation and tool calls.

## Search Feedback

For APB-style tau3 runs, search and validation feedback comes from the official
tau3 task evaluator. There is no reward model, no LLM judge critique, and no
separate evaluator harness in the loop.

Search traces may include:

- ordered action/observation events
- conversation messages
- tool calls and tool results
- official scalar reward / pass-fail outcome

Treat the score as a sparse task-level label. Infer the recurring failure mode
from the trace, tool history, policy, and final outcome rather than optimizing
for surface wording. The intended target is real tau3 task success: policy
compliance, correct tool use, correct external state, and correct customer
resolution.

## Trace Files

For chronological tau3 diagnosis, read ordered trace sidecars first:

- `per_task/*_action_sequence.jsonl`: primary source for ordered
  action/observation/grading flow.
- `per_task/*_events.jsonl`: full ordered telemetry, including harness events.
- `per_task/*_tau2_conversation.jsonl`: tau2-format conversation if needed.

Do not rely on `per_task/*_trace.jsonl` for chronological diagnosis. It is a
legacy summary and may group conversation messages separately from tool calls,
which can make actions appear out of order.

## What Correct Behavior Means

A successful tau3 airline agent should:

- identify the customer's actual goal
- ask for required identifying information before private reads or sensitive
  changes
- read the relevant user, reservation, and flight records before deciding
- apply airline policy before any mutation
- avoid state-changing tools when policy says the requested action is not
  allowed
- execute the exact allowed mutation when policy requires it
- recover from tool errors or missing information
- accurately communicate the final state to the customer
- continue the conversation until the customer ends or transfer/out-of-scope is
  appropriate

Tool success alone is not enough. A tool may execute even when the action was
policy-wrong. Surface politeness is not enough. A helpful-sounding answer that
violates policy or mutates the wrong state is a failure.

## Good Candidate Levers

Strong candidates move at least one control step into code:

- clearer prompt construction from policy, conversation, and tool history
- action schema enforcement and JSON repair
- explicit state machine for greet -> identify -> inspect -> decide -> mutate ->
  confirm
- routing by observable request type, such as cancel, change flight, baggage,
  passenger update, certificate, flight status, or transfer
- policy checklist before state-changing tools
- verification pass before mutations
- final-state confirmation after mutations
- guardrails against premature finish
- recovery logic after tool errors
- compact trace/event logging for later proposer diagnosis

Prompt-only changes are allowed only when the proposal is explicitly testing a
prompt hypothesis. Prefer small, evidence-backed changes over broad rewrites.

## Anti-Goodhart Rules

Do not add customer-facing text whose main purpose is to persuade the reward
evaluator, such as "I followed all policies" or "all constraints are
satisfied."

Do not fabricate summaries of tool results. Do not claim a database change
happened unless the tool result confirms it.

Do not branch on `task_id`, split membership, known reservation IDs, known user
IDs, hidden scenarios, reward values, judge model names, or evaluator harness
details.

Do not inspect or modify benchmark adapters, reward models, split manifests,
official graders, hidden holdout plumbing, or `_internal` files.

## What To Inspect

If prior candidates exist in the current experience store:

- read `candidate_index.json` if present
- inspect the accepted parent harness
- compare it with one or two lower-scoring candidates
- read traces for several lower-scoring tasks

If only the baseline exists:

- inspect the baseline harness
- read baseline traces
- identify one recurring, general failure mode before changing code

Look for recurring failure modes, not one-off task-specific fixes.

## Typical Trace Issues To Check For

- early stop before the customer ended
- missing identity verification
- read tool omitted before mutation
- mutation made despite policy disallowing it
- correct tool called with wrong arguments
- tool error ignored
- customer request misunderstood
- final message contradicts tool state
- excessive loops without progress

## Proposal Notes

Write `proposal_notes.json` with:

```json
{
  "hypothesis": "...",
  "mechanism": "...",
  "control_moved_to_code": "...",
  "inspected_tasks": ["..."],
  "expected_gains": {},
  "expected_regressions": {},
  "anti_goodhart_check": "..."
}
```
