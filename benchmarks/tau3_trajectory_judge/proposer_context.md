# Tau3 Trajectory Judge — Proposer Context

## What this benchmark is

You are optimizing a **pairwise trajectory judge**: an LLM that reads two customer-service agent conversations (same task, one pass, one fail) and picks the better one. The judge calls `submit_verdict(choice="A>B"|"B>A", rationale="...")` to finalize.

This judge will be used as a **reward model** for training an actor agent. Getting this right matters more than raw accuracy — a noisy or biased judge produces a noisy reward signal that derails actor training.

## What the judge sees per pair

1. A framing question describing the tau-airline domain
2. **Response A**: flattened conversation transcript (role-tagged messages + tool calls + tool results)
3. **Response B**: same format, different trajectory
4. One trajectory led to task success (correct policy + correct DB mutations), one didn't

Position is swapped and evaluated twice per pair. A pair counts as correct only if both orderings agree. This means **position-biased judges score ~0%, not ~50%**.

## If the active target is `program_harness`

The benchmark passes a safe task object as `ctx.task`. It deliberately does not
include the gold label.

Relevant fields:

- `task.question`: benchmark framing question.
- `task.response_a`: current-ordering Response A transcript.
- `task.response_b`: current-ordering Response B transcript.
- `task.pair_id`: stable pair identifier. Use only for telemetry/debugging,
  never for branching.
- `task.category`: task bucket such as `task19`. Use only as an aggregate-safe
  route label or telemetry field, not as a memorized answer key.
- `task.ordering_label`: `original` or `swapped`.
- `task.response_a_ref` / `task.response_b_ref`: whether current A/B came from
  the original pass/fail slot. These are for trace audit only; do not branch on
  them as a shortcut.
- `task.as_prompt()`: benchmark-owned rendering of the current A/B comparison.

Your program must return a normalized pairwise decision: `A>B` or `B>A`. The
cleanest convention is:

```python
return ctx.finish("A>B", decision="A>B", ...)
```

Strong Tau3 judge programs should make at least one decision-control step
explicit in code: trace rendering, policy/evidence checklist construction,
mutation audit, verdict parsing, consistency verification, retry/fallback, or
final-verdict aggregation. Prompt-only rewrites are valid only when that is the
intended hypothesis and should be labeled as such in `proposal_notes.json`.

Useful programmatic surfaces for this dataset:

- Render a concise current-ordering view that preserves tool calls, tool
  results, errors, and final user-facing claims.
- Extract evidence about identity verification, policy checks, tool-result
  grounding, database mutations, recovery from errors, and final task outcome.
- Ask the model for structured fields before the verdict, then let code parse
  and validate the final `A>B` / `B>A`.
- Add a verifier pass only when the first pass is low-confidence, malformed, or
  contradicted by extracted evidence.
- Log `ctx.log_event(...)` entries that make the candidate's route, checks, and
  finalization auditable in later traces.

Do not hardcode airline task IDs, trajectory IDs, split membership, actor model
names, or any pattern that only works because you inspected a specific labeled
pair.

## What "correct" means in this domain

The ground truth is **task success** — did the agent follow airline policy AND make the correct database changes? Trajectories that sound helpful but violate policy or misread tool output are failures. The judge must assess:

- Did the agent correctly interpret tool results (flight searches, booking lookups)?
- Did the agent verify policy constraints BEFORE executing mutations?
- Did tool success actually mean the action was policy-compliant? (Often it doesn't — the API executes invalid operations.)
- Did the agent fulfill the customer's actual request?

Surface quality (politeness, verbosity, formatting) is irrelevant to ground truth.

## Scoring

Metric is **macro-averaged pairwise accuracy across tasks** — each task contributes equally regardless of how many pairs it has. The pooled pair accuracy is also tracked but macro is the acceptance criterion.

## Reward model requirements

Since this judge becomes a reward signal:

- **Consistency > peak accuracy**: a judge that scores 70% reliably is more useful than one that scores 75% but is noisy across runs
- **False positives are worse than false negatives**: rewarding a bad trajectory reinforces bad actor behavior
- **Cost matters**: the judge runs at scale during actor training — $30/eval is 2x worse than $15/eval for the same accuracy
- **Generalization is mandatory**: holdout tasks are from the same domain but different task IDs. Train-only improvements that don't transfer indicate memorized patterns, not learned judging

## Known failure patterns from prior candidates

- **Hooks that return malformed output** (missing `hookEventName`) crash the Claude CLI with a ZodError. If you write a hook, test the return dict shape carefully — return `{}` for no-op, `{"decision": "block", "reason": "..."}` for PreToolUse blocking.
