# Tau3 Trajectory Pointwise Reward - Proposer Context

## Objective

You are optimizing a pointwise scalar reward harness for τ³/tau-airline
trajectory success. The benchmark contains pass/fail pairs, but each candidate
harness call is strictly pointwise:

```python
score = score_trace(task, trajectory)
return score
```

The adapter then runs the harness twice and compares scalar outputs:

```python
score_a = harness(task, trajectory_a)
score_b = harness(task, trajectory_b)
decision = "A>B" if score_a > score_b else "B>A"
```

Do not build a direct A/B judge. The scoring model should see exactly one
trajectory, no `Response A` / `Response B` label, and no unseen alternative.

## Why This Benchmark Exists

The downstream reward interface we care about is production-shaped:

```text
R(task, trajectory) -> scalar reward
```

That scalar can be used for failure mining, best-of-N trajectory selection, and
meta-agent search feedback. Pairwise τ³ accuracy is only the evaluation
adapter: a pair passes when the successful trajectory gets a higher scalar than
the failed trajectory.

## What The Program Harness Sees

The adapter passes a safe single-trajectory task object as `ctx.task`:

- `pair_id`
- `question`
- `trajectory`
- `category`
- `source`
- `ordering_label`
- `trajectory_label`
- `trajectory_ref`

The object does not expose the gold label, the other trajectory, pass/fail
labels, or pairwise response fields. `question` contains only pointwise scoring
framing for this trajectory.

## Required Contract

The final output must be a scalar score, mirrored in metadata:

```python
return ctx.finish(
    score,
    score=score,
    critique=critique,
    rubric_issue=rubric_issue,
    severity=severity,
)
```

The harness must not return `A>B` or `B>A`. The benchmark adapter owns the
deterministic comparison between two scalar scores. Exact ties are treated as
abstentions/wrong, not resolved by a direct pairwise LLM tie-breaker.

Keep the scaffold's forced `record_score` structured-output tool intact. The
smoke gate rejects candidates whose scoring call emits prose/free text, fails
to expose `output_mode="forced_tool_score"`, or does not include a
`record_score` tool-use block in `model_raw`. The forced tool must include:

```json
{
  "score": 47,
  "critique": "One concise sentence explaining the main reason for the score.",
  "rubric_issue": "policy_or_constraint_violation",
  "severity": "major"
}
```

Only `score` is consumed by the benchmark reward/comparator. `critique`,
`rubric_issue`, and `severity` are proposer-facing diagnostics for search-trace
analysis; do not use them as explicit comparator flags or pairwise tie-breakers.
You may improve prompting, trajectory rendering, score-scale design, evidence
extraction, and aggregation, but do not remove or weaken the structured
scalar-plus-diagnostic output plumbing.

## Target Criteria

The target is official τ³/tau-airline task success:

- Did the agent correctly identify the customer's goal?
- Did it verify identity and policy constraints before sensitive reads or
  database mutations?
- Did it read the right records and correctly interpret tool results?
- Did booking, cancellation, refund, exchange, passenger, seat, or baggage
  changes match policy and the requested outcome?
- Did it recover from tool errors and accurately communicate final state?

Surface polish, verbosity, and confidence should not dominate. A trajectory can
sound helpful while still failing if it violates policy, hallucinates tool
state, performs the wrong mutation, or leaves the actual customer request
unresolved.

Prefer a fine-grained scalar, such as an integer 1-100 latent reward. The
number is ordinal within the same task, not a globally calibrated percentage.

## Search Protocol

This wrapper uses the repaired balanced τ³ v2 splits:

- train: 25 tasks / 300 capped pairs.
- val: 9 tasks / 108 capped pairs.
- test: 9 tasks / 108 capped pairs.

The splits are balanced by pairable task group, pool pass rate, and baseline
judge difficulty. Primary optimization uses `position_swap: false` because a
true pointwise scorer never sees A/B position in the scoring prompt.
`judge-v2-balanced-test-swap-audit` exists only as a diagnostic to catch
rendering bugs or accidental A/B leakage.

Do not encode task IDs, trajectory IDs, split membership, actor model names, or
examples from validation/test.

## Useful Candidate Levers

Good candidates should improve scalar success estimation while preserving the
pointwise interface:

- trajectory rendering/compression for one trajectory at a time.
- extraction of user goal, identity/policy checks, tool calls, tool outputs,
  mutations, and final answer claims.
- score-scale design and consistency checks.
- lightweight independent scoring passes, if cost is justified.
- score-margin and concise critique diagnostics in metadata so future proposers
  can inspect failures.

Avoid:

- direct A/B comparison prompts.
- prompts that mention `Response A`, `Response B`, pass/fail, or winner.
- pairwise tie-breaker calls.
- long free-form rationales that distract the scorer or bloat trace artifacts.
- hardcoded task-specific answer patterns.
