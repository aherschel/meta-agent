# Plan-RewardBench Pointwise Reward - Proposer Context

## Objective

You are optimizing a pointwise trajectory reward harness for Plan-RewardBench.
The benchmark still contains chosen/rejected pairs, but each candidate harness
call is strictly pointwise:

```python
score = score_trace(task, tools, trajectory)
return score
```

The adapter then runs the harness twice and compares the two scalar outputs:

```python
score_a = harness(task, trajectory_a)
score_b = harness(task, trajectory_b)
decision = "A>B" if score_a > score_b else "B>A"
```

Do not build a direct A/B judge. The model call that scores a trajectory should
see exactly one trajectory, no `Response A` / `Response B` label, and no unseen
alternative trajectory.

## Why This Benchmark Exists

The downstream reward interface we care about is production-shaped:

```text
R(task, trajectory) -> scalar reward
```

That scalar can be used for failure mining, best-of-N trajectory selection, and
meta-agent search feedback. Pairwise Plan-RB accuracy is only the evaluation
adapter: a pair passes when the chosen trajectory gets a higher scalar than the
rejected trajectory.

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

The object does not expose the gold label, the other trajectory, chosen/rejected
labels, or any pairwise response fields. `question` contains only the user task
and available tool environment for this scoring call.

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
  "rubric_issue": "stale_constraint",
  "severity": "major"
}
```

Only `score` is consumed by the benchmark reward/comparator. `critique`,
`rubric_issue`, and `severity` are proposer-facing diagnostics for search-trace
analysis; do not use them as explicit comparator flags or pairwise tie-breakers.
You may improve prompting, trajectory rendering, rubric routing, and score
interpretation, but do not remove or weaken the structured scalar-plus-diagnostic
output plumbing.

## Fixed Target Criteria

Use the Plan-RewardBench Appendix C rubrics as the target criteria. They live in:

```text
benchmarks/plan_rewardbench/rubric_prompts.py
```

Route by category:

- `planning_single_easy`, `planning_single_hard`, `planning_multi_easy`,
  `planning_multi_hard`: Planning rubric.
- `planning_robustness`: Robustness rubric.
- `refusal`: Safety refusal rubric.
- `irrelevance_unavailable`: Tool irrelevance / unavailability rubric.

The scalar should absorb task completion, grounding in tool outputs, safety,
robustness, redundancy, stale constraints, hallucinated tool use/results, and
final answer quality. Treat the rubric's original JSON output schemas as source
paper context only; do not ask the scorer to output those schemas or expose
hand-coded flags to the comparator.

Prefer a fine-grained scalar, such as an integer 1-100 latent reward, because
exact ties produce weak search signal. The number is still only ordinal within
the same task, not a globally calibrated percentage.

## Search Protocol

This wrapper uses grouped Plan-RB splits:

- search examples are grouped by normalized task/query, with UUID fallback.
- validation and test are disjoint by pair id, normalized query, group, and UUID.
- primary optimization runs with `position_swap: false` because a true pointwise
  scorer never sees A/B position in the scoring prompt.
- `grouped-test-50-swap-audit` exists only as a diagnostic to catch rendering
  bugs or accidental A/B leakage.

Do not encode task-specific answer patterns, pair IDs, split membership, UUIDs,
or examples from the validation/test manifests.

## Useful Candidate Levers

Good candidates should improve scalar reward estimation while preserving the
pointwise interface:

- rubric routing and score-scale design.
- trajectory rendering/compression for one trajectory at a time.
- extraction of user task, tool definitions, tool calls, tool outputs, and final
  assistant answer.
- stricter score parsing and retry/repair for malformed scalar outputs.
- lightweight independent scoring passes, if cost is justified.
- score-margin and concise critique diagnostics in metadata so future proposers
  can inspect failures.

Avoid:

- direct A/B comparison prompts.
- prompts that mention `Response A`, `Response B`, chosen/rejected, or winner.
- pairwise tie-breaker calls.
- comparator logic using explicit flags instead of scalar scores.
- long free-form rationales that distract the scorer or bloat trace artifacts.
- overfitting to `grouped-val`; final evidence comes from grouped test.
