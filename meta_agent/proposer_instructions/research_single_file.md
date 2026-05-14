# Research Single-File Proposer Instructions

Read `shared.md` for the shared proposer contract. This file defines the
`research_single_file` target.

## Harness Contract

`harness.py` exports:

```python
def build_harness(ctx: RunContext) -> ResearchHarnessSpec: ...
```

The harness owns prompt/context/examples and a small set of runtime-facing
knobs. The repo-owned task runner owns the task loop, workspace setup,
verification, and scoring.

Useful imports:

```python
from meta_agent.harness_contracts.research import (
    ResearchExample,
    ResearchHarnessSpec,
    ResearchRuntimeSettings,
)
```

## Editable Surface

- `system_instructions`: durable behavior instructions.
- `task_context`: additional context derived from `ctx`.
- `examples`: small few-shot examples as `ResearchExample`.
- `max_attempts`: `1` or `2`.
- `runtime_settings`: approval policy and sandbox.

## Constraints

- Do not write adapter or task-runner code.
- Do not hardcode task names or verifier details.
- Keep the single file interpretable.
- Prefer mechanism-level changes over string-only prompt churn.

## Good Candidates

Good candidates usually change:

- the workflow encoded in `system_instructions`,
- a concise few-shot pattern,
- retry behavior through `max_attempts`,
- runtime settings when trace evidence shows the default blocks useful work.

Write the candidate to the staging path from the prompt, plus
`proposal_notes.json` as described in `shared.md`.
