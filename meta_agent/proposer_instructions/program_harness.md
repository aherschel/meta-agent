# Program Harness Proposer Instructions

Read `shared.md` for the shared proposer contract. This file defines the
`program_harness` target.

## Candidate Contract

`harness.py` exports one of:

```python
async def run(ctx): ...

class Harness:
    async def run(self, ctx): ...
```

The candidate is the full harness program. The benchmark gives you a safe
`ctx.task` object and helper APIs:

- `ctx.call_model(...)` for model calls through the repo-owned client.
- `ctx.run_command(...)` for candidate-owned local commands when appropriate.
- `ctx.log_event(...)` for interpretable runtime telemetry.
- `ctx.finish(final_output, **metadata)` for returning the final answer plus
  structured metadata.

The benchmark-specific `proposer_context.md` defines the exact task input shape,
required output convention, and any domain-specific constraints. Read it before
designing candidates.

## Editable Surface

Write one candidate file:

```text
staging/<candidate_name>/
  harness.py              # required candidate-owned harness
  proposal_notes.json     # required proposal rationale
```

Keep the implementation in `harness.py` by default. Do not create helper files
unless the outer prompt explicitly asks for them. This keeps candidates easy to
diff, audit, and compare across iterations.

Do not modify benchmark adapters, scorers, labels, split manifests, eval
runners, Modal/runtime files, hidden-holdout plumbing, or `_internal/` state.
Do not branch on task-specific identifiers, split membership, or leaked labels.

Think of this as an AutoAgent-style editable harness region, but scoped to a
Meta-Harness-style candidate artifact: the outer optimizer owns evaluation and
storage; your candidate owns the decision procedure.

## What The Program Can Control

For task agents, the program can control prompt construction, tools, state,
verification, retry logic, stop conditions, and subagent-style orchestration.

For judge agents, the same program surface can control trace rendering, rubric
routing, evidence extraction, model judgment, verdict checking, and final
verdict control.

The point is not to rephrase a prompt. The point is to make the decision
procedure explicit, inspectable, and easy to debug from traces.

A strong program-harness candidate moves at least one control step into code.
Examples of code-owned control include rendering, routing, tool/subroutine
selection, evidence extraction, verification, aggregation, retry/fallback,
parser enforcement, final-output computation, state management, or stop logic.

Concrete mechanisms you may add or change include:

- Prompt and message construction.
- Input rendering and trace/view compression.
- Candidate-local tools or helper functions.
- Multiple model-call phases, including planner/solver/verifier/critic passes.
- Subagent-style orchestration implemented as helper functions, classes, or
  additional model calls inside the candidate.
- Routing based on safe observable properties of `ctx.task`.
- Evidence extraction, checklists, rubrics, parsers, schema validation, and
  final-answer verification.
- Retry, repair, fallback, stop, and confidence logic.
- Candidate-local state or memory within a single task run.
- Cost, latency, token-budget, and output-length controls.
- Runtime telemetry through `ctx.log_event(...)`.

These are examples, not a closed list. If a mechanism is candidate-local,
general-purpose, and respects the benchmark contract, it is in scope.

Expose those mechanisms as named sections, functions, constants, or classes
inside `harness.py`, for example:

```text
harness.py
  PROMPTS / RUBRICS
  INPUT RENDERING
  ROUTING
  EVIDENCE EXTRACTION
  TOOLS / SUBROUTINES
  MODEL CALLS
  VERIFICATION / REPAIR
  PARSING / FINALIZATION
  STATE / ACTIONS
  async def run(ctx)
```

## Recommended File Shape

Prefer a readable, explicit `harness.py`:

```python
# Candidate-owned harness. Keep the candidate in this file.
# Required public contract: async def run(ctx) -> HarnessResult-compatible value.

# --- PROMPTS / RUBRICS -------------------------------------------------------
SYSTEM_PROMPT = "..."


# --- INPUT RENDERING ---------------------------------------------------------
def render_input(task):
    ...


# --- ROUTING / EVIDENCE ------------------------------------------------------
def choose_route(task):
    ...


# --- PARSING / FINALIZATION --------------------------------------------------
def parse_output(text):
    ...


# --- STATE / ACTIONS ----------------------------------------------------------
MAX_STEPS = 4


def init_state(task):
    return {"prompt": render_input(task), "final": None, "steps": []}


async def choose_next_action(ctx, state):
    if state["final"] is None:
        return {"type": "call_model"}
    return {"type": "finish"}


async def execute_action(ctx, state, action):
    if action["type"] == "call_model":
        return await ctx.call_model(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["prompt"]}],
            max_tokens=1024,
        )
    return None


def update_state(state, action, obs):
    state["steps"].append(action["type"])
    if action["type"] == "call_model":
        state["final"] = parse_output(obs.text)
    return state


# --- RUN LOOP ----------------------------------------------------------------
async def run(ctx):
    state = init_state(ctx.task)
    ctx.log_event("start", mechanism="...")
    for step in range(MAX_STEPS):
        action = await choose_next_action(ctx, state)
        ctx.log_event("action", step=step, action=action["type"])
        if action["type"] == "finish":
            return ctx.finish(state["final"], steps=state["steps"])
        obs = await execute_action(ctx, state, action)
        state = update_state(state, action, obs)
    return ctx.finish(state["final"], exhausted=True, steps=state["steps"])
```

## Good Program Candidates

Good candidates move control into code:

- Render task inputs into a clearer model-facing view.
- Extract salient evidence before asking the model to judge.
- Route to different procedures based on safe, observable input properties.
- Run a verifier pass that checks whether the final answer follows from the
  extracted evidence or intermediate state.
- Log events that explain what the harness did, without exposing labels or
  scoring internals.

Weak candidates are giant brittle rewrites or prompt-only variants. Prompt-only
changes are allowed when they are the intended hypothesis, but make that
explicit in `proposal_notes.json` and explain why a prompt-only change is the
right test. If the candidate cannot be explained as a decision procedure,
simplify it.

## Self-Critique

Before finishing:

- Can a human read `harness.py` and understand the decision flow?
- Is the candidate contained in one `harness.py`, unless explicitly permitted
  otherwise?
- Does the program avoid task-specific identifiers, hidden labels, and
  split-specific branches?
- Does `proposal_notes.json` name the mechanism, expected gains, likely
  regressions, and inspected traces?

For program-harness candidates, `proposal_notes.json` should also make the
control boundary clear:

```json
{
  "hypothesis": "...",
  "mechanism": "...",
  "control_moved_to_code": "...",
  "model_outputs": "...",
  "code_computes": "...",
  "why_not_prompt_only": "...",
  "expected_gains": {},
  "expected_regressions": {}
}
```

Before finishing, re-read the candidate and ask: could this have been
implemented as one longer system prompt with no code-owned decision step? If
yes, either simplify it into an explicit prompt-variant hypothesis or redesign
it so the harness program owns a real control step.
