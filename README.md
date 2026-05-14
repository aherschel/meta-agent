<h1 align="center">meta-agent</h1>

<p align="center">
  <strong>Recursive self-improvement for agents, starting with the harness.</strong>
</p>

<p align="center">
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="#learn-more">Learn more</a>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Backed by Y Combinator" src="https://img.shields.io/badge/backed%20by-Y%20Combinator-F26625">
</p>

---

meta-agent is an open-source framework for automatic harness optimization.
System prompts, tools, hooks, stop conditions, subagents, and control flow
become editable surfaces that the optimizer rewrites from execution traces.

It reads execution traces from the current harness, proposes a targeted change,
evaluates the candidate on a search split, and keeps it only when the held-out
score improves.

> meta-agent is an open-source project from **Canvas Labs**.

## Results

### Agent harness optimization

On tau-bench v3 airline, meta-agent improved a frozen Haiku 4.5 agent from 67%
to 87% holdout within 4 to 10 iterations. No fine-tuning, no model changes, no
benchmark changes.

The optimizer rewrote the harness around the model: system instructions,
tool-use discipline, stop hooks, turn budget, and control flow. The model stayed
fixed; the agent system got better around it.

| Benchmark        | Domain                   | Baseline | Optimized       | Setup                                        |
| ---------------- | ------------------------ | -------- | --------------- | -------------------------------------------- |
| **tau-bench v3** | airline customer service | 67%      | **87%** holdout | 50 tasks, Haiku 4.5 agent, Opus 4.6 proposer |

### Additional use case: evaluator harnesses

The same loop can also tune evaluator harnesses: how an LLM judge renders
trajectories, extracts evidence, and structures its verdict. This is useful
when the agent task itself has no simple verifier. Examples are included for
Plan-RewardBench and tau3 trajectory judging, but the primary use case is agent
harness optimization.

## How it works

```
propose harness → validate → evaluate on search split → keep if holdout improves → repeat
```

The harness is one Python file with one entrypoint:

```python
async def run(ctx):
    result = await ctx.call_model(
        system="You are a careful task solver.",
        messages=[{"role": "user", "content": str(ctx.task)}],
        max_tokens=1024,
    )
    return ctx.finish(result.text.strip())
```

The benchmark adapter owns task selection, labels, and scoring. The harness
owns the decision procedure. The proposer reads prior candidates and traces
before writing the next one. Acceptance is gated on a holdout split that the
proposer never sees at the per-task level.

## Quickstart

```bash
git clone https://github.com/canvas-org/meta-agent
cd meta-agent
pip install -e .
meta-agent --help
```

Run an optimization loop:

```bash
meta-agent loop \
  --benchmark benchmarks/plan_rewardbench/benchmark.yaml:search \
  --holdout benchmarks/plan_rewardbench/benchmark.yaml:val \
  --baseline harnesses/reward_models/plan_rewardbench/pairwise_judge \
  --run-name plan-rb-demo \
  --iterations 5
```

Inspect results with `meta-agent list`, `meta-agent diff`, and
`meta-agent failures`. Run on [Modal](./meta_agent/cloud/MODAL.md) for longer
searches.

**Prerequisites**: Python 3.11+. Codex-based runs need `OPENAI_API_KEY`.
Claude-based runs need AWS Bedrock credentials. See
[`.env.example`](./.env.example).

<details>
<summary><strong>Repo layout</strong></summary>

```
meta_agent/
  core/                        benchmarks, adapters, experience store, targets
  commands/                    CLI command implementations
  loop/                        propose / validate / evaluate / accept loop
  task_runner/                 runtime dispatch and execution
  harness_contracts/           program / Claude SDK / research harness loaders
  cloud/                       Modal deployment
  proposer_instructions/       prompts the proposer reads

benchmarks/
  tau3/                        tau-bench v3 (agent)
  plan_rewardbench/            Plan-RewardBench (reward model)
  tau3_trajectory_judge/       tau3 trajectory judge (reward model)
  ...

harnesses/
  starter/program_harness/     minimal program harness template
  agents/tau3_airline/         tau3 customer-service reference harness
  reward_models/
    plan_rewardbench/
    tau3_airline_trajectory/
```

</details>

## Learn more

- [`meta_agent/proposer_instructions/program_harness.md`](./meta_agent/proposer_instructions/program_harness.md) — program harness contract
- [`meta_agent/cloud/MODAL.md`](./meta_agent/cloud/MODAL.md) — running on Modal

## License

MIT. See [`LICENSE`](./LICENSE).
