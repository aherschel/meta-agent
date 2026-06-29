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

## LLM providers (Bedrock · OpenRouter · Anthropic)

By default every Claude call routes through **AWS Bedrock**. To run on a host
that only has an OpenRouter or Anthropic key (no AWS access — e.g. inside the
ASP fleet), select a provider with `META_AGENT_LLM_PROVIDER`. It governs the
eval / judge / program-harness model calls (`invoke_model`) *and* the in-process
proposer; when it is not `bedrock`, **boto3 is never imported**.

| `META_AGENT_LLM_PROVIDER` | API | Base URL | Auth | Model id |
| ------------------------- | --- | -------- | ---- | -------- |
| `bedrock` *(default)* | Bedrock `invoke_model` | `AWS_REGION` | AWS cred chain | short name → inference profile |
| `openrouter` | OpenAI-compatible **Chat Completions** (`/chat/completions`, never `/responses`) | `OPENROUTER_BASE_URL` (def. `https://openrouter.ai/api/v1`) | `Authorization: Bearer $OPENROUTER_API_KEY` | raw slug, e.g. `minimax/minimax-m2.7` |
| `anthropic` | **Messages API** (`/v1/messages`) | `ANTHROPIC_BASE_URL` (def. `https://api.anthropic.com`) | `x-api-key: $ANTHROPIC_API_KEY` + `anthropic-version` | `claude-*` slug |

All provider HTTP honors the standard proxy env (`HTTPS_PROXY` / `HTTP_PROXY` /
`NO_PROXY`) via `httpx(trust_env=True)`, and uses the same equal-jitter
retry/backoff as the Bedrock path. Existing Bedrock/Azure behavior is unchanged
when the variable is unset.

Minimal OpenRouter loop (no AWS creds, no external CLI):

```bash
export META_AGENT_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
meta-agent loop \
  --benchmark benchmarks/example/benchmark.yaml \
  --baseline harnesses/starter/program_harness \
  --run-name openrouter-demo \
  --model minimax/minimax-m2.7 \
  --iterations 1
# then:
meta-agent propose --project openrouter-demo --harness program_harness
```

### CLI-free proposer

The proposer normally execs an external agent CLI (`claude`/`codex`). For
providers without those binaries, pass `--proposer-cli inprocess` (the default
when `META_AGENT_LLM_PROVIDER` is `openrouter`/`anthropic`): it drives the
selected provider directly through one forced-tool call and stages the candidate
file(s) — no `claude`/`codex` on PATH. Eval and proposer models default from
`META_AGENT_MODEL` / `META_AGENT_PROPOSER_MODEL` so neither is hardwired.

> **Note on the agentic `claude_agent_sdk` runtime.** The `anthropic` provider
> runs it natively (the SDK uses your `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`
> with the Bedrock flag cleared). The OpenAI-compatible OpenRouter Chat
> Completions API is fully supported by the **direct-call** paths
> (program/research/judge harnesses + the in-process proposer); the in-process
> agentic SDK runtime needs an Anthropic-compatible endpoint.

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
