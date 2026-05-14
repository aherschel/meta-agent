# Running meta-agent on Modal

Use Modal when you want `meta-agent loop` or `meta-agent eval` to keep running
after you close your laptop.

## Setup

Install and authenticate Modal:

```bash
pip install modal
modal token new
```

Create the secrets used by the Modal runner:

```bash
source .env

modal secret create bedrock-creds \
  AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  AWS_REGION="${AWS_REGION:-us-east-1}"

modal secret create openai-key OPENAI_API_KEY="$OPENAI_API_KEY"

python -m meta_agent.cloud.modal_runner upload-codex-auth
```

Check that Modal can see them:

```bash
modal secret list | grep -E "bedrock-creds|openai-key|codex-auth"
```

## Run a Loop

```bash
modal run --detach meta_agent/cloud/modal_runner.py::loop \
  --benchmark benchmarks/tau3/benchmark.yaml:search \
  --holdout benchmarks/tau3/benchmark.yaml:holdout \
  --run-name tau3-modal \
  --baseline harnesses/agents/tau3_airline \
  --iterations 10 \
  --model claude-sonnet-4-6 \
  --proposer-model gpt-5.3-codex \
  --proposer-cli codex \
  --concurrency 100
```

Watch logs:

```bash
modal app logs meta-agent
modal app logs meta-agent --tail 50
```

## Run an Eval

```bash
modal run --detach meta_agent/cloud/modal_runner.py::eval \
  --benchmark benchmarks/tau3/benchmark.yaml:holdout \
  --config experience/tau3-modal/candidates/evo_007 \
  --name tau3_holdout_eval \
  --model claude-sonnet-4-6 \
  --concurrency 100
```

## Pull Results

Modal writes results to the `meta-agent-experience` volume. Pull a run back
into local `experience/`:

```bash
modal volume get meta-agent-experience tau3-modal ./experience/
```

Inspect locally:

```bash
meta-agent list     --benchmark tau3-modal
meta-agent pareto   --benchmark tau3-modal
meta-agent show     evo_007 --benchmark tau3-modal
meta-agent failures evo_007 --benchmark tau3-modal
meta-agent diff     baseline evo_007 --benchmark tau3-modal
```

## Optional: Deploy First

For repeated long runs, deploy the app once:

```bash
modal deploy meta_agent/cloud/modal_runner.py
```

Then launch against the deployed function:

```bash
python -m meta_agent.cloud.modal_runner launch-loop \
  --benchmark benchmarks/tau3/benchmark.yaml:search \
  --holdout benchmarks/tau3/benchmark.yaml:holdout \
  --run-name tau3-modal \
  --baseline harnesses/agents/tau3_airline \
  --iterations 10 \
  --model claude-sonnet-4-6 \
  --proposer-model gpt-5.3-codex \
  --proposer-cli codex \
  --concurrency 100
```
