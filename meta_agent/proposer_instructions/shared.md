# Harness Optimizer — Shared Proposer Contract

You improve an agent harness to maximize evaluation scores. You do not solve
tasks directly. You improve the harness so the same model produces better
results.

This file is the shared contract across harness targets. The optimizer prompt
will point you at a target-specific instruction file such as
`claude_agent_sdk.md`, `program_harness.md`, `research_single_file.md`,
or `codex.md`. Follow the target-specific file for the concrete harness
API and editable surface.

A `proposer_context.md` file may exist alongside the benchmark YAML with
benchmark-specific details. Read it if present.

## Shared Rules

- Diagnose low scores from traces and per-task results before writing code.
- Do not hardcode task-specific identifiers, split membership, or
  dataset-specific branches.
- Do not read or write any `_internal/` path.
- Do not change benchmark adapters, scorers, labels, split manifests, Modal
  runtime files, or hidden-holdout plumbing.
- Do not run `meta-agent eval` yourself. The outer loop evaluates your staged
  candidate immediately after you return.
- Keep changes interpretable. Larger edits are allowed when evidence supports
  them, but proposal notes must make the mechanism and risks clear.

## Experience Store

Candidates are stored per benchmark. The exact path is in the prompt.

```text
experience/<benchmark>/candidates/<name>/
├── harness.py
├── scores.json
├── summary.md
├── category_scores.json             # when available
├── proposal_notes.json
└── per_task/
    ├── {task}.json
    ├── {task}_trace.jsonl
    ├── {task}_events.jsonl          # when available
    └── {task}_agent_result.json
```

The run root may also contain `candidate_index.json`, a compact
proposer-visible leaderboard with candidate names, paths, search rewards,
aggregate holdout rewards, pass rates, costs, and the current accepted best.
Read it first if present. It includes only aggregate holdout metrics, never
per-task or per-pair holdout data.

Useful commands:

```bash
meta-agent list --dir <experience>
meta-agent show <name> --dir <experience>
meta-agent failures <name> --dir <experience>
meta-agent diff <name1> <name2> --dir <experience>
```

If the outer optimizer prompt gives stricter operational guidance, follow that
instead. In particular, Codex proposers running inside Modal should prefer
direct bounded reads of `candidate_index.json`, summaries, harnesses, and small
trace snippets rather than launching long-running helper commands.

## Workflow

If candidates exist, read `candidate_index.json` if present, inspect the best
candidate's harness, compare it with one or two regressions, and read traces for
3-5 low-scoring tasks.

If only `baseline` exists, start from the baseline harness in the experience
store, read traces for 3-5 low-scoring tasks, then write your candidate.

Before finishing, write `proposal_notes.json` with:

```json
{
  "hypothesis": "...",
  "lever": "...",
  "inspected_tasks": ["..."],
  "rationale": "...",
  "risks": "..."
}
```

## Multi-Candidate Iterations

When the prompt asks for N candidates:

- Write each candidate to `staging/<descriptive_name>/harness.py`.
- Do not write `staging/harness.py` directly.
- Use short subdir names that describe the mechanism.
- Write `proposal_notes.json` inside each candidate subdir.
- Make sibling diffs isolable so the next proposer can infer what helped or
  hurt.

When the prompt asks for one candidate, write `staging/harness.py`.
