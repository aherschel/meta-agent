# Claude Agent SDK Proposer Instructions

Read `shared.md` for the shared proposer contract. This file defines the
Claude Agent SDK target.

## Harness Contract

`harness.py` exports:

```python
def build_options(ctx: RunContext) -> ClaudeAgentOptions: ...
```

`ctx` carries runtime-owned values such as `cwd`, `model`, and
`task_instruction`. Everything else on `ClaudeAgentOptions` is the editable
harness surface.

## Editable Levers

| Lever | What it controls |
| --- | --- |
| `system_prompt` | Role, priorities, rubric, bias controls. |
| `allowed_tools` / `disallowed_tools` | Which tools the agent can call. Extend required tools; do not exclude them. |
| `mcp_servers` + custom MCP tools | New capabilities via `@tool(...)` and `create_sdk_mcp_server(...)`. |
| `hooks` | Lifecycle callbacks at `PreToolUse`, `PostToolUse`, `Stop`, `UserPromptSubmit`. |
| `agents` | Subagent definitions for narrow delegated roles. |
| `permission_mode` | Usually `"bypassPermissions"` for unattended eval runs. |
| `max_turns` | Turn budget per task. |
| `thinking` | Adaptive or fixed-budget extended thinking. |
| `tools` preset | Claude Code tools such as Bash/Read/Write/Edit/Glob/Grep when appropriate. |

## Runtime-Owned Fields

- Pass through `ctx.cwd` as `cwd=ctx.cwd`.
- Pass through `ctx.model` as `model=ctx.model`.
- Do not shadow benchmark-injected MCP servers or required tools such as
  `submit_verdict`.
- Do not write benchmark adapter or scoring code.

## Good SDK Candidates

Good candidates change a real mechanism:

- A hook that intercepts or rewrites model behavior.
- A custom MCP tool that gives the model a structured capability.
- A subagent that handles a narrow decision.
- A thinking or turn-budget change grounded in trace evidence.
- A multi-pass architecture, such as judge then verify.

Weak candidates are only prompt rephrasings. If the only diff from the parent
is inside a string literal, redesign around a structural lever.

## SDK-Specific Self-Critique

Before finishing, compare your `build_options` with the parent:

- Did you preserve runtime-owned fields?
- Did you keep benchmark-required tools available?
- Did the change target a trace-observed failure mode?
- Could the next proposer tell which mechanism caused a gain or regression?

Write the candidate to the staging path from the prompt, plus
`proposal_notes.json` as described in `shared.md`.
