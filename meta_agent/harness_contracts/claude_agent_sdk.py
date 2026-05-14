"""Claude Agent SDK harness contract: `build_options(ctx) -> ClaudeAgentOptions`.

The candidate exports `build_options(ctx) -> ClaudeAgentOptions`. The runtime
passes `ctx.cwd` and `ctx.model` because those are plumbing determined at run
time; everything else on `ClaudeAgentOptions` — system prompt, tools, hooks,
skills (MCP tools), subagents, permission mode, thinking, max_turns,
max_budget_usd, allowed_tools, disallowed_tools, mcp_servers — is a
proposer-editable lever.

Benchmark adapters compose on top of the proposer's options: e.g. tau
injects its user-simulator MCP server and prepends tau tools into
`allowed_tools`; the judge adapter injects a `submit_verdict` MCP tool.
This is by design — the harness describes an agent's *behavior*, the
benchmark describes the *exit contract*.

This mirrors `meta_agent.harness_contracts.research`: proposer owns the content,
runtime owns the control flow.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from meta_agent.core.run_context import RunContext

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions


class ClaudeAgentHarnessError(Exception):
    """Raised when a Claude Agent SDK harness module is invalid."""


ClaudeAgentOptionsBuilder = Callable[[RunContext], Any]
"""Shape of the sync callable a candidate harness exports as `build_options`."""


# Fields the runtime owns — the proposer should set them from `ctx`.
# Any other value is either accepted as-is or force-overridden (see note below).
_RUNTIME_OWNED_FIELDS: tuple[str, ...] = ("cwd", "model")


# ---------------------------------------------------------------------------
# Module loading (with mtime-keyed cache)
# ---------------------------------------------------------------------------


# Cache key: (resolved_path_str, mtime_ns). Value: imported builder.
# Per eval run we invoke `build_options(ctx)` hundreds of times against the
# same `harness.py`; without caching, each call does a fresh importlib +
# exec_module, which compounds to seconds of overhead. Keying on mtime means
# an edited harness file still gets reloaded on its next use.
_BUILDER_CACHE: dict[tuple[str, int], ClaudeAgentOptionsBuilder] = {}


def _load_module_from_path(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ClaudeAgentHarnessError(f"cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_options_builder(harness_path: Path) -> ClaudeAgentOptionsBuilder:
    """Import `harness.py` and return its `build_options` callable.

    Results are cached per (resolved path, mtime). A freshly edited harness
    file is reloaded on its next use; repeated calls against an unchanged
    file return the same builder instance.
    """
    if not harness_path.exists():
        raise ClaudeAgentHarnessError(f"{harness_path} does not exist")

    resolved = str(harness_path.resolve())
    mtime_ns = harness_path.stat().st_mtime_ns
    cache_key = (resolved, mtime_ns)
    cached = _BUILDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module = _load_module_from_path(harness_path, "candidate_claude_agent_harness")
    fn = getattr(module, "build_options", None)
    if fn is None or not callable(fn):
        raise ClaudeAgentHarnessError(
            "harness.py must define a callable `build_options(ctx) -> ClaudeAgentOptions`"
        )
    builder = cast(ClaudeAgentOptionsBuilder, fn)
    _BUILDER_CACHE[cache_key] = builder
    return builder


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _coerce_options(value: object, ctx: RunContext) -> "ClaudeAgentOptions":
    """Validate the returned value is a `ClaudeAgentOptions` and enforce invariants."""
    from claude_agent_sdk import ClaudeAgentOptions as _Options

    if not isinstance(value, _Options):
        raise ClaudeAgentHarnessError(
            f"build_options must return ClaudeAgentOptions, got {type(value).__name__}"
        )

    # Runtime-owned fields: the proposer must pass `ctx.cwd` / `ctx.model`.
    # If they diverge we reject — it's almost always a bug (hardcoded path or
    # model name), and silently overriding would hide that bug. Adapters that
    # legitimately need a different cwd/model should set ctx accordingly.
    if value.cwd != ctx.cwd:
        raise ClaudeAgentHarnessError(
            f"ClaudeAgentOptions.cwd must be set from ctx.cwd (got {value.cwd!r}, "
            f"expected {ctx.cwd!r}). Use `cwd=ctx.cwd` in build_options."
        )
    if value.model != ctx.model:
        raise ClaudeAgentHarnessError(
            f"ClaudeAgentOptions.model must be set from ctx.model (got {value.model!r}, "
            f"expected {ctx.model!r}). Use `model=ctx.model` in build_options."
        )

    return value


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_claude_agent_options(
    harness_path: Path,
    ctx: RunContext,
) -> "ClaudeAgentOptions":
    """Import `harness.py`, call `build_options(ctx)`, validate, return options."""
    builder = load_options_builder(harness_path)
    return _coerce_options(builder(ctx), ctx)


_SMOKE_CTX = RunContext(
    cwd="/tmp/meta_agent_smoke",
    model="claude-haiku-4-5",
    task_instruction="smoke",
)


def validate_claude_agent_harness(
    harness_path: Path,
    ctx: Optional[RunContext] = None,
) -> "ClaudeAgentOptions":
    """Shape-check a harness without executing it. Returns the options on success."""
    return build_claude_agent_options(harness_path, ctx or _SMOKE_CTX)


def validate_claude_agent_harness_shape_only(
    harness_path: Path,
) -> Optional["ClaudeAgentOptions"]:
    """Low-risk validation entrypoint for `meta_agent.loop.validate_config`."""
    try:
        return validate_claude_agent_harness(harness_path)
    except ClaudeAgentHarnessError:
        return None


# ---------------------------------------------------------------------------
# Benchmark-side composition helpers
# ---------------------------------------------------------------------------


def extend_allowed_tools(
    options: "ClaudeAgentOptions",
    extra_tools: list[str],
) -> None:
    """Append `extra_tools` to `options.allowed_tools`, deduping.

    `allowed_tools=None` means "all tools allowed" per the SDK; a benchmark that
    injects a required tool still needs that tool to be reachable, so when
    `allowed_tools` is None we leave it alone (everything is already allowed).
    When it's a list (even empty), we union in the extras.
    """
    if options.allowed_tools is None:
        return
    existing = list(options.allowed_tools)
    for t in extra_tools:
        if t not in existing:
            existing.append(t)
    options.allowed_tools = existing


def merge_mcp_server(
    options: "ClaudeAgentOptions",
    name: str,
    server: Any,
) -> None:
    """Merge a benchmark-owned MCP server into `options.mcp_servers`.

    Benchmark-owned servers take precedence over proposer-defined servers at
    the same key — if the proposer tried to shadow a reserved name, we win,
    because the benchmark's exit contract depends on this server.
    """
    existing = dict(options.mcp_servers or {})
    existing[name] = server
    options.mcp_servers = existing


def set_default_max_turns(
    options: "ClaudeAgentOptions",
    default: int,
) -> None:
    """Set `max_turns` only if the proposer left it at the SDK default (None).

    Each benchmark has a sensible ceiling: a judge needs 1-3 turns, tau needs
    50+, artifacts needs ~20. Rather than baking that into the vanilla
    baseline (which is benchmark-agnostic), each benchmark's adapter applies
    its own default on top of the proposer's options.
    """
    if getattr(options, "max_turns", None) is None:
        options.max_turns = default


def append_hooks(
    options: "ClaudeAgentOptions",
    event: str,
    hooks_to_add: list,
) -> None:
    """Append benchmark-required hooks for one event, preserving proposer hooks.

    The proposer's hooks for the same event run first (in declaration order);
    the benchmark's hooks run last. This lets a benchmark enforce a terminal
    invariant (e.g. "stop after submit_verdict fires") without having to
    discard the proposer's contributions.
    """
    existing = dict(options.hooks or {})
    existing_event = list(existing.get(event, []))
    existing_event.extend(hooks_to_add)
    existing[event] = existing_event
    options.hooks = existing


def prepend_system_prompt(
    options: "ClaudeAgentOptions",
    benchmark_prefix: str,
) -> None:
    """Prepend benchmark-required instructions to `options.system_prompt`.

    Handles the three supported shapes:
    - None or empty       -> `benchmark_prefix` becomes the system prompt (raw string)
    - raw string          -> `benchmark_prefix + "\\n\\n" + existing`
    - {type: preset, ...} -> append prefix to the preset's `append` field

    The proposer's content follows the benchmark prefix because the prefix
    describes the exit contract (e.g. "call submit_verdict when done"), which
    has to be visible regardless of what the proposer configured.
    """
    existing = options.system_prompt
    if existing is None or existing == "":
        options.system_prompt = benchmark_prefix
        return
    if isinstance(existing, str):
        options.system_prompt = f"{benchmark_prefix}\n\n{existing}"
        return
    if isinstance(existing, dict):
        merged = dict(existing)
        append = str(merged.get("append") or "")
        merged["append"] = f"{benchmark_prefix}\n\n{append}" if append else benchmark_prefix
        options.system_prompt = merged  # type: ignore[assignment]
        return
    # Unknown shape — fall back to raw string and let the SDK complain if it
    # can't accept it.
    options.system_prompt = benchmark_prefix
