"""Single source of truth for harness/runtime targets.

One `AgentTarget` descriptor per harness kind replaces the if/elif chains
and duplicated file-allowlists that previously lived in outer_loop.py,
propose.py, and task_runner.py.

A target answers:
- which runtime runs it by default
- which files on disk constitute its harness
- what the proposer must write (and the prompt instruction text)
- which proposer-instruction file the proposer reads
- (codex only) which path prefixes belong to each surface-lock slot
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class AgentTarget:
    name: str
    default_runtime: str
    is_file_based: bool
    proposer_output_instruction: str           # "{staging}" placeholder
    skill_filename: str
    required_written_files: Tuple[str, ...]    # proposer must write at least one
    module_filename: str = "config.py"         # only meaningful when is_file_based is False
    harness_files: Tuple[str, ...] = ()        # literal top-level filenames
    harness_globs: Tuple[str, ...] = ()        # glob patterns at top of harness dir
    harness_dirs: Tuple[str, ...] = ()         # sub-directories copied wholesale
    surface_lock_slots: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def is_surface_target_path(self, path: str, surface: str) -> bool:
        """Does the relative `path` belong to the given surface-lock slot?"""
        for prefix in self.surface_lock_slots.get(surface, ()):
            if prefix.endswith("/"):
                if path.startswith(prefix):
                    return True
            elif path == prefix:
                return True
        return False

    def is_harness_file_path(self, path: str) -> bool:
        """Does `path` (relative to harness root) belong to this target's layout?"""
        if path in self.harness_files:
            return True
        p = Path(path)
        for pattern in self.harness_globs:
            if p.match(pattern):
                return True
        parts = p.parts
        if parts and parts[0] in self.harness_dirs:
            return True
        return False


_SHARED_FILE_BASED_DIRS = (".codex", ".claude")
_SHARED_FILE_BASED_GLOBS = ("*.sh",)

TARGETS: Dict[str, AgentTarget] = {
    "codex": AgentTarget(
        name="codex",
        default_runtime="codex_sdk",
        is_file_based=True,
        proposer_output_instruction=(
            "write improved harness files to {staging}/. "
            "This includes AGENTS.md, .codex/hooks.json, .codex/hooks/*.sh, "
            ".codex/config.toml, .codex/skills/*.md, and .codex/agents/*.md"
        ),
        skill_filename="codex.md",
        required_written_files=("AGENTS.md",),
        harness_files=("AGENTS.md", "CLAUDE.md"),
        harness_globs=_SHARED_FILE_BASED_GLOBS,
        harness_dirs=_SHARED_FILE_BASED_DIRS,
        surface_lock_slots={
            "agents": ("AGENTS.md",),
            "hooks": (".codex/hooks.json", ".codex/hooks/"),
            "config": (".codex/config.toml",),
            "skills": (".codex/skills/",),
            "subagents": (".codex/agents/",),
        },
    ),
    "claude_agent_sdk": AgentTarget(
        name="claude_agent_sdk",
        default_runtime="claude_sdk",
        is_file_based=False,
        proposer_output_instruction=(
            "write an improved Claude Agent SDK harness to {staging}/harness.py. "
            "The harness must export `build_options(ctx) -> ClaudeAgentOptions`. "
            "Proposer-editable levers include system_prompt, tools, hooks, "
            "skills (MCP tools), subagents (agents=), permission_mode, thinking, "
            "max_turns, max_budget_usd, allowed_tools. Do not write runtime adapter code."
        ),
        skill_filename="claude_agent_sdk.md",
        required_written_files=("harness.py",),
        module_filename="harness.py",
    ),
    "claude_code": AgentTarget(
        name="claude_code",
        default_runtime="claude_code_cli",
        is_file_based=True,
        proposer_output_instruction=(
            "write an improved CLAUDE.md (or AGENTS.md plus CLAUDE.md that imports it), "
            "and optionally .claude/rules/*.md to {staging}/"
        ),
        skill_filename="codex.md",
        required_written_files=("CLAUDE.md", "AGENTS.md"),  # either satisfies
        harness_files=("AGENTS.md", "CLAUDE.md"),
        harness_globs=_SHARED_FILE_BASED_GLOBS,
        harness_dirs=_SHARED_FILE_BASED_DIRS,
    ),
    "research_single_file": AgentTarget(
        name="research_single_file",
        default_runtime="codex_sdk",
        is_file_based=False,
        proposer_output_instruction=(
            "write an improved single-file research harness to {staging}/harness.py. "
            "Do not write runtime adapter code."
        ),
        skill_filename="research_single_file.md",
        required_written_files=("harness.py",),
        module_filename="harness.py",
    ),
    "program_harness": AgentTarget(
        name="program_harness",
        default_runtime="program_harness",
        is_file_based=False,
        proposer_output_instruction=(
            "write a program harness candidate to {staging}/harness.py. "
            "The harness must define `async def run(ctx)` or "
            "`class Harness` with `async def run(self, ctx)`. Keep the candidate "
            "in this one file by default, using clear sections/functions for "
            "prompts, routing, tools, evidence extraction, verification, "
            "state, parsers, retries, telemetry, or subagent-style "
            "orchestration. Do not write benchmark adapters, scorers, labels, "
            "split manifests, eval runners, Modal/runtime files, hidden-holdout "
            "plumbing, or _internal files."
        ),
        skill_filename="program_harness.md",
        required_written_files=("harness.py",),
        module_filename="harness.py",
    ),
    "harbor_agent": AgentTarget(
        name="harbor_agent",
        default_runtime="harbor_agent",
        is_file_based=True,
        proposer_output_instruction=(
            "write an improved Harbor/Terminal-Bench agent to {staging}/agent.py. "
            "The file must define `class HarnessAgent(BaseAgent)` or otherwise "
            "export `HarnessAgent` with Harbor's agent contract. Do not write "
            "benchmark adapters, split manifests, Harbor tasks, verifier code, "
            "or infrastructure files."
        ),
        skill_filename="harbor_agent.md",
        required_written_files=("agent.py",),
        harness_files=("agent.py", "README.md"),
    ),
}


def get_target(name: str) -> AgentTarget:
    target = TARGETS.get(name)
    if target is None:
        raise ValueError(
            f"Unknown agent target: {name!r}. Known: {sorted(TARGETS)}"
        )
    return target


HARNESS_DEFAULT_RUNTIME: Dict[str, str] = {
    name: target.default_runtime for name, target in TARGETS.items()
}

FILE_BASED_HARNESSES = frozenset(
    name for name, target in TARGETS.items() if target.is_file_based
)


def get_module_harness_filename(harness: str) -> str:
    return get_target(harness).module_filename


# ---------------------------------------------------------------------------
# Detection — derive the target from a config directory's layout
# ---------------------------------------------------------------------------


class TargetDetectionError(Exception):
    """Raised when a config path doesn't match any known harness target."""


def detect_target(config_path: Path) -> AgentTarget:
    """Sniff a config directory (or module file) and return its `AgentTarget`.

    Rules, in priority order:

    1. ``harness.py`` present and exports ``build_options``
       → ``claude_agent_sdk`` (the Claude Agent SDK contract).
    2. ``harness.py`` present and exports ``build_harness``
       → ``research_single_file`` (typed research spec).
    3. ``harness.py`` present and exports async ``run`` or ``Harness.run``
       → ``program_harness`` (candidate-owned procedure).
    4. ``agent.py`` present and exports ``HarnessAgent``
       → ``harbor_agent`` (Harbor/Terminal-Bench native agent contract).
    5. ``.codex/`` directory OR ``AGENTS.md`` present
       → ``codex``.
    6. ``CLAUDE.md`` present (and no ``.codex/``)
       → ``claude_code``.

    The harness.py-based branches parse the file's source rather than importing
    it, so detection stays cheap and doesn't run user code.
    """
    p = Path(config_path).expanduser()
    if p.is_file():
        if p.name == "harness.py":
            return _detect_from_harness_py(p)
        p = p.parent

    if not p.is_dir():
        raise TargetDetectionError(f"Config path does not exist: {p}")

    harness_py = p / "harness.py"
    if harness_py.is_file():
        return _detect_from_harness_py(harness_py)

    agent_py = p / "agent.py"
    if agent_py.is_file():
        src = agent_py.read_text()
        if "HarnessAgent" in src:
            return TARGETS["harbor_agent"]

    has_codex_dir = (p / ".codex").is_dir()
    has_agents = (p / "AGENTS.md").is_file()
    has_claude = (p / "CLAUDE.md").is_file()

    if has_codex_dir or has_agents:
        return TARGETS["codex"]
    if has_claude:
        return TARGETS["claude_code"]

    raise TargetDetectionError(
        f"Could not detect a harness target at {p}. Expected one of: "
        "harness.py, agent.py, AGENTS.md, .codex/, or CLAUDE.md."
    )


def _detect_from_harness_py(harness_py: Path) -> AgentTarget:
    """Decide between claude_agent_sdk and research_single_file by grepping.

    Both contracts expose a single top-level callable. We look for which one
    is defined as a top-level `def` or is imported/named at module level.
    """
    src = harness_py.read_text()
    if "def build_options" in src or "build_options =" in src:
        return TARGETS["claude_agent_sdk"]
    if "def build_harness" in src or "build_harness =" in src:
        return TARGETS["research_single_file"]
    if "async def run" in src or "class Harness" in src:
        return TARGETS["program_harness"]
    raise TargetDetectionError(
        f"{harness_py} must export `build_options` (claude_agent_sdk), "
        "`build_harness` (research_single_file), or async `run`/`Harness.run` "
        "(program_harness); none found."
    )
