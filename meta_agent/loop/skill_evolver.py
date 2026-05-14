"""Instruction co-evolution: rewrite proposer instructions from past iterations.

After every N epochs (controlled by --evolve-skill / --skill-evolve-every),
this runs a second LLM pass over the proposer's own traces and lets it
edit the active target skill. Each new version is archived under
`experience/skills/`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from meta_agent.utils.logging import get_logger
from meta_agent.core.paths import rel_to_workspace
from meta_agent.loop.proposer import _run_proposer_cli
from meta_agent.loop.state import SHARED_PROPOSER_INSTRUCTIONS_PATH, PROPOSER_INSTRUCTIONS_HISTORY_DIR, import_time
from meta_agent.loop.validate import validate_skill

logger = get_logger("loop")


SKILL_EVOLVER_PROMPT_TEMPLATE = """\
You are improving the skill document ({skill_name}) that guides a harness optimization proposer.

The proposer is a coding agent that reads execution traces from failed tasks, diagnoses why \
they failed, and writes improved harnesses. {skill_name} tells it how to do this — what to \
read, what to change, what to avoid, how to reason.

Your job: analyze how the proposer actually behaved over the last {n_iters} iterations \
({iter_names}), compare that to the outcomes (did pass rate improve?), and make targeted \
edits to {skill_name} that correct bad patterns or reinforce good ones.

## What you have

1. The current {skill_name} at the project root.
2. Proposer reasoning traces at {exp_dir}/<name>/proposer_trace.jsonl — these \
show every file the proposer read, every tool call, its reasoning (ThinkingBlocks).
3. Scores at {exp_dir}/<name>/scores.json — pass_rate, tasks passed/failed.
4. The harness files the proposer wrote at {exp_dir}/<name>/.

Analyze iterations: {iter_names}

## What to look for

Read the proposer traces and scores. Identify:

- REPEATED FAILURES: Does the proposer keep trying a class of change that consistently \
regresses? (e.g. modifying prompt templates, changing hook logic, adding subagents) \
→ Add a warning or constraint to {skill_name} about that pattern.

- MISSED SIGNALS: Does the proposer skip reading traces for certain tasks, or always \
start from the same parent candidate, or never use `cli diff`? \
→ Add a process step reminding it.

- BUNDLED CHANGES: Does the proposer stack multiple unrelated changes despite the skill \
saying "one change at a time"? \
→ Strengthen the constraint with a concrete example of what went wrong.

- SUCCESSFUL PATTERNS: Did certain types of changes consistently improve pass rate? \
→ Add a positive heuristic (e.g. "additive modifications that don't touch existing \
logic are safer than structural rewrites").

- STAGNATION: Is the proposer cycling through similar ideas without progress? \
→ Add guidance to try a fundamentally different lever.

## Rules

- Make TARGETED edits to the existing {skill_name}. Do NOT rewrite it from scratch.
- Add at most 3 new observations or refinements per evolution step.
- Do NOT add task-specific guidance (no "for task X, try Y").
- Do NOT change the shared harness contract or runtime adapter boundaries.
- Do NOT change the directory layout, CLI, or SDK reference sections — those are factual.
- Focus on PROCESS guidance (how to reason, what to inspect, what to avoid) \
not CONTENT guidance (what specific hooks to write).
- If the proposer is improving consistently, make minimal or no changes.
- Preserve all existing sections. Add new guidance inline or append a \
"## Lessons learned" section.

## Output

Write the updated skill to {staging_dir}/{skill_name}
Write a brief (3-5 sentence) summary of what you changed and why to \
{staging_dir}/skill_evolution_notes.md
"""


def _load_skill_history() -> list[dict[str, Any]]:
    path = PROPOSER_INSTRUCTIONS_HISTORY_DIR / "history.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("versions", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _save_skill_history(versions: list[dict[str, Any]]) -> None:
    PROPOSER_INSTRUCTIONS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (PROPOSER_INSTRUCTIONS_HISTORY_DIR / "history.json").write_text(
        json.dumps({"versions": versions}, indent=2)
    )


def _backup_skill(skill_path: Path, version: int) -> Path:
    """Copy the active skill to a versioned archive."""
    PROPOSER_INSTRUCTIONS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROPOSER_INSTRUCTIONS_HISTORY_DIR / f"{skill_path.stem}_v{version:03d}.md"
    if skill_path.exists():
        shutil.copy2(skill_path, dest)
    return dest


def invoke_skill_evolver(
    iterations_analyzed: list[str],
    staging_dir: Path,
    experience_dir: Path,
    model: Optional[str] = None,
    skill_path: Optional[Path] = None,
) -> bool:
    """Run the meta-proposer to evolve the active skill based on proposer behavior."""
    active_skill_path = skill_path or SHARED_PROPOSER_INSTRUCTIONS_PATH
    skill_name = active_skill_path.name
    staging_dir.mkdir(parents=True, exist_ok=True)
    for f in staging_dir.iterdir():
        if f.name in (skill_name, "skill_evolution_notes.md"):
            f.unlink()

    iter_names = ", ".join(iterations_analyzed)
    prompt = SKILL_EVOLVER_PROMPT_TEMPLATE.format(
        n_iters=len(iterations_analyzed),
        iter_names=iter_names,
        exp_dir=rel_to_workspace(experience_dir),
        staging_dir=rel_to_workspace(staging_dir),
        skill_name=skill_name,
    )

    trace_path = PROPOSER_INSTRUCTIONS_HISTORY_DIR / f"evolver_trace_v{len(_load_skill_history()):03d}.jsonl"
    PROPOSER_INSTRUCTIONS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    rc = _run_proposer_cli(
        prompt=prompt,
        system_append=(
            "You are a meta-proposer improving a skill document. "
            f"Read {skill_name} first, then analyze the proposer traces."
        ),
        label="skill-evolver",
        trace_path=trace_path,
        session_dir=PROPOSER_INSTRUCTIONS_HISTORY_DIR / f"evolver_session_v{len(_load_skill_history()):03d}",
        max_turns=30,
        model=model,
    )
    if rc.exit_code != 0:
        logger.info(f"Skill evolver exited with code {rc.exit_code}")
        return False

    staged_skill = staging_dir / skill_name
    if not staged_skill.exists():
        logger.info(f"Skill evolver did not write {staging_dir}/{skill_name}")
        return False

    if not validate_skill(staged_skill, reference_path=active_skill_path):
        logger.info(f"Evolved skill failed validation, keeping current {skill_name}")
        return False

    versions = _load_skill_history()
    if not versions and active_skill_path.exists():
        _backup_skill(active_skill_path, 0)
        versions.append({
            "version": 0,
            "path": f"{active_skill_path.stem}_v000.md",
            "source": "original",
            "skill": skill_name,
        })

    next_version = max((v["version"] for v in versions), default=-1) + 1
    shutil.copy2(staged_skill, active_skill_path)
    _backup_skill(active_skill_path, next_version)

    versions.append({
        "version": next_version,
        "path": f"{active_skill_path.stem}_v{next_version:03d}.md",
        "source": "evolved",
        "skill": skill_name,
        "iterations_analyzed": iterations_analyzed,
        "timestamp": import_time(),
    })
    _save_skill_history(versions)

    notes_path = staging_dir / "skill_evolution_notes.md"
    notes = notes_path.read_text() if notes_path.exists() else "(no notes)"
    logger.info(f"Skill evolved to v{next_version}: {notes[:200]}")
    return True
