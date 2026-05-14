"""Run the proposer once against an experience store and output a harness diff.

Public surface: `propose(...)`. Invoked by `meta-agent propose`.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Optional

from meta_agent.core import experience
from meta_agent.utils.logging import get_logger
from meta_agent.loop import invoke_proposer
from meta_agent.core.paths import get_experience_root
from meta_agent.core.targets import AgentTarget, get_target

logger = get_logger("propose")


def propose(
    project: str,
    harness: str = "codex",
    model: Optional[str] = "gpt-5.4",
    proposer_cli: str = "codex",
    apply: bool = False,
) -> bool:
    """Run the proposer once and print the resulting diff.

    Returns True if the proposer produced output.
    """
    experience_dir = get_experience_root() / project / "candidates"
    staging_dir = get_experience_root() / project / "staging"

    if not experience_dir.exists() or not any(experience_dir.iterdir()):
        logger.info(f"No candidates found in {experience_dir}")
        logger.info("Populate the experience store first by running an evaluation that writes candidates.")
        return False

    result = invoke_proposer(
        staging_dir=staging_dir,
        experience_dir=experience_dir,
        bench_name=project,
        model=model,
        harness=harness,
        proposer_cli=proposer_cli,
    )

    if not result.success:
        return False

    target = get_target(harness)
    _print_diff(staging_dir, experience_dir, target)

    if apply:
        _apply_to_latest(staging_dir, experience_dir, target)

    return True


def _print_diff(staging_dir: Path, experience_dir: Path, target: AgentTarget) -> None:
    """Print a unified diff between the best candidate's config and the proposed one."""
    best = experience.best_candidate(experience_dir)
    if not best:
        return

    best_dir = best.dir
    if target.is_file_based:
        changed = False
        rel_files = _collect_harness_files(best_dir, target) | _collect_harness_files(staging_dir, target)
        for rel_path in sorted(rel_files, key=str):
            changed = _diff_file(
                best_dir / rel_path,
                staging_dir / rel_path,
                str(rel_path),
                print_no_changes=False,
            ) or changed
        if not changed:
            logger.info("No harness file changes detected")
    else:
        filename = target.module_filename
        _diff_file(best_dir / filename, staging_dir / filename, filename)


def _collect_harness_files(config_dir: Path, target: AgentTarget) -> set[Path]:
    """Collect relative harness file paths, using the target descriptor as the source of truth."""
    files: set[Path] = set()
    for name in target.harness_files:
        f = config_dir / name
        if f.is_file():
            files.add(Path(name))

    for pattern in target.harness_globs:
        for f in config_dir.glob(pattern):
            if f.is_file():
                files.add(f.relative_to(config_dir))

    for dirname in target.harness_dirs:
        root = config_dir / dirname
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if f.is_file():
                files.add(f.relative_to(config_dir))
    return files


def _diff_file(old_path: Path, new_path: Path, label: str, print_no_changes: bool = True) -> bool:
    old_lines = old_path.read_text().splitlines(keepends=True) if old_path.exists() else []
    new_lines = new_path.read_text().splitlines(keepends=True) if new_path.exists() else []

    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{label}", tofile=f"b/{label}"))
    if diff:
        print(f"\n{'='*60}")
        print(f"  Proposed changes to {label}")
        print(f"{'='*60}\n")
        sys.stdout.writelines(diff)
        print()
        return True
    if print_no_changes:
        logger.info(f"No changes to {label}")
    return False


def _apply_to_latest(staging_dir: Path, experience_dir: Path, target: AgentTarget) -> None:
    """Copy the proposed config into a new candidate directory."""
    import shutil

    existing = [d.name for d in experience_dir.iterdir() if d.is_dir()]
    next_idx = len(existing)
    new_name = f"proposed_{next_idx:03d}"
    new_dir = experience_dir / new_name
    new_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file_based:
        for item in staging_dir.iterdir():
            dest = new_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    else:
        filename = target.module_filename
        config_src = staging_dir / filename
        if config_src.exists():
            shutil.copy2(config_src, new_dir / filename)

    logger.info(f"Applied to {new_dir}")
    logger.info("Re-run your agent or benchmark pipeline to add the next candidate, then propose again.")
