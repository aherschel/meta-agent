"""LoopState + setup helpers used across every loop phase.

Owns:
- shared module constants (PROPOSER_INSTRUCTIONS_DIR, SHARED_PROPOSER_INSTRUCTIONS_PATH, PROPOSER_INSTRUCTIONS_HISTORY_DIR)
- small pure helpers (_spark, import_time, _select_parent_candidate_name,
  _parse_tasks_csv, _compute_score_delta)
- reproducibility manifest builders
- run-level lockfile (acquire/release with stale-PID takeover)
- the LoopState dataclass and its one-shot constructor `_prepare_loop_state`
- header printing
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from meta_agent.core.benchmark import Benchmark, primary_reward, reward_or_none
from meta_agent.core.benchmark import parse_benchmark_ref
from meta_agent.utils.logging import get_logger
from meta_agent.core.paths import PACKAGE_ROOT, get_experience_root, get_workspace_root, rel_to_workspace
from meta_agent.core.targets import AgentTarget, TargetDetectionError, detect_target

logger = get_logger("loop")

PROPOSER_INSTRUCTIONS_DIR = PACKAGE_ROOT / "meta_agent" / "proposer_instructions"
SHARED_PROPOSER_INSTRUCTIONS_PATH = PROPOSER_INSTRUCTIONS_DIR / "shared.md"
CODEX_PROPOSER_INSTRUCTIONS_PATH = PROPOSER_INSTRUCTIONS_DIR / "codex.md"
PROPOSER_INSTRUCTIONS_HISTORY_DIR = get_experience_root() / "skills"

SPARK_CHARS = " ▁▂▃▄▅▆▇█"
_EPOCH_NAME_RE = re.compile(r"^evo_(\d{3})(?:_|$)")


# --- Tiny utilities -------------------------------------------------------

def import_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spark(values: Any) -> str:
    vals = list(values)
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / span * 8), 8)] for v in vals)


def _parse_tasks_csv(tasks_csv: Optional[str]) -> list[str]:
    if not tasks_csv:
        return []
    return [t.strip() for t in tasks_csv.split(",") if t.strip()]


def _benchmark_candidates_per_iter_default(
    benchmark_ref: str,
    workspace_root: Optional[Path] = None,
) -> Optional[int]:
    """Read ``optimizer.candidates_per_iter`` from a benchmark family YAML.

    The CLI documents this benchmark-level default, and Plan-RewardBench now
    relies on it for a wider candidate slate. This helper intentionally reads
    the raw family YAML instead of the resolved :class:`Benchmark`, because
    optimizer settings are loop policy, not adapter backend.
    """
    yaml_path, _ = parse_benchmark_ref(benchmark_ref)
    path = Path(yaml_path)
    if not path.is_absolute() and workspace_root is not None:
        path = workspace_root / path
    if not path.is_file():
        return None

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return None
    optimizer = data.get("optimizer") or {}
    if not isinstance(optimizer, dict):
        return None
    raw_value = optimizer.get("candidates_per_iter")
    if not isinstance(raw_value, int) or raw_value < 1:
        return None
    return raw_value


def _epoch_idx_from_candidate_name(name: Any) -> Optional[int]:
    if not isinstance(name, str):
        return None
    match = _EPOCH_NAME_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_resume_window(
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> tuple[int, int, Optional[int]]:
    """Compute the actual epoch window for this invocation.

    On `--resume`, never replay epochs that are already persisted in history.
    Preserve the caller's original end epoch so container retries continue from
    the next unfinished epoch instead of restarting from the original
    `--start-from`.
    """
    requested_start = int(getattr(args, "start_from", 1))
    requested_iterations = max(int(getattr(args, "iterations", 0)), 0)
    requested_final_epoch = requested_start + requested_iterations - 1
    requested_resume_from_proposal = bool(
        getattr(args, "resume_from_proposal", False)
    )

    if not getattr(args, "resume", False):
        return (
            requested_start,
            requested_iterations,
            requested_start if requested_resume_from_proposal else None,
        )

    completed_epochs = [
        epoch_idx
        for row in history
        if (epoch_idx := _epoch_idx_from_candidate_name(row.get("name"))) is not None
    ]
    next_unfinished_epoch = max(completed_epochs, default=0) + 1
    effective_start = max(requested_start, next_unfinished_epoch)
    effective_iterations = max(0, requested_final_epoch - effective_start + 1)
    resume_from_proposal_epoch = (
        requested_start
        if requested_resume_from_proposal and effective_start == requested_start
        else None
    )
    return effective_start, effective_iterations, resume_from_proposal_epoch


def accept_reward_from_row(
    row: dict[str, Any], accept_on_holdout: bool,
) -> Optional[float]:
    """Reward used for acceptance-gate + parent-selection decisions.

    When --accept-on-holdout is on, we care about holdout reward (that's the
    signal that prevents search-set overfitting). If a row has no holdout
    reward (e.g. legacy baseline row where holdout was never run), return
    None so callers can skip it rather than silently falling back to train.
    This mirrors the gate's semantics: a candidate with no holdout score
    shouldn't become a new champion or seed on the basis of train alone.

    When --accept-on-holdout is off, fall back to the canonical train reward.
    """
    if accept_on_holdout:
        hr = row.get("holdout_reward")
        return float(hr) if isinstance(hr, (int, float)) else None
    return reward_or_none(row)


def _select_parent_candidate_name(
    history: list[dict[str, Any]],
    accept_on_holdout: bool = False,
) -> Optional[str]:
    """Highest-accept-reward row in `history`, or None if empty/unrewarded.

    When `accept_on_holdout` is True, parent selection uses holdout reward
    so rejected-on-holdout candidates never seed the next proposal (which
    would silently re-introduce train-overfit branches into the evolution
    chain).
    """
    best_name: Optional[str] = None
    best_reward = float("-inf")
    for row in history:
        name = row.get("name")
        if not isinstance(name, str):
            continue
        value = accept_reward_from_row(row, accept_on_holdout)
        if value is None:
            continue
        if value > best_reward:
            best_reward = value
            best_name = name
    return best_name


def _compute_score_delta(
    current: Optional[dict[str, Any]],
    parent: Optional[dict[str, Any]],
) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {
        "reward_delta": None,
        "pass_rate_delta": None,
        "n_passed_delta": None,
    }

    cur_reward = reward_or_none(current)
    par_reward = reward_or_none(parent)
    if cur_reward is not None and par_reward is not None:
        out["reward_delta"] = cur_reward - par_reward

    if isinstance(current, dict) and isinstance(parent, dict):
        cur_rate = current.get("pass_rate")
        par_rate = parent.get("pass_rate")
        if isinstance(cur_rate, (int, float)) and isinstance(par_rate, (int, float)):
            out["pass_rate_delta"] = float(cur_rate) - float(par_rate)

        cur_passed = current.get("n_passed")
        par_passed = parent.get("n_passed")
        if isinstance(cur_passed, (int, float)) and isinstance(par_passed, (int, float)):
            out["n_passed_delta"] = float(cur_passed) - float(par_passed)

    return out


def _build_frontier(
    history: list[dict[str, Any]],
    *,
    run_name: str,
    accept_on_holdout: bool,
    include_holdout: bool = False,
) -> dict[str, Any]:
    """Build a small Pareto/frontier summary from loop history.

    Primary metric is the same reward used by the acceptance gate. Secondary
    metric is eval cost, with lower cost preferred. Rows missing the primary
    metric are omitted from the frontier but retained in the full candidate
    list so diagnostics can still see failures.
    """

    def primary(row: dict[str, Any]) -> Optional[float]:
        return accept_reward_from_row(row, accept_on_holdout)

    def cost(row: dict[str, Any]) -> Optional[float]:
        value = (
            row.get("search_cost_usd")
            if isinstance(row.get("search_cost_usd"), (int, float))
            else row.get("cost_usd")
        )
        return float(value) if isinstance(value, (int, float)) else None

    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(history):
        item = dict(row)
        item.setdefault("rank", idx + 1)
        item["accept_reward"] = primary(row)
        if not include_holdout:
            item = _strip_holdout_fields(item)
        candidates.append(item)

    scored = [row for row in candidates if isinstance(row.get("accept_reward"), (int, float))]
    current_best = _select_parent_candidate_name(history, accept_on_holdout)

    pareto: list[dict[str, Any]] = []
    for row in scored:
        row_reward = float(row["accept_reward"])
        row_cost = cost(row)
        dominated = False
        for other in scored:
            if other is row:
                continue
            other_reward = float(other["accept_reward"])
            other_cost = cost(other)
            reward_at_least = other_reward >= row_reward
            cost_at_most = (
                row_cost is None
                or other_cost is not None
                and other_cost <= row_cost
            )
            strictly_better = (
                other_reward > row_reward
                or (
                    row_cost is not None
                    and other_cost is not None
                    and other_cost < row_cost
                )
            )
            if reward_at_least and cost_at_most and strictly_better:
                dominated = True
                break
        if not dominated:
            item = dict(row)
            item["is_pareto"] = True
            pareto.append(item)

    pareto.sort(
        key=lambda row: (
            -float(row.get("accept_reward") or float("-inf")),
            cost(row) if cost(row) is not None else float("inf"),
            str(row.get("name") or ""),
        )
    )

    return {
        "run_name": run_name,
        "accept_on_holdout": accept_on_holdout,
        "includes_holdout": include_holdout,
        "primary_metric": "holdout_reward" if accept_on_holdout else "reward",
        "secondary_metric": "search_cost_usd",
        "current_best": current_best,
        "pareto": pareto,
        "candidates": candidates,
    }


def _build_candidate_index(
    history: list[dict[str, Any]],
    *,
    run_name: str,
    accept_on_holdout: bool,
    include_holdout: bool = False,
) -> dict[str, Any]:
    """Build a proposer-readable candidate ranking.

    This is intentionally flatter than ``frontier.json`` because CLI/table
    consumers want one row per candidate with normalized search/holdout field
    names. When ``include_holdout`` is true, only aggregate holdout fields are
    copied; detailed holdout traces and gate-private fields remain elsewhere.
    """

    current_best = _select_parent_candidate_name(history, accept_on_holdout)
    rows: list[dict[str, Any]] = []
    for row in history:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            continue

        item: dict[str, Any] = {
            "name": name,
            "candidate_path": f"experience/{run_name}/candidates/{name}",
            "accept_reward": accept_reward_from_row(row, accept_on_holdout),
            "search_reward": reward_or_none(row),
            "search_pass_rate": row.get("pass_rate"),
            "search_n_passed": row.get("n_passed"),
            "search_n_tasks": row.get("n_tasks"),
            "search_cost_usd": row.get("cost_usd"),
            "search_total_cost_with_proposer_usd": row.get(
                "total_cost_with_proposer_usd"
            ),
            "is_current_best": name == current_best,
        }
        if include_holdout:
            for key in _HOLDOUT_HISTORY_FIELDS:
                if key in row:
                    item[key] = row[key]
        rows.append(item)

    rows.sort(
        key=lambda row: (
            -float(row["accept_reward"])
            if isinstance(row.get("accept_reward"), (int, float))
            else float("inf"),
            float(row["search_cost_usd"])
            if isinstance(row.get("search_cost_usd"), (int, float))
            else float("inf"),
            str(row.get("name") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "run_name": run_name,
        "accept_on_holdout": accept_on_holdout,
        "includes_holdout": include_holdout,
        "primary_metric": "holdout_reward" if accept_on_holdout else "reward",
        "current_best": current_best,
        "candidates": rows,
    }


def _score_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _load_search_bucket_scores(candidate_dir: Path) -> dict[str, float]:
    """Load proposer-visible search bucket scores for one candidate."""
    category_path = candidate_dir / "category_scores.json"
    if not category_path.is_file():
        return {}
    try:
        category_scores = json.loads(category_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(category_scores, dict):
        return {}

    summary = category_scores.get("_summary")
    buckets = summary.get("buckets") if isinstance(summary, dict) else None
    if isinstance(buckets, dict):
        return {
            str(bucket): float(rate)
            for bucket, rate in buckets.items()
            if isinstance(rate, (int, float))
        }

    loaded: dict[str, float] = {}
    for bucket, payload in category_scores.items():
        if str(bucket).startswith("_") or not isinstance(payload, dict):
            continue
        rate = payload.get("pass_rate")
        if isinstance(rate, (int, float)):
            loaded[str(bucket)] = float(rate)
    return loaded


def _summarize_bucket_deltas(deltas: dict[str, float], limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    gains = sorted(
        ((bucket, delta) for bucket, delta in deltas.items() if delta > 0),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    regressions = sorted(
        ((bucket, delta) for bucket, delta in deltas.items() if delta < 0),
        key=lambda item: (item[1], item[0]),
    )[:limit]
    return {
        "gains": [{"bucket": bucket, "delta": delta} for bucket, delta in gains],
        "regressions": [
            {"bucket": bucket, "delta": delta}
            for bucket, delta in regressions
        ],
    }


def _attach_candidate_diagnostics(
    payload: dict[str, Any],
    *,
    experience_dir: Path,
) -> None:
    """Add search-only bucket deltas and aggregate search/holdout gaps.

    The proposer may inspect search traces and public aggregate holdout scores.
    We therefore attach only search bucket breakdowns, plus aggregate
    search-vs-holdout disagreement that is already derivable from the table.
    """
    candidate_rows: list[dict[str, Any]] = []
    for key in ("candidates", "pareto"):
        rows = payload.get(key)
        if isinstance(rows, list):
            candidate_rows.extend(row for row in rows if isinstance(row, dict))
    if not candidate_rows:
        return

    baseline_buckets = _load_search_bucket_scores(experience_dir / "baseline")
    for row in candidate_rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            continue

        search = _score_float(row.get("search_reward"))
        holdout = _score_float(row.get("holdout_reward"))
        if search is not None and holdout is not None:
            row["search_holdout_gap"] = search - holdout

        buckets = _load_search_bucket_scores(experience_dir / name)
        if buckets:
            row["search_bucket_scores"] = buckets
        if buckets and baseline_buckets:
            deltas = {
                bucket: rate - baseline_buckets[bucket]
                for bucket, rate in buckets.items()
                if bucket in baseline_buckets
            }
            row["search_bucket_deltas_vs_baseline"] = deltas
            row["search_bucket_delta_summary"] = _summarize_bucket_deltas(deltas)


# --- Run lockfile --------------------------------------------------------


def _lockfile_path(run_name: str) -> Path:
    return get_experience_root() / run_name / ".lock"


def _pid_alive(pid: int) -> bool:
    """Cheap liveness check: signal 0 raises iff the pid doesn't exist or isn't ours."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


class RunLockError(Exception):
    """Raised when a run can't acquire its lockfile because another run holds it."""


def acquire_run_lock(run_name: str, *, allow_resume: bool) -> Path:
    """Take the per-run lock on `experience/<run_name>/.lock`.

    If the file exists, read the PID:
    * PID is a live process  → refuse (another run in flight). ``allow_resume``
      overrides this, useful when the prior process is genuinely still running
      and the user explicitly wants two writers (rare; not recommended).
    * PID is dead             → treat as crash debris, overwrite the lock.

    On acquire, registers an ``atexit`` handler that removes the lock. Callers
    can also call :func:`release_run_lock` for deterministic cleanup.
    """
    lock = _lockfile_path(run_name)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            payload = json.loads(lock.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
        held_pid = payload.get("pid")
        held_task = payload.get("modal_task_id")
        alive = isinstance(held_pid, int) and _pid_alive(held_pid)
        if alive and not allow_resume:
            raise RunLockError(
                f"Run {run_name!r} is already in use "
                f"(pid={held_pid}, modal_task_id={held_task!r}, "
                f"lock={lock}). Pass --resume to override, or choose a "
                "different --run-name."
            )
        if alive and allow_resume:
            logger.warning(
                f"Acquiring {run_name!r} lock while pid={held_pid} is still "
                "alive (--resume). Two concurrent writers may stomp state."
            )
    lock.write_text(json.dumps({
        "pid": os.getpid(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID", ""),
        "started_at": import_time(),
        "argv": sys.argv,
    }, indent=2))
    atexit.register(_release_run_lock_silent, run_name)
    return lock


def release_run_lock(run_name: str) -> None:
    """Remove the lockfile if we own it (PID matches)."""
    lock = _lockfile_path(run_name)
    if not lock.exists():
        return
    try:
        payload = json.loads(lock.read_text())
    except (json.JSONDecodeError, OSError):
        payload = {}
    if payload.get("pid") == os.getpid():
        try:
            lock.unlink()
        except OSError:
            pass


def _release_run_lock_silent(run_name: str) -> None:
    try:
        release_run_lock(run_name)
    except Exception:
        pass


# --- Run dir collision / --fresh wipe ------------------------------------


def _run_dirs_for_wipe(run_name: str) -> list[Path]:
    """Return the list of top-level experience dirs that belong to a run.

    Includes the search dir ``experience/<run_name>/`` plus every
    namespaced holdout dir ``experience/<run_name>__<holdout_name>/`` we
    create. Used by ``--fresh`` to know exactly what to delete.
    """
    root = get_experience_root()
    dirs: list[Path] = []
    search_dir = root / run_name
    if search_dir.exists():
        dirs.append(search_dir)
    prefix = f"{run_name}__"
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith(prefix):
                dirs.append(child)
    return dirs


def _run_has_history(experience_root_for_run: Path) -> bool:
    """True iff ``experience/<run_name>/history.json`` lists at least one iteration."""
    history_path = experience_root_for_run / "history.json"
    if not history_path.exists():
        return False
    try:
        payload = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    iterations = payload.get("iterations") if isinstance(payload, dict) else None
    return isinstance(iterations, list) and len(iterations) > 0


class RunCollisionError(Exception):
    """Raised when a run-name's dir already carries history and neither
    ``--resume`` nor ``--fresh`` was passed.

    The message names the offending directory and lists the two flags so the
    user can pick what they actually meant.
    """


def check_run_collision(run_name: str, *, resume: bool, fresh: bool) -> None:
    """Refuse if the run's dir already has history and no explicit flag was set.

    * Fresh dir: no-op.
    * Dir exists but empty / no history rows: no-op (treated as fresh).
    * Dir exists + history present + ``--resume``: no-op (legitimate resume).
    * Dir exists + history present + ``--fresh``: no-op here (caller wipes).
    * Dir exists + history present + neither flag: raise
      :class:`RunCollisionError`.
    """
    if resume or fresh:
        return
    experience_root_for_run = get_experience_root() / run_name
    if not experience_root_for_run.exists():
        return
    if not _run_has_history(experience_root_for_run):
        return
    raise RunCollisionError(
        f"Run {run_name!r} already has history at "
        f"{experience_root_for_run / 'history.json'}. "
        f"Pass --resume to continue it, or --fresh to wipe and start over."
    )


def wipe_run_dirs(run_name: str, *, countdown_s: float = 3.0) -> None:
    """Delete every dir that belongs to `run_name`, with a grace countdown.

    Prints the list of dirs being deleted and sleeps ``countdown_s`` seconds
    so a mistyped ``--fresh`` is recoverable with Ctrl-C. Passes through
    any KeyboardInterrupt raised during the countdown to the caller.

    Holds on to the caller's lockfile semantics: if a ``.lock`` exists under
    ``experience/<run_name>/`` we delete it along with the rest — the whole
    dir is being wiped, so a lingering lockfile would be meaningless.
    """
    import shutil
    import time

    dirs = _run_dirs_for_wipe(run_name)
    if not dirs:
        logger.info(f"--fresh: nothing to wipe for run {run_name!r}")
        return

    logger.warning(
        f"--fresh: about to delete {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}:"
    )
    for d in dirs:
        logger.warning(f"  {d}")
    if countdown_s > 0:
        for remaining in range(int(countdown_s), 0, -1):
            logger.warning(f"  deleting in {remaining}s... (Ctrl-C to abort)")
            time.sleep(1)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    logger.info(f"--fresh: wiped {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}.")


def _fresh_launch_guard_path(launch_id: str) -> Path:
    safe_launch_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", launch_id)[:160]
    return get_experience_root() / ".fresh_launches" / f"{safe_launch_id}.json"


def _convert_fresh_retry_to_resume(args: argparse.Namespace, run_name: str) -> bool:
    """Turn a Modal replay of the same --fresh launch into --resume.

    Modal retries call the function with exactly the same argv. If that argv has
    ``--fresh``, a retry after baseline/proposal work would otherwise wipe the
    very run state it is supposed to recover. The Modal runner passes one stable
    launch id across retries, so a second sighting means "same launch replay",
    not a new intentional fresh run.
    """
    launch_id = os.environ.get("META_AGENT_MODAL_LAUNCH_ID")
    if not launch_id:
        return False
    guard_path = _fresh_launch_guard_path(launch_id)
    if not guard_path.exists():
        return False

    try:
        payload = json.loads(guard_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}
    guarded_run_name = payload.get("run_name")
    if guarded_run_name and guarded_run_name != run_name:
        logger.warning(
            "--fresh retry guard %s belongs to run %r, not %r; leaving --fresh active.",
            launch_id,
            guarded_run_name,
            run_name,
        )
        return False

    setattr(args, "fresh", False)
    setattr(args, "resume", True)
    setattr(args, "_auto_resume_from_proposal", True)
    logger.warning(
        "--fresh: detected Modal retry for launch_id=%s; treating replay as "
        "--resume so existing run state is not wiped again.",
        launch_id,
    )
    return True


def _mark_fresh_launch_wiped(args: argparse.Namespace, run_name: str) -> None:
    launch_id = os.environ.get("META_AGENT_MODAL_LAUNCH_ID")
    if not launch_id:
        return
    guard_path = _fresh_launch_guard_path(launch_id)
    payload = {
        "launch_id": launch_id,
        "run_name": run_name,
        "benchmark": getattr(args, "benchmark", None),
        "created_at": import_time(),
        "argv": sys.argv[1:],
    }
    try:
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        guard_path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning(f"--fresh: could not persist retry guard {guard_path}: {exc}")


def _proposal_checkpoint_exists(history_path: Path, epoch_idx: int) -> bool:
    checkpoint = (
        history_path.parent
        / "_internal"
        / "proposal_checkpoints"
        / f"evo_{epoch_idx:03d}.json"
    )
    return checkpoint.exists()


# --- Reproducibility manifest --------------------------------------------

def _run_text_cmd(cmd: list[str], cwd: Optional[Path] = None, timeout_sec: int = 8) -> Optional[str]:
    """Run `cmd`, return its first non-empty output line or None on failure."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or "").strip() or (result.stderr or "").strip()
    return text.splitlines()[0].strip() if text else None


def _resolve_from_workspace(workspace_root: Path, raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = workspace_root / p
    return str(p.resolve())


def _build_reproducibility_manifest(
    *,
    args: argparse.Namespace,
    bench_name: str,
    workspace_root: Path,
) -> dict[str, Any]:
    return {
        "benchmark_name": bench_name,
        "benchmark_path": _resolve_from_workspace(workspace_root, args.benchmark),
        "holdout_benchmark_path": _resolve_from_workspace(
            workspace_root, getattr(args, "holdout_benchmark", None),
        ),
        "seed": getattr(args, "seed", None),
        "batch_size": getattr(args, "batch_size", None),
        "git_sha": _run_text_cmd(["git", "rev-parse", "HEAD"], cwd=workspace_root),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "codex_version": _run_text_cmd(["codex", "--version"], cwd=workspace_root),
        "claude_version": _run_text_cmd(["claude", "--version"], cwd=workspace_root),
        "cli_args": vars(args),
        "workspace_root": str(workspace_root),
        "captured_at": import_time(),
    }


# --- LoopState + setup ----------------------------------------------------

@dataclass
class LoopState:
    """Long-lived values shared across every phase of the outer loop."""

    args: argparse.Namespace
    bench: Benchmark
    bench_target: AgentTarget
    workspace_root: Path
    run_name: str                        # experience-dir name (overrides bench.name via --run-name)
    experience_dir: Path
    staging_dir: Path
    holdout_bench: Optional[Benchmark]   # resolved holdout benchmark (with split applied)
    holdout_dir: Optional[Path]
    history_path: Path
    all_task_names: list[str]
    batch_size: Optional[int]
    _batch_rng: random.Random
    effective_start_from: int = 1
    effective_iterations: int = 0
    resume_from_proposal_epoch: Optional[int] = None
    _batch_queue: list[str] = field(default_factory=list)
    run_repro_manifest: dict[str, Any] = field(default_factory=dict)
    experiment_config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    best_rate: float = 0.0
    iterations_since_skill_evolve: list[str] = field(default_factory=list)

    def next_batch(self) -> Optional[str]:
        """Comma-separated task names for the next epoch's batch, or None for all."""
        if self.batch_size is None or not self.all_task_names:
            return None
        return ",".join(self._pop_batch())

    def _pop_batch(self) -> list[str]:
        """DataLoader-style: shuffle, slice, reshuffle on exhaustion."""
        assert self.batch_size is not None
        if len(self._batch_queue) < self.batch_size:
            fresh = list(self.all_task_names)
            self._batch_rng.shuffle(fresh)
            self._batch_queue.extend(fresh)
        batch = self._batch_queue[:self.batch_size]
        self._batch_queue = self._batch_queue[self.batch_size:]
        return batch

    def write_history(self) -> None:
        """Write two history files: public (proposer-visible, no holdout) + internal (full).

        The public `history.json` lives directly under the search experience dir
        and is readable by the proposer — it MUST NOT contain any holdout
        signal. The full history with holdout reward / pass_rate / n_passed
        lives under `_internal/history.json`; proposer instructions forbid
        reading `_internal/` paths. On resume, the orchestrator reads from
        `_internal/history.json` to recover the full picture.
        """
        payload_full = {
            "benchmark": self.run_name,
            "model": self.args.model,
            "config": self.experiment_config,
            "iterations": self.history,
        }
        payload_public = {
            "benchmark": self.run_name,
            "model": self.args.model,
            "config": self.experiment_config,
            "iterations": [_strip_holdout_fields(r) for r in self.history],
        }
        internal_dir = self.history_path.parent / "_internal"
        internal_dir.mkdir(parents=True, exist_ok=True)
        (internal_dir / "history.json").write_text(json.dumps(payload_full, indent=2))
        self.history_path.write_text(json.dumps(payload_public, indent=2))
        accept_on_holdout = bool(getattr(self.args, "accept_on_holdout", False))
        candidate_index = _build_candidate_index(
            self.history,
            run_name=self.run_name,
            accept_on_holdout=accept_on_holdout,
            include_holdout=True,
        )
        _attach_candidate_diagnostics(
            candidate_index,
            experience_dir=self.experience_dir,
        )
        frontier = _build_frontier(
            self.history,
            run_name=self.run_name,
            accept_on_holdout=accept_on_holdout,
            include_holdout=True,
        )
        _attach_candidate_diagnostics(
            frontier,
            experience_dir=self.experience_dir,
        )
        (self.history_path.parent / "candidate_index.json").write_text(
            json.dumps(candidate_index, indent=2)
        )
        (self.history_path.parent / "frontier.json").write_text(
            json.dumps(frontier, indent=2)
        )


_HOLDOUT_HISTORY_FIELDS: tuple[str, ...] = (
    "holdout_reward",
    "holdout_pass_rate",
    "holdout_n_passed",
    "holdout_n_tasks",
    "holdout_cost",
)


def _strip_holdout_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of `row` with every holdout_* field removed."""
    return {k: v for k, v in row.items() if k not in _HOLDOUT_HISTORY_FIELDS}


def _build_task_pool(bench: Benchmark, fast: bool) -> list[str]:
    """Pick the full task pool — fast_tasks if requested, else the adapter's pool."""
    from meta_agent.core import adapters

    if fast and bench.fast_tasks:
        return list(bench.fast_tasks)
    if bench.tasks:
        return [t.name for t in bench.tasks]
    adapter = adapters.get(bench.type)
    return adapter.task_pool(bench) if adapter.task_pool else []


def _build_experiment_config(
    args: argparse.Namespace,
    bench: Benchmark,
    target: AgentTarget,
    all_task_names: list[str],
    batch_size: Optional[int],
) -> dict[str, Any]:
    return {
        "description": bench.description or None,
        "harness": target.name,
        "runtime": target.default_runtime,
        "bench_type": bench.type,
        "bench_family": bench.family,
        "bench_split": bench.split,
        "n_search_tasks": len(all_task_names),
        "n_total_tasks": len(all_task_names),
        "batch_size": batch_size,
        "seed": args.seed,
        "holdout_benchmark": args.holdout_benchmark or None,
        "holdout_split": getattr(args, "holdout_split", None),
        "proposer_model": args.proposer_model,
        "proposer_cli": getattr(args, "proposer_cli", "claude"),
        "max_iterations": args.iterations,
        "concurrency": args.concurrency,
        "fast": args.fast,
        "surface_lock": args.surface_lock,
    }


def _reconstruct_skill_counter(history: list[dict[str, Any]]) -> list[str]:
    """Rebuild `iterations_since_skill_evolve` from disk on resume.

    Scans ``experience/skills/history.json`` for every version that's been
    emitted and unions their ``iterations_analyzed`` lists — those candidates
    were already consumed by the skill evolver. Anything in the loop history
    that's NOT been consumed goes into the counter so the next threshold
    check is accurate after a crash+resume.

    Returns ``[]`` when no skill versions exist or no history is available.
    """
    versions_path = PROPOSER_INSTRUCTIONS_HISTORY_DIR / "history.json"
    consumed: set[str] = set()
    if versions_path.exists():
        try:
            payload = json.loads(versions_path.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
        for v in payload.get("versions", []) or []:
            for name in v.get("iterations_analyzed", []) or []:
                if isinstance(name, str):
                    consumed.add(name)
    # Preserve history order; only include candidate-like rows (names that
    # start with evo_; the "baseline" row never contributes to the counter).
    return [
        row["name"]
        for row in history
        if isinstance(row.get("name"), str)
        and row["name"].startswith("evo_")
        and row["name"] not in consumed
    ]


def _load_history_from_disk(history_path: Path) -> list[dict[str, Any]]:
    """Load history on resume. Prefer `_internal/history.json` (full, with holdout);
    fall back to the public `history.json` for backward-compat with pre-split runs.
    """
    internal_path = history_path.parent / "_internal" / "history.json"
    for path in (internal_path, history_path):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text()).get("iterations", [])
        except (json.JSONDecodeError, KeyError):
            continue
    return []


def _resolve_bench_target(args: argparse.Namespace, run_name: str) -> AgentTarget:
    """Pick the agent target by sniffing `--baseline` or any candidate already
    on disk under the experience store.

    The harness kind is an agent-side fact, not a benchmark-side one. We
    derive it from whatever concrete config is available: the `--baseline`
    flag is the usual source; for resumed runs with no `--baseline`, we
    pick up the existing `baseline/` (or any other candidate) under
    `experience/<run_name>/candidates/`.
    """
    if args.baseline:
        try:
            return detect_target(Path(args.baseline))
        except TargetDetectionError as exc:
            logger.info(f"ERROR: could not detect harness target at --baseline: {exc}")
            sys.exit(2)

    candidates_root = get_experience_root() / run_name / "candidates"
    if candidates_root.is_dir():
        for candidate in sorted(candidates_root.iterdir()):
            if not candidate.is_dir():
                continue
            try:
                return detect_target(candidate)
            except TargetDetectionError:
                continue

    logger.info(
        "ERROR: no --baseline set and no existing candidates found; cannot "
        "determine harness target. Pass --baseline <config_dir>."
    )
    sys.exit(2)


def _prepare_loop_state(args: argparse.Namespace) -> LoopState:
    """One-shot setup: load bench, make dirs, build task pool, init RNG + history.

    Collision policy (runs with an existing dir + history on disk):
    * ``--fresh``: wipe every dir belonging to this run-name before anything
      else happens (3s countdown). Afterwards we proceed as a fresh run.
    * ``--resume``: proceed; history is loaded and the loop continues from
      where it left off.
    * Neither flag: :class:`RunCollisionError` → exit 2.
    """
    from meta_agent.core import adapters, experience
    from meta_agent.core.benchmark import load_benchmark

    workspace_root = get_workspace_root()
    bench = load_benchmark(args.benchmark, split=getattr(args, "split", None))
    run_name = getattr(args, "run_name", None) or bench.name

    # --fresh: wipe the dir first (loud warning + countdown). Must run BEFORE
    # collision check so --fresh actually clears the collision.
    if getattr(args, "fresh", False):
        converted_to_resume = _convert_fresh_retry_to_resume(args, run_name)
        if not converted_to_resume:
            try:
                wipe_run_dirs(run_name)
            except KeyboardInterrupt:
                logger.info("--fresh: aborted by user. Nothing was deleted.")
                sys.exit(130)
            _mark_fresh_launch_wiped(args, run_name)

    # Collision refusal: if there's existing history and neither --resume
    # nor --fresh was passed, the user probably didn't intend to stomp.
    try:
        check_run_collision(
            run_name,
            resume=getattr(args, "resume", False),
            fresh=getattr(args, "fresh", False),
        )
    except RunCollisionError as exc:
        logger.info(f"ERROR: {exc}")
        sys.exit(2)

    target = _resolve_bench_target(args, run_name)

    if args.surface_lock and target.name != "codex":
        logger.info("ERROR: --surface-lock currently supports only codex harnesses")
        sys.exit(2)

    # Enforce adapter/harness compatibility up-front so we fail fast instead
    # of after the first epoch burns proposer/eval cost.
    try:
        adapter = adapters.get(bench.type)
    except ValueError as exc:
        logger.info(f"ERROR: {exc}")
        sys.exit(2)
    if args.baseline:
        try:
            adapters.assert_target_supported(adapter, args.baseline)
        except ValueError as exc:
            logger.info(f"ERROR: {exc}")
            sys.exit(2)

    experience_dir = experience.candidates_dir(run_name)
    staging_dir = get_experience_root() / run_name / "staging"
    experience_dir.mkdir(parents=True, exist_ok=True)

    # Acquire a per-run lockfile so concurrent runs with the same --run-name
    # fail loudly. Stale locks (dead PID) are taken over silently. --resume
    # bypasses the live-PID refusal.
    try:
        acquire_run_lock(run_name, allow_resume=getattr(args, "resume", False))
    except RunLockError as exc:
        logger.info(f"ERROR: {exc}")
        sys.exit(2)

    holdout_bench: Optional[Benchmark] = None
    holdout_dir: Optional[Path] = None
    if args.holdout_benchmark:
        holdout_bench = load_benchmark(
            args.holdout_benchmark, split=getattr(args, "holdout_split", None),
        )
        # Namespaced holdout dir: each run owns its own baseline-holdout +
        # evo_NNN_holdout entries, so two concurrent runs never stomp each
        # other's holdout results even when they share a holdout family:split.
        holdout_dir = experience.candidates_dir(f"{run_name}__{holdout_bench.name}")
        holdout_dir.mkdir(parents=True, exist_ok=True)

    if not SHARED_PROPOSER_INSTRUCTIONS_PATH.exists():
        logger.info(f"ERROR: {SHARED_PROPOSER_INSTRUCTIONS_PATH} not found")
        sys.exit(1)

    history_path = get_experience_root() / run_name / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    all_task_names = _build_task_pool(bench, args.fast)
    batch_size = args.batch_size
    batch_rng = random.Random(args.seed) if args.seed is not None else random.Random()

    loaded_history = _load_history_from_disk(history_path)
    effective_start_from, effective_iterations, resume_from_proposal_epoch = (
        _resolve_resume_window(args, loaded_history)
    )
    if (
        getattr(args, "_auto_resume_from_proposal", False)
        and effective_iterations > 0
        and int(getattr(args, "candidates_per_iter", 1) or 1) > 1
        and _proposal_checkpoint_exists(history_path, effective_start_from)
    ):
        resume_from_proposal_epoch = effective_start_from
        logger.warning(
            f"--fresh retry: reusing proposal checkpoint for epoch "
            f"{effective_start_from} instead of invoking proposer again."
        )
    state = LoopState(
        args=args,
        bench=bench,
        bench_target=target,
        workspace_root=workspace_root,
        run_name=run_name,
        experience_dir=experience_dir,
        staging_dir=staging_dir,
        holdout_bench=holdout_bench,
        holdout_dir=holdout_dir,
        history_path=history_path,
        all_task_names=all_task_names,
        batch_size=batch_size,
        _batch_rng=batch_rng,
        effective_start_from=effective_start_from,
        effective_iterations=effective_iterations,
        resume_from_proposal_epoch=resume_from_proposal_epoch,
        history=loaded_history,
        iterations_since_skill_evolve=_reconstruct_skill_counter(loaded_history),
    )
    state.experiment_config = _build_experiment_config(
        args, bench, target, all_task_names, batch_size,
    )
    state.run_repro_manifest = _build_reproducibility_manifest(
        args=args, bench_name=run_name, workspace_root=workspace_root,
    )

    if batch_size and state.effective_start_from > 1:
        has_baseline_row = (
            args.baseline is not None and experience.has_candidate(experience_dir, "baseline")
        )
        n_skip = (state.effective_start_from - 1) + (1 if has_baseline_row else 0)
        for _ in range(n_skip):
            state._pop_batch()
        print(
            f"[LOOP] Batch RNG: skipped {n_skip} batches "
            f"for resume at start_from={state.effective_start_from}"
        )

    accept_on_holdout = getattr(args, "accept_on_holdout", False)
    state.best_rate = max(
        (
            r for r in (
                accept_reward_from_row(h, accept_on_holdout) for h in state.history
            ) if r is not None
        ),
        default=0.0,
    )
    return state


def _print_run_header(state: LoopState) -> None:
    from meta_agent.loop.proposer import SURFACE_LOCK_DESCRIPTION

    args = state.args
    bench = state.bench
    target = state.bench_target
    logger.info(f"=== Harness Optimizer Outer Loop ===")
    bench_desc = f"{bench.name} (type={bench.type}"
    if bench.family and bench.split:
        bench_desc += f", family={bench.family}, split={bench.split}"
    bench_desc += f", harness={target.name}, runtime={target.default_runtime})"
    logger.info(f"Benchmark: {bench_desc}")
    logger.info(f"Run name: {state.run_name}")
    logger.info(f"Experience: {rel_to_workspace(state.experience_dir)}")
    logger.info(f"Iterations: {state.effective_iterations}")
    if getattr(args, "resume", False) and (
        state.effective_start_from != args.start_from
        or state.effective_iterations != args.iterations
    ):
        requested_final_epoch = args.start_from + args.iterations - 1
        effective_final_epoch = (
            state.effective_start_from + state.effective_iterations - 1
        )
        logger.info(
            f"Resume window: {args.start_from}..{requested_final_epoch} "
            f"-> {state.effective_start_from}..{effective_final_epoch}"
        )
    if (
        getattr(args, "resume_from_proposal", False)
        and state.resume_from_proposal_epoch is None
        and state.effective_start_from != args.start_from
    ):
        logger.info(
            "Resume-from-proposal: already advanced past the requested start "
            "epoch; ignoring stale proposal checkpoint hint."
        )
    logger.info(f"Eval model: {args.model}")
    logger.info(f"Proposer model: {args.proposer_model}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Fast: {args.fast}")
    logger.info(f"Task pool: {len(state.all_task_names)} tasks")
    if state.batch_size:
        logger.info(f"Batch size: {state.batch_size} (seed={args.seed})")
    if args.surface_lock:
        print(
            f"[LOOP] Surface lock: {args.surface_lock} "
            f"({SURFACE_LOCK_DESCRIPTION[args.surface_lock]})"
        )
    if args.evolve_skill:
        logger.info(f"Skill evolution: every {args.skill_evolve_every} iterations")
    if state.holdout_bench:
        logger.info(f"Holdout: {state.holdout_bench.name}")
    print()
