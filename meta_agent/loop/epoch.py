"""One epoch of the loop: propose → validate → persist → evaluate → record.

Plus the cross-epoch glue (`_maybe_run_baseline`, `_seed_history_from_baseline_on_disk`,
`_maybe_evolve_skill`, `run_evaluation`).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from meta_agent.core import experience
from meta_agent.core.benchmark import Benchmark, primary_reward, reward_or_none
from meta_agent.utils.logging import get_logger
from meta_agent.core.paths import rel_to_workspace
from meta_agent.core.targets import AgentTarget, get_target

from meta_agent.loop.proposer import (
    ProposerRunResult,
    invoke_proposer,
    write_proposer_cost_sidecar,
)
from meta_agent.loop.reports import write_epoch_report
from meta_agent.loop.skill_evolver import invoke_skill_evolver
from meta_agent.loop.state import (
    LoopState,
    PROPOSER_INSTRUCTIONS_DIR,
    _compute_score_delta,
    _parse_tasks_csv,
    _select_parent_candidate_name,
    _spark,
    accept_reward_from_row,
    import_time,
)
from meta_agent.loop.smoke_gate import SmokeResult, smoke_candidate
from meta_agent.loop.validate import validate_config

logger = get_logger("loop")


def _display_reward_from_row(row: dict[str, Any], accept_on_holdout: bool) -> float:
    accept_reward = accept_reward_from_row(row, accept_on_holdout)
    if accept_reward is not None:
        return accept_reward
    return primary_reward(row)


def _run_smoke_and_persist(
    candidate_name: str,
    candidate_dir: Path,
    harness_path: Path,
    bench: Any,
    model: str,
) -> SmokeResult:
    """Smoke-test a single candidate; if it fails, persist as validation_error.

    Mirrors the ``validate_config``-failure path shape (validation_error.json
    + scores.json stub with ``validation_failed=true``) so failed smokes
    surface through ``meta-agent list`` and downstream audit tools exactly
    like shape-validation failures. The next proposer reads the same fields
    and can learn "don't do that" without any additional plumbing.

    Runs asynchronously inside ``asyncio.run`` to match the surrounding
    sync loop. Safe to call from ``_run_one_epoch`` and
    ``_run_one_epoch_multi``; the global event loop is not required.
    """
    import asyncio

    sr = asyncio.run(smoke_candidate(
        harness_path=harness_path,
        benchmark=bench,
        model=model,
    ))
    if sr.ok:
        return sr

    (candidate_dir / "validation_error.json").write_text(json.dumps({
        "error": sr.error or "smoke-gate rejected candidate",
        "stage": "smoke_gate",
        "pair_id": sr.pair_id,
        "duration_s": sr.duration_s,
    }, indent=2))
    (candidate_dir / "scores.json").write_text(json.dumps({
        "name": candidate_name,
        "validation_failed": True,
        "validation_error": sr.error,
        "smoke_failed": True,
        "smoke_pair_id": sr.pair_id,
        "n_tasks": 0,
        "n_passed": 0,
        "pass_rate": 0.0,
        "mean_reward": None,
        "total_cost_usd": None,
    }, indent=2))
    return sr


@dataclass
class StagedCandidate:
    """One candidate resolved from the staging directory layout.

    When k=1 the proposer writes directly to `staging/` (e.g. `staging/harness.py`)
    and `staging_subdir == staging_dir`. When k>1 the proposer writes each
    candidate to `staging/<descriptive_name>/harness.py` and `staging_subdir`
    points at that inner directory.
    """

    name: str                # "evo_003" (k=1) or "evo_003_ablate_prompt" (k>1)
    staging_subdir: Path     # absolute path to the proposer's output for this candidate
    manifest_entry: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ProposalCheckpoint:
    """Checkpoint for resuming a multi-candidate epoch after proposal."""

    epoch_idx: int
    batch_tasks_csv: Optional[str]
    batch_tasks_list: list[str]
    proposer_trace_path: Path
    proposer_run: Optional[ProposerRunResult]


def _resolve_candidates_from_staging(
    staging_dir: Path, epoch_idx: int, target: AgentTarget,
) -> list[StagedCandidate]:
    """Discover the set of candidates the proposer actually wrote.

    k=1 layout (all required files present directly under `staging/`):
        returns a single StagedCandidate named `evo_NNN`.

    k>1 layout (required files present inside one-or-more subdirectories):
        returns one StagedCandidate per matching subdirectory, named
        `evo_NNN_<subdir>`.

    Subdirs missing any required file are silently skipped (the loop will
    proceed with whichever candidates were fully written).
    """
    base_name = f"evo_{epoch_idx:03d}"
    required = target.required_written_files
    if not required:
        return [StagedCandidate(name=base_name, staging_subdir=staging_dir)]

    manifest_candidates = _resolve_candidates_from_manifest(
        staging_dir, base_name, required,
    )
    if manifest_candidates:
        return manifest_candidates

    if any((staging_dir / f).exists() for f in required):
        return [StagedCandidate(name=base_name, staging_subdir=staging_dir)]

    out: list[StagedCandidate] = []
    if not staging_dir.exists():
        return out
    for sub in sorted(staging_dir.iterdir()):
        if not sub.is_dir():
            continue
        if any((sub / f).exists() for f in required):
            out.append(StagedCandidate(
                name=f"{base_name}_{sub.name}", staging_subdir=sub,
            ))
    return out


def _safe_candidate_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "candidate"


def _candidate_name_from_manifest(base_name: str, raw_name: Any) -> str:
    if not isinstance(raw_name, str) or not raw_name.strip():
        return base_name
    safe = _safe_candidate_suffix(raw_name)
    return safe if safe.startswith(f"{base_name}") else f"{base_name}_{safe}"


def _manifest_staging_dir(staging_dir: Path, entry: dict[str, Any]) -> Path:
    raw_path = entry.get("path") or entry.get("file") or entry.get("harness")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
        if not path.is_absolute():
            path = staging_dir / path
        return path.parent if path.suffix else path
    raw_name = entry.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        candidate_dir = staging_dir / _safe_candidate_suffix(raw_name)
        if candidate_dir.is_dir():
            return candidate_dir
    return staging_dir


def _resolve_candidates_from_manifest(
    staging_dir: Path,
    base_name: str,
    required: tuple[str, ...],
) -> list[StagedCandidate]:
    manifest_path = staging_dir / "pending_eval.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Ignoring invalid proposal manifest: {manifest_path}")
        return []
    raw_candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    if not isinstance(raw_candidates, list):
        logger.warning(f"Ignoring proposal manifest without candidates list: {manifest_path}")
        return []

    out: list[StagedCandidate] = []
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            continue
        subdir = _manifest_staging_dir(staging_dir, entry)
        if any((subdir / f).exists() for f in required):
            out.append(StagedCandidate(
                name=_candidate_name_from_manifest(base_name, entry.get("name")),
                staging_subdir=subdir,
                manifest_entry=entry,
            ))
    if not out:
        logger.warning(f"Proposal manifest listed no complete candidates: {manifest_path}")
    return out


def _copy_optional_proposal_manifest(sc: StagedCandidate, candidate_dir: Path) -> None:
    if sc.manifest_entry is None:
        return
    (candidate_dir / "proposal_manifest.json").write_text(
        json.dumps(sc.manifest_entry, indent=2)
    )


def _copy_pending_eval_manifest(staging_dir: Path, candidate_dir: Path) -> None:
    src = staging_dir / "pending_eval.json"
    if not src.is_file():
        return
    try:
        manifest = json.loads(src.read_text())
    except json.JSONDecodeError:
        shutil.copy2(src, candidate_dir / "proposal_manifest.json")
        return
    entries = manifest.get("candidates") if isinstance(manifest, dict) else None
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        (candidate_dir / "proposal_manifest.json").write_text(
            json.dumps(entries[0], indent=2)
        )
        return
    shutil.copy2(src, candidate_dir / "proposal_manifest.json")


def _score_cost(scores: dict[str, Any]) -> Optional[float]:
    value = scores.get("total_cost_usd")
    return float(value) if isinstance(value, (int, float)) else None


def _combined_score_cost(scores: dict[str, Any]) -> Optional[float]:
    value = scores.get("total_cost_with_proposer_usd")
    if isinstance(value, (int, float)):
        return float(value)
    return _score_cost(scores)


def _fmt_cost(value: Optional[float]) -> str:
    return f"${value:.3f}" if isinstance(value, (int, float)) else "$?"




# --- Small helpers -------------------------------------------------------

def _copy_optional_proposal_notes(
    staging_dir: Path, candidate_dir: Path,
) -> Optional[dict[str, Any]]:
    src = staging_dir / "proposal_notes.json"
    if not src.exists():
        return None
    dest = candidate_dir / "proposal_notes.json"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    try:
        parsed = json.loads(dest.read_text())
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        return {"_parse_error": True, "path": str(dest)}


def _proposal_checkpoint_dir(state: LoopState) -> Path:
    return state.history_path.parent / "_internal" / "proposal_checkpoints"


def _proposal_checkpoint_path(state: LoopState, epoch_idx: int) -> Path:
    return _proposal_checkpoint_dir(state) / f"evo_{epoch_idx:03d}.json"


def _proposal_trace_path(state: LoopState, epoch_idx: int) -> Path:
    return _proposal_checkpoint_dir(state) / f"evo_{epoch_idx:03d}_proposer_trace.jsonl"


def _serialize_proposer_run(run: Optional[ProposerRunResult]) -> Optional[dict[str, Any]]:
    if run is None:
        return None
    return {
        "exit_code": run.exit_code,
        "cost_usd": run.cost_usd,
        "num_turns": run.num_turns,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cache_read_tokens": run.cache_read_tokens,
        "model": run.model,
        "cli": run.cli,
    }


def _deserialize_proposer_run(payload: Optional[dict[str, Any]]) -> Optional[ProposerRunResult]:
    if payload is None:
        return None
    return ProposerRunResult(
        exit_code=int(payload.get("exit_code", 0)),
        cost_usd=payload.get("cost_usd"),
        num_turns=payload.get("num_turns"),
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        cache_read_tokens=payload.get("cache_read_tokens"),
        model=payload.get("model"),
        cli=str(payload.get("cli") or ""),
    )


def _write_proposal_checkpoint(
    state: LoopState,
    *,
    epoch_idx: int,
    batch_tasks_csv: Optional[str],
    batch_tasks_list: list[str],
    proposer_trace_path: Path,
    proposer_run: Optional[ProposerRunResult],
) -> None:
    root = _proposal_checkpoint_dir(state)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch_idx": epoch_idx,
        "batch_tasks_csv": batch_tasks_csv,
        "batch_tasks_list": batch_tasks_list,
        "proposer_trace_path": str(proposer_trace_path),
        "proposer_run": _serialize_proposer_run(proposer_run),
        "batch_queue": list(state._batch_queue),
        "batch_rng_state": repr(state._batch_rng.getstate()) if state.batch_size else None,
        "created_at": import_time(),
    }
    _proposal_checkpoint_path(state, epoch_idx).write_text(json.dumps(payload, indent=2))


def _load_proposal_checkpoint(state: LoopState, epoch_idx: int) -> ProposalCheckpoint:
    path = _proposal_checkpoint_path(state, epoch_idx)
    if not path.exists():
        raise RuntimeError(
            f"--resume-from-proposal requested, but no proposal checkpoint exists for "
            f"epoch {epoch_idx} at {path}"
        )
    payload = json.loads(path.read_text())
    if int(payload.get("epoch_idx", -1)) != epoch_idx:
        raise RuntimeError(
            f"Proposal checkpoint at {path} does not match epoch {epoch_idx}"
        )
    batch_queue = payload.get("batch_queue")
    if isinstance(batch_queue, list):
        state._batch_queue = [str(item) for item in batch_queue]
    batch_rng_state = payload.get("batch_rng_state")
    if isinstance(batch_rng_state, str) and state.batch_size:
        state._batch_rng.setstate(ast.literal_eval(batch_rng_state))
    return ProposalCheckpoint(
        epoch_idx=epoch_idx,
        batch_tasks_csv=payload.get("batch_tasks_csv"),
        batch_tasks_list=[
            str(item) for item in payload.get("batch_tasks_list", [])
        ],
        proposer_trace_path=Path(str(payload["proposer_trace_path"])),
        proposer_run=_deserialize_proposer_run(payload.get("proposer_run")),
    )


def _clear_proposal_checkpoint(state: LoopState, epoch_idx: int) -> None:
    for path in (
        _proposal_checkpoint_path(state, epoch_idx),
        _proposal_trace_path(state, epoch_idx),
    ):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            logger.warning(f"Could not remove proposal checkpoint artifact {path}")


def _should_resume_from_proposal(state: LoopState, epoch_idx: int) -> bool:
    return bool(
        state.resume_from_proposal_epoch is not None
        and epoch_idx == state.resume_from_proposal_epoch
    )


_HOLDOUT_EPOCH_META_KEYS: tuple[str, ...] = (
    # NOTE: deliberately NOT including "holdout" here — we want the aggregate
    # holdout score (mean_reward / pass_rate / n_passed / n_tasks / cost) to
    # be visible to the proposer so it can steer against train-overfitting.
    # See the shared proposer instructions and the internal experiment plan.
    # The per-pair / per-task holdout data still lives in a sibling experience
    # dir (`<run>__<holdout-bench>-<split>/`) that the proposer's prompt never
    # names, so individual val pairs / tasks remain hidden.
    "holdout_delta",
    "holdout_is_winner",
)


def _strip_holdout_from_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _HOLDOUT_EPOCH_META_KEYS}


def _write_epoch_meta(candidate_dir: Path, payload: dict[str, Any]) -> None:
    """Write two versions of epoch_meta so holdout numbers never leak to the proposer.

    * Public `epoch_meta.json` — stripped of holdout fields (proposer-readable).
    * Internal `_internal/epoch_meta.json` — full payload (orchestrator-only).

    Proposer instructions forbid reading `_internal/` paths; stripping the
    public copy is belt-and-suspenders.
    """
    internal_dir = candidate_dir / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    (internal_dir / "epoch_meta.json").write_text(json.dumps(payload, indent=2))
    (candidate_dir / "epoch_meta.json").write_text(
        json.dumps(_strip_holdout_from_meta(payload), indent=2)
    )


def _persist_staging_to_candidate(
    staging_dir: Path, candidate_dir: Path, target: AgentTarget, config_path: Path,
) -> None:
    """Copy the validated staging artifacts into the candidate dir."""
    if target.is_file_based or target.name == "program_harness":
        src_dir = staging_dir if staging_dir.is_dir() else config_path.parent
        for item in src_dir.iterdir():
            dest = candidate_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    else:
        shutil.copy2(config_path, candidate_dir / config_path.name)


_TRANSIENT_EXC_NAMES: tuple[str, ...] = (
    # httpx / requests / urllib3
    "HTTPStatusError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "NetworkError",
    # stdlib
    "ConnectionError", "ConnectionResetError", "ConnectionRefusedError",
    "TimeoutError",
    # boto3 throttling / Bedrock transient
    "ThrottlingException", "ServiceUnavailableException",
    "ModelStreamErrorException", "ModelTimeoutException",
    # HuggingFace hub
    "HfHubHTTPError", "HTTPError",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if `exc` (or any cause in its chain) is a retry-worthy class name.

    We match by class name rather than importing every upstream library, so
    the allowlist stays robust across SDK version bumps.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _TRANSIENT_EXC_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _write_eval_error(
    experience_dir: Optional[Path], name: str, error: dict[str, Any],
) -> None:
    """Persist `error` as a sidecar under the candidate dir + a stub scores.json.

    The scores.json stub has ``eval_failed: true`` so `meta-agent list`
    surfaces the failure like a validation failure. The error sidecar
    preserves the type/message/traceback so the next proposer can diagnose.
    """
    if experience_dir is None:
        return
    try:
        candidate_dir = experience_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        (candidate_dir / "eval_error.json").write_text(
            _json.dumps(error, indent=2)
        )
        # Only write a stub scores.json if the real one didn't make it to disk
        # (a partial eval_runner.run that did write scores then raised should
        # keep its real scores).
        scores_path = candidate_dir / "scores.json"
        if not scores_path.exists():
            scores_path.write_text(_json.dumps({
                "name": name,
                "eval_failed": True,
                "eval_error": error.get("message"),
                "n_tasks": 0,
                "n_passed": 0,
                "pass_rate": 0.0,
                "mean_reward": None,
                "total_cost_usd": None,
            }, indent=2))
    except OSError as write_exc:
        logger.warning(f"Could not persist eval error for {name}: {write_exc}")


def _clear_eval_artifacts(candidate_dir: Path) -> None:
    """Remove stale search/holdout eval outputs before rerunning a candidate."""
    for name in (
        "scores.json",
        "summary.md",
        "eval_error.json",
        "validation_error.json",
        "category_scores.json",
        "epoch_meta.json",
    ):
        path = candidate_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning(f"Could not remove stale artifact {path}")

    per_task_dir = candidate_dir / "per_task"
    if per_task_dir.exists():
        shutil.rmtree(per_task_dir, ignore_errors=True)

    internal_epoch_meta = candidate_dir / "_internal" / "epoch_meta.json"
    if internal_epoch_meta.exists():
        try:
            internal_epoch_meta.unlink()
        except OSError:
            logger.warning(f"Could not remove stale artifact {internal_epoch_meta}")


def _maybe_commit_modal_experience_volume(label: str) -> None:
    """Best-effort Modal Volume commit barrier before cross-container reads."""
    import importlib
    import os

    module_name = os.environ.get("META_AGENT_MODAL_RUNNER_MODULE", "").strip()
    if not module_name:
        return
    try:
        module = importlib.import_module(module_name)
        commit = getattr(module, "_commit_experience_volume_barrier", None)
        if callable(commit):
            commit(label)
    except Exception as exc:  # noqa: BLE001 - commit barrier is best-effort
        logger.warning(
            f"Could not commit Modal experience volume ({label}): "
            f"{type(exc).__name__}: {exc}"
        )


def run_evaluation(
    config_path: Path,
    name: str,
    model: str,
    benchmark_path: str,
    fast: bool,
    tasks: Optional[str],
    concurrency: int,
    split: Optional[str] = None,
    experience_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> Optional[dict[str, Any]]:
    """Thin in-process wrapper over `eval_runner.run`. Returns scores or None.

    Transient failures (HTTP 5xx, connection errors, rate-limit throttles,
    HF hub timeouts) retry up to ``max_retries`` times with exponential
    backoff (2s, 4s, 8s). Non-transient failures raise once and surface as
    an ``eval_error.json`` sidecar under the candidate dir so the proposer
    can read why the epoch died.
    """
    import time
    import traceback as _tb

    from meta_agent.commands import evaluate

    logger.info(f"Running evaluation: {name}")

    attempts = 0
    last_exc: Optional[BaseException] = None
    while attempts < max_retries:
        attempts += 1
        try:
            scores = evaluate.run(
                benchmark_path=benchmark_path,
                config_path=str(config_path),
                name=name,
                split=split,
                model=model,
                fast=fast,
                tasks=tasks,
                concurrency=concurrency,
                experience_dir=experience_dir,
            )
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_error(exc)
            logger.warning(
                f"Evaluation {name!r} attempt {attempts}/{max_retries} "
                f"raised {type(exc).__name__}: {exc}  "
                f"(transient={transient})"
            )
            if not transient or attempts >= max_retries:
                break
            backoff = 2 ** attempts  # 2s, 4s, 8s
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
            continue
        # Success path.
        if scores is None:
            logger.info("Evaluation returned no scores (dry-run?)")
            return None
        return scores

    # All retries exhausted.
    error_payload = {
        "type": type(last_exc).__name__ if last_exc else "UnknownError",
        "message": str(last_exc) if last_exc else "eval_runner raised",
        "transient": _is_transient_error(last_exc) if last_exc else False,
        "attempts": attempts,
        "traceback": _tb.format_exception(type(last_exc), last_exc, last_exc.__traceback__)
                     if last_exc else None,
    }
    logger.error(
        f"Evaluation {name!r} FAILED after {attempts} attempt(s): "
        f"{error_payload['type']}: {error_payload['message']}"
    )
    _write_eval_error(experience_dir, name, error_payload)
    return None


# --- Cross-epoch glue ----------------------------------------------------

def _maybe_run_baseline(state: LoopState) -> None:
    """Run baseline on train (and holdout, if configured) when missing on disk.

    Two independent checks:
    1. Train baseline: run once per benchmark. Skipped if a non-baseline evo
       candidate already exists (which would mean mid-run resume — re-running
       baseline would desync the batch RNG and shift later batch boundaries).
    2. Holdout baseline: run once per holdout benchmark. Required so
       --accept-on-holdout has a correct floor (baseline-holdout-reward);
       otherwise evo candidates get compared to baseline's train reward,
       which is a category mismatch.
    """
    from meta_agent.commands.inspect import list_candidates

    if state.args.baseline is None:
        return

    baseline_config = state.args.baseline
    experience_has_baseline = experience.has_candidate(state.experience_dir, "baseline")
    mid_run_resume = any(
        c.name.startswith("evo_")
        for c in experience.iter_candidates(state.experience_dir)
    )

    if not experience_has_baseline and not mid_run_resume:
        baseline_batch = state.next_batch()
        if baseline_batch:
            logger.info(f"Running baseline: {baseline_config} (batch: {baseline_batch})")
        else:
            logger.info(f"Running baseline: {baseline_config}")

        baseline_scores = run_evaluation(
            config_path=Path(baseline_config),
            name="baseline",
            model=state.args.model,
            benchmark_path=state.args.benchmark,
            split=getattr(state.args, "split", None),
            fast=state.args.fast if not baseline_batch else False,
            tasks=baseline_batch,
            concurrency=state.args.concurrency,
            experience_dir=state.experience_dir,
        )
        if baseline_scores:
            rate = baseline_scores["pass_rate"]
            print(
                f"[LOOP] Baseline: {baseline_scores['n_passed']}/"
                f"{baseline_scores['n_tasks']} ({rate:.0%})"
            )
        else:
            logger.info("Baseline evaluation failed")

    holdout_has_baseline = (
        state.holdout_dir is not None
        and experience.has_candidate(state.holdout_dir, "baseline")
    )
    if (
        state.holdout_dir is not None
        and state.args.holdout_benchmark
        and not holdout_has_baseline
    ):
        logger.info(f"Running baseline on holdout: {state.args.holdout_benchmark}")
        baseline_ho_scores = run_evaluation(
            config_path=Path(baseline_config),
            name="baseline",
            model=state.args.model,
            benchmark_path=state.args.holdout_benchmark,
            split=getattr(state.args, "holdout_split", None),
            fast=False,
            tasks=None,
            concurrency=state.args.concurrency,
            experience_dir=state.holdout_dir,
        )
        if baseline_ho_scores:
            rate = baseline_ho_scores["pass_rate"]
            print(
                f"[LOOP] Baseline holdout: {baseline_ho_scores['n_passed']}/"
                f"{baseline_ho_scores['n_tasks']} ({rate:.0%})"
            )
        else:
            logger.info("Baseline holdout evaluation failed")

    _maybe_write_baseline_epoch_meta(state)
    _maybe_commit_modal_experience_volume("after-baseline")

    print()
    list_candidates(state.experience_dir)
    print()


def _maybe_write_baseline_epoch_meta(state: LoopState) -> None:
    """Write ``baseline/epoch_meta.json`` if missing, to mirror evo_* candidates.

    Baseline is evaluated via ``run_evaluation`` (not the evo-epoch path), so
    it doesn't get an ``epoch_meta.json`` by default. That hides the baseline's
    holdout aggregate score from the proposer, which needs a concrete floor to
    judge whether its own harness improves on vs regresses against.

    This helper reads the on-disk search and holdout scores, builds a minimal
    ``epoch_meta`` payload (``search`` + optional ``holdout`` aggregate blocks),
    and writes it via ``_write_epoch_meta`` — which applies the same
    holdout-stripping contract we use for evo candidates (drops
    ``holdout_delta`` / ``holdout_is_winner`` on the public file, keeps the
    ``holdout`` aggregate visible).

    Idempotent: skips when the file already exists or baseline scores look
    unhealthy (eval/validation failure).
    """
    baseline_dir = state.experience_dir / "baseline"
    meta_path = baseline_dir / "epoch_meta.json"
    if meta_path.exists():
        return
    baseline_scores = experience.load_scores(baseline_dir)
    if baseline_scores is None:
        return
    if baseline_scores.get("eval_failed") or baseline_scores.get("validation_failed"):
        return

    payload: dict[str, Any] = {
        "epoch": 0,
        "candidate_name": "baseline",
        "parent_name": None,
        "status": "completed",
        "search": {
            "reward": reward_or_none(baseline_scores),
            "pass_rate": baseline_scores.get("pass_rate"),
            "n_passed": baseline_scores.get("n_passed"),
            "n_tasks": baseline_scores.get("n_tasks"),
            "cost_usd": baseline_scores.get("total_cost_usd"),
        },
        "timestamp": import_time(),
    }

    if state.holdout_dir is not None:
        baseline_ho_scores = experience.load_scores(state.holdout_dir / "baseline")
        if (
            baseline_ho_scores is not None
            and not baseline_ho_scores.get("eval_failed")
            and not baseline_ho_scores.get("validation_failed")
        ):
            payload["holdout"] = {
                "reward": reward_or_none(baseline_ho_scores),
                "pass_rate": baseline_ho_scores.get("pass_rate"),
                "n_passed": baseline_ho_scores.get("n_passed"),
                "n_tasks": baseline_ho_scores.get("n_tasks"),
                "cost_usd": baseline_ho_scores.get("total_cost_usd"),
            }

    _write_epoch_meta(baseline_dir, payload)


def _seed_history_from_baseline_on_disk(state: LoopState) -> None:
    """If history.json is empty but a baseline candidate exists on disk, add a row.

    Also populates `holdout_reward` from `baseline/scores.json` on the holdout
    volume when present, so the acceptance gate has a correct baseline floor
    when --accept-on-holdout is set.
    """
    if state.history:
        return
    bs = experience.load_scores(state.experience_dir / "baseline")
    if bs is None:
        return
    row: dict[str, Any] = {
        "name": "baseline",
        "reward": primary_reward(bs),
        "pass_rate": bs["pass_rate"],
        "n_passed": bs["n_passed"],
        "n_tasks": bs["n_tasks"],
        "cost_usd": bs.get("total_cost_usd"),
        "timestamp": import_time(),
    }
    if state.holdout_dir is not None:
        bs_ho = experience.load_scores(state.holdout_dir / "baseline")
        if bs_ho is not None:
            row["holdout_reward"] = primary_reward(bs_ho)
            row["holdout_pass_rate"] = bs_ho.get("pass_rate")
            row["holdout_n_passed"] = bs_ho.get("n_passed")
            row["holdout_n_tasks"] = bs_ho.get("n_tasks")
            row["holdout_cost"] = bs_ho.get("total_cost_usd")
    state.history.append(row)
    state.write_history()
    accept_on_holdout = getattr(state.args, "accept_on_holdout", False)
    baseline_row_reward = accept_reward_from_row(state.history[-1], accept_on_holdout)
    if baseline_row_reward is not None:
        state.best_rate = max(state.best_rate, baseline_row_reward)


def _maybe_evolve_skill(state: LoopState) -> None:
    """Run the meta-proposer if we've accumulated enough iterations."""
    if not state.args.evolve_skill:
        return
    if len(state.iterations_since_skill_evolve) < state.args.skill_evolve_every:
        return

    print(f"\n{'='*60}")
    print(
        f"  Skill Evolution — analyzing "
        f"{len(state.iterations_since_skill_evolve)} iterations"
    )
    print(f"{'='*60}\n")
    evolved = invoke_skill_evolver(
        state.iterations_since_skill_evolve,
        staging_dir=state.staging_dir,
        experience_dir=state.experience_dir,
        model=state.args.proposer_model,
        skill_path=PROPOSER_INSTRUCTIONS_DIR / state.bench_target.skill_filename,
    )
    if evolved:
        state.iterations_since_skill_evolve = []
    else:
        logger.info("Instruction evolution failed, continuing with current proposer instructions")


# --- Epoch body ----------------------------------------------------------

def _record_epoch_success(
    state: LoopState,
    epoch_idx: int,
    evo_name: str,
    scores: dict[str, Any],
    parent_scores: Optional[dict[str, Any]],
    parent_holdout_scores: Optional[dict[str, Any]],
    epoch_meta: dict[str, Any],
    candidate_dir: Path,
    target: AgentTarget,
) -> None:
    """Print epoch summary, update history, run holdout eval if configured.

    Acceptance logic:

    * Default (no --accept-on-holdout): `state.best_rate` tracks the best
      search-set reward; a candidate is "new best" when its search reward
      exceeds the prior best. This is the original behavior.
    * With --accept-on-holdout: we run the holdout eval first, then gate
      "new best" on holdout reward. This implements paper 1's stated
      policy ("keep only if it improves holdout accuracy") and is the
      generic fix for proposer overfitting on the search set. It applies
      to every benchmark that sets `--holdout-benchmark`, not just any
      one adapter.
    """
    search_reward = primary_reward(scores)
    cost = _score_cost(scores)
    accept_on_holdout = getattr(state.args, "accept_on_holdout", False)

    epoch_meta["status"] = "search_passed"
    epoch_meta["search"] = {
        "reward": reward_or_none(scores),
        "pass_rate": scores.get("pass_rate"),
        "n_passed": scores.get("n_passed"),
        "n_tasks": scores.get("n_tasks"),
        "cost_usd": scores.get("total_cost_usd"),
    }
    epoch_meta["search_delta"] = _compute_score_delta(scores, parent_scores)

    # History row first with search-side stats; holdout fields are filled in
    # below if holdout runs successfully.
    state.history.append({
        "name": evo_name,
        "reward": search_reward,
        "pass_rate": scores["pass_rate"],
        "n_passed": scores["n_passed"],
        "n_tasks": scores["n_tasks"],
        "cost_usd": cost,
        "total_cost_with_proposer_usd": _combined_score_cost(scores),
        "timestamp": import_time(),
    })

    # Run holdout BEFORE the is-best decision so --accept-on-holdout can
    # gate on it. When --accept-on-holdout is off, this is purely
    # informational (matches prior behavior in substance, just reordered).
    holdout_reward: Optional[float] = None
    if state.holdout_dir and state.args.holdout_benchmark:
        holdout_name = f"{evo_name}_holdout"
        logger.info("holdout: evaluating on held-out split...")
        eval_config = candidate_dir if target.is_file_based else candidate_dir / target.module_filename
        holdout_scores = run_evaluation(
            config_path=eval_config,
            name=holdout_name,
            model=state.args.model,
            benchmark_path=state.args.holdout_benchmark,
            split=getattr(state.args, "holdout_split", None),
            fast=False,
            tasks=None,
            concurrency=state.args.concurrency,
            experience_dir=state.holdout_dir,
        )
        if holdout_scores:
            holdout_reward = primary_reward(holdout_scores)
            ho_cost = _score_cost(holdout_scores)
            logger.info(f"holdout: {holdout_reward:.1%}  cost={_fmt_cost(ho_cost)}")
            epoch_meta["status"] = "completed"
            epoch_meta["holdout"] = {
                "reward": reward_or_none(holdout_scores),
                "pass_rate": holdout_scores.get("pass_rate"),
                "n_passed": holdout_scores.get("n_passed"),
                "n_tasks": holdout_scores.get("n_tasks"),
                "cost_usd": holdout_scores.get("total_cost_usd"),
            }
            epoch_meta["holdout_delta"] = _compute_score_delta(
                holdout_scores, parent_holdout_scores,
            )
            state.history[-1]["holdout_reward"] = holdout_reward
            state.history[-1]["holdout_cost"] = ho_cost
            state.history[-1]["holdout_n_passed"] = holdout_scores.get("n_passed", 0)
            state.history[-1]["holdout_n_tasks"] = holdout_scores.get("n_tasks", 0)
            state.history[-1]["holdout_pass_rate"] = holdout_scores.get("pass_rate", 0)
        else:
            logger.warning("holdout: FAILED")
            epoch_meta["status"] = "holdout_failed"

    # Select acceptance metric. When --accept-on-holdout is set but
    # holdout failed, we fall back to search reward for the gate (safer
    # than skipping the candidate entirely) and log the fallback.
    if accept_on_holdout and holdout_reward is not None:
        accept_reward = holdout_reward
        gate_label = "holdout"
    else:
        if accept_on_holdout and holdout_reward is None:
            logger.warning(
                "--accept-on-holdout was set but holdout eval produced no "
                "reward; falling back to search reward for acceptance gate."
            )
        accept_reward = search_reward
        gate_label = "search"

    is_best = accept_reward > state.best_rate
    if is_best:
        state.best_rate = accept_reward
    arrow = " *** NEW BEST ***" if is_best else ""

    print(f"\n  {'─'*50}")
    search_line = f"  EPOCH {epoch_idx} RESULT: search={search_reward:.1%}"
    if holdout_reward is not None:
        search_line += f"  holdout={holdout_reward:.1%}"
    search_line += f"  cost={_fmt_cost(cost)}{arrow}"
    print(search_line)
    print(f"  Best so far ({gate_label}): {state.best_rate:.1%}")

    state.write_history()
    write_epoch_report(state, epoch_idx, [evo_name])

    display_values = [_display_reward_from_row(h, accept_on_holdout) for h in state.history]
    rates = " -> ".join(f"{v:.0%}" for v in display_values[-8:])
    spark = _spark(display_values)
    print(f"  History ({gate_label}): {rates}  {spark}")
    print(f"  {'─'*50}")


def _run_one_epoch(state: LoopState, epoch_idx: int) -> bool:
    """Propose → validate → persist → evaluate → record. Mutates state.

    Returns True once the epoch produced a candidate attempt worth counting.
    Returns False only for proposer/infrastructure failures that produced no
    candidate artifacts and should be retried without consuming the epoch.

    When `args.candidates_per_iter > 1`, dispatches to `_run_one_epoch_multi`.
    The k=1 path is kept intact to preserve byte-for-byte identical behavior
    with pre-change runs.
    """
    if getattr(state.args, "candidates_per_iter", 1) > 1:
        return _run_one_epoch_multi(state, epoch_idx)

    args = state.args
    bench = state.bench
    evo_name = f"evo_{epoch_idx:03d}"
    total_iters = state.effective_start_from + state.effective_iterations - 1

    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch_idx}/{total_iters}  ({evo_name})")
    print(f"{'='*60}")
    print(f"\n  [1/3] Proposing new config...")

    candidate_dir = state.experience_dir / evo_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    proposer_trace = candidate_dir / "proposer_trace.jsonl"
    parent_name = _select_parent_candidate_name(
        state.history,
        accept_on_holdout=getattr(state.args, "accept_on_holdout", False),
    )

    parent_scores = (
        experience.load_scores(state.experience_dir / parent_name)
        if parent_name else None
    )
    parent_holdout_scores: Optional[dict[str, Any]] = None
    if state.holdout_dir and args.holdout_benchmark and parent_name:
        parent_holdout_scores = experience.load_scores(
            state.holdout_dir / f"{parent_name}_holdout"
        )

    target = state.bench_target
    proposer_result = invoke_proposer(
        staging_dir=state.staging_dir,
        experience_dir=state.experience_dir,
        bench_name=state.run_name,
        trace_path=proposer_trace,
        model=args.proposer_model,
        harness=target.name,
        proposer_cli=args.proposer_cli,
        max_turns=args.proposer_max_turns,
        parent_seed_dir=None,
        surface_lock=args.surface_lock,
        benchmark_path=args.benchmark,
    )
    if not proposer_result.success:
        logger.info(
            f"Proposer failed at epoch {epoch_idx}; retrying without consuming epoch"
        )
        return False

    config_path = (
        state.staging_dir if target.is_file_based
        else state.staging_dir / target.module_filename
    )
    print(f"  [2/3] Validating config...")
    vr = validate_config(config_path, bench_type=bench.type, harness=target.name)
    if not vr:
        # Persist the failed candidate so the next proposer can see what the
        # validation error was. Keeps information that would otherwise be
        # thrown away; the proposer learns "don't make this mistake again."
        _persist_staging_to_candidate(state.staging_dir, candidate_dir, target, config_path)
        _copy_optional_proposal_notes(state.staging_dir, candidate_dir)
        _copy_pending_eval_manifest(state.staging_dir, candidate_dir)
        (candidate_dir / "validation_error.json").write_text(json.dumps({
            "error": vr.error or "unknown validation failure",
            "traceback": vr.traceback,
            "harness": target.name,
            "bench_type": bench.type,
        }, indent=2))
        (candidate_dir / "scores.json").write_text(json.dumps({
            "name": evo_name,
            "validation_failed": True,
            "validation_error": vr.error,
            "n_tasks": 0,
            "n_passed": 0,
            "pass_rate": 0.0,
            "mean_reward": None,
            "total_cost_usd": None,
        }, indent=2))
        print(f"  [2/3] FAILED — persisted as {evo_name} with validation_error.json ({vr.error})")
        return True

    _persist_staging_to_candidate(state.staging_dir, candidate_dir, target, config_path)
    proposal_notes = _copy_optional_proposal_notes(state.staging_dir, candidate_dir)
    _copy_pending_eval_manifest(state.staging_dir, candidate_dir)
    if proposer_result.run is not None:
        write_proposer_cost_sidecar(candidate_dir, proposer_result.run, shared_across=1)

    # Smoke-gate: runtime check that shape-validation can't do.
    # Runs one real benchmark item through the candidate; if the hook
    # callback crashes (e.g., malformed hookSpecificOutput shape), the
    # gate rejects here instead of burning a full 232-pair eval. See
    # meta_agent/loop/smoke_gate.py for the benchmark-type dispatch.
    print(f"  [2.5/3] Smoke-testing candidate...")
    eval_config = (
        candidate_dir if target.is_file_based
        else candidate_dir / target.module_filename
    )
    sr = _run_smoke_and_persist(
        candidate_name=evo_name,
        candidate_dir=candidate_dir,
        harness_path=eval_config,
        bench=bench,
        model=args.model,
    )
    if not sr.ok:
        print(
            f"  [2.5/3] SMOKE FAILED — persisted as {evo_name} with "
            f"validation_error.json ({sr.error[:120] if sr.error else ''})"
        )
        return True

    print(f"  [3/3] Evaluating on benchmark...")
    # eval_config already resolved above for the smoke-gate.
    batch_tasks = state.next_batch()
    batch_tasks_list = _parse_tasks_csv(batch_tasks)
    if batch_tasks:
        logger.info(f"batch: {batch_tasks}")

    epoch_meta: dict[str, Any] = {
        "epoch": epoch_idx,
        "candidate_name": evo_name,
        "parent_name": parent_name,
        "parent_reference": "best_so_far",
        "batch_tasks": batch_tasks_list,
        "batch_tasks_csv": batch_tasks,
        "status": "running",
        "search": None,
        "search_delta": None,
        "holdout": None,
        "holdout_delta": None,
        "models": {
            "eval_model": args.model,
            "proposer_model": args.proposer_model,
            "proposer_cli": args.proposer_cli,
        },
        "surface_lock": args.surface_lock,
        "proposal_notes": proposal_notes,
        "proposer_trace_path": rel_to_workspace(proposer_trace),
        "reproducibility_manifest": "reproducibility.json",
        "timestamp": import_time(),
    }
    reproducibility_manifest = {
        **state.run_repro_manifest,
        "epoch": epoch_idx,
        "candidate_name": evo_name,
        "parent_name": parent_name,
        "captured_at": import_time(),
    }
    (candidate_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility_manifest, indent=2)
    )

    scores = run_evaluation(
        config_path=eval_config,
        name=evo_name,
        model=args.model,
        benchmark_path=args.benchmark,
        split=getattr(args, "split", None),
        fast=args.fast if not batch_tasks else False,
        tasks=batch_tasks,
        concurrency=args.concurrency,
        experience_dir=state.experience_dir,
    )

    if scores:
        _record_epoch_success(
            state=state,
            epoch_idx=epoch_idx,
            evo_name=evo_name,
            scores=scores,
            parent_scores=parent_scores,
            parent_holdout_scores=parent_holdout_scores,
            epoch_meta=epoch_meta,
            candidate_dir=candidate_dir,
            target=target,
        )
    else:
        print(f"\n  EPOCH {epoch_idx} RESULT: FAILED (no scores)")
        epoch_meta["status"] = "search_failed"
        # Fold any persisted eval_error into the public epoch_meta so the
        # next proposer can read it without needing to crack open the sidecar.
        err_path = candidate_dir / "eval_error.json"
        if err_path.exists():
            try:
                epoch_meta["error"] = json.loads(err_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

    _write_epoch_meta(candidate_dir, epoch_meta)
    state.iterations_since_skill_evolve.append(evo_name)
    return True


# --- Multi-candidate epoch (k > 1) ---------------------------------------

def _build_epoch_meta(
    *,
    epoch_idx: int,
    candidate_name: str,
    parent_name: Optional[str],
    batch_tasks_csv: Optional[str],
    batch_tasks_list: list[str],
    args: Any,
    proposer_trace_ref: str,
    proposal_notes: Optional[dict[str, Any]],
    siblings: Optional[list[str]] = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "epoch": epoch_idx,
        "candidate_name": candidate_name,
        "parent_name": parent_name,
        "parent_reference": "best_so_far",
        "batch_tasks": batch_tasks_list,
        "batch_tasks_csv": batch_tasks_csv,
        "status": "running",
        "search": None,
        "search_delta": None,
        "holdout": None,
        "holdout_delta": None,
        "models": {
            "eval_model": args.model,
            "proposer_model": args.proposer_model,
            "proposer_cli": args.proposer_cli,
        },
        "surface_lock": args.surface_lock,
        "proposal_notes": proposal_notes,
        "proposer_trace_path": proposer_trace_ref,
        "reproducibility_manifest": "reproducibility.json",
        "timestamp": import_time(),
    }
    if siblings:
        meta["siblings"] = siblings
    return meta


def _on_modal() -> bool:
    """True when executing inside a Modal container.

    Detected via Modal's own environment variables (`MODAL_TASK_ID` and
    `MODAL_FUNCTION_CALL_ID`), which are set by Modal when a function runs
    remotely and are inherited by any subprocess the function spawns. We
    use env vars instead of `modal.is_local()` because the loop may run as
    a subprocess of `_run_meta_agent`, and `modal.is_local()` checks
    in-process state that doesn't survive across fork/exec.
    """
    return bool(
        os.environ.get("MODAL_TASK_ID")
        or os.environ.get("MODAL_FUNCTION_CALL_ID")
    )


def _run_evals_parallel(
    *,
    candidate_dirs: list[tuple[str, Path]],  # [(name, candidate_dir), ...]
    target: AgentTarget,
    args: Any,
    batch_tasks: Optional[str],
    experience_dir: Path,
    benchmark_path: Optional[str] = None,    # defaults to args.benchmark
    split: Optional[str] = None,              # defaults to args.split
) -> list[Optional[dict[str, Any]]]:
    """Run N evals in parallel against ``benchmark_path`` / ``split``.

    By default dispatches against the search benchmark + split (``args.benchmark``
    / ``args.split``). Pass ``benchmark_path`` and ``split`` explicitly to fan
    out holdout evals — the k>1 epoch uses this to run holdout on every sibling
    instead of just the best-on-search.

    Implementation: ``ThreadPoolExecutor`` inside the current process, always.

    Why not Modal fanout:

    We tried two flavours of nested Modal invocation (``Function.from_name`` +
    ``.spawn`` / direct import + ``.spawn``). Both have sharp edges:

    * ``Function.from_name(app, fn)`` resolves against the *deployed* app
      registry, which is a separate artifact from the ephemeral
      ``modal run`` session. When the deployment is stale, ``.get()``
      raises ``TypeError: unexpected keyword argument`` at runtime — and
      only after the proposer has already burned $2.
    * Direct imports return an *un-hydrated* Function handle when run
      inside a nested Modal worker (the App context is only live in the
      process that bootstrapped it). ``.spawn()`` immediately fails with
      ``ExecutionError: Function has not been hydrated``.

    Threading sidesteps both: evals are I/O-bound (waiting on Bedrock),
    not CPU-bound, and the orchestrator container is provisioned with
    16 CPU / 128GB memory, which is plenty for k up to ~10. Each worker
    thread calls ``run_evaluation`` in-process against the same shared
    volume, so results are visible immediately without commit/reload
    dances.

    Returns a list of score dicts (or None on failure) aligned with the
    input candidate order.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(candidate_dirs)
    if n == 0:
        return []

    bench_path = benchmark_path if benchmark_path is not None else args.benchmark
    bench_split = split if split is not None else getattr(args, "split", None)

    jobs: list[tuple[Path, str, str, Optional[str], Optional[str]]] = []
    for name, candidate_dir in candidate_dirs:
        eval_config = (
            candidate_dir if target.is_file_based
            else candidate_dir / target.module_filename
        )
        jobs.append((eval_config, name, bench_path, bench_split, batch_tasks))

    def _one(config_path: Path, name: str, this_benchmark_path: str,
             split_arg: Optional[str],
             tasks: Optional[str]) -> Optional[dict[str, Any]]:
        return run_evaluation(
            config_path=config_path,
            name=name,
            model=args.model,
            benchmark_path=this_benchmark_path,
            split=split_arg,
            fast=args.fast if not tasks else False,
            tasks=tasks,
            concurrency=args.concurrency,
            experience_dir=experience_dir,
        )

    if n == 1:
        logger.info("Running 1 eval in orchestrator")
        return [_one(*jobs[0])]

    logger.info(f"Running {n} evals in thread pool")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_one, *job) for job in jobs]
        return [f.result() for f in futures]


def _run_smokes_parallel(
    *,
    candidates: list[tuple[StagedCandidate, Path, Optional[dict[str, Any]]]],
    target: AgentTarget,
    bench: Benchmark,
    model: str,
) -> list[tuple[StagedCandidate, Path, Optional[dict[str, Any]], SmokeResult]]:
    """Smoke-test sibling candidates concurrently.

    Plan-RewardBench smoke is now a real SDK call, so doing five siblings
    serially would add minutes of idle wall time before the expensive batch
    evals even start. Running the smoke gate with the same fanout shape as
    evaluation keeps the stronger guard cheap enough to use every epoch.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not candidates:
        return []

    def _one(
        sc: StagedCandidate,
        candidate_dir: Path,
        notes: Optional[dict[str, Any]],
    ) -> tuple[StagedCandidate, Path, Optional[dict[str, Any]], SmokeResult]:
        harness_path = (
            candidate_dir if target.is_file_based
            else candidate_dir / target.module_filename
        )
        sr = _run_smoke_and_persist(
            candidate_name=sc.name,
            candidate_dir=candidate_dir,
            harness_path=harness_path,
            bench=bench,
            model=model,
        )
        return sc, candidate_dir, notes, sr

    n = len(candidates)
    logger.info(f"Running {n} smoke gates in thread pool")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_one, *job) for job in candidates]
        return [f.result() for f in futures]


def _run_one_epoch_multi(state: LoopState, epoch_idx: int) -> bool:
    """k>1 epoch: single proposer call produces N candidates; evaluate all in parallel.

    Kept separate from the k=1 path to preserve byte-for-byte identical
    behavior of the default (k=1) code path.
    """
    args = state.args
    bench = state.bench
    target = state.bench_target
    k = args.candidates_per_iter
    total_iters = state.effective_start_from + state.effective_iterations - 1
    base_name = f"evo_{epoch_idx:03d}"
    resume_checkpoint = _load_proposal_checkpoint(state, epoch_idx) if _should_resume_from_proposal(
        state, epoch_idx
    ) else None

    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch_idx}/{total_iters}  (k={k})")
    print(f"{'='*60}")
    if resume_checkpoint is not None:
        print(f"\n  [1/4] Reusing {k} proposed candidates from checkpoint...")
    else:
        print(f"\n  [1/4] Proposing {k} candidates (one proposer session)...")

    # Parent reference is identical to k=1: best-so-far by reward (or holdout).
    parent_name = _select_parent_candidate_name(
        state.history,
        accept_on_holdout=getattr(state.args, "accept_on_holdout", False),
    )
    parent_scores = (
        experience.load_scores(state.experience_dir / parent_name)
        if parent_name else None
    )
    parent_holdout_scores: Optional[dict[str, Any]] = None
    if state.holdout_dir and args.holdout_benchmark and parent_name:
        parent_holdout_scores = experience.load_scores(
            state.holdout_dir / f"{parent_name}_holdout"
        )

    if resume_checkpoint is not None:
        batch_tasks = resume_checkpoint.batch_tasks_csv
        batch_tasks_list = list(resume_checkpoint.batch_tasks_list)
        staging_trace = resume_checkpoint.proposer_trace_path
        proposer_run = resume_checkpoint.proposer_run
        if batch_tasks:
            logger.info(f"batch: {batch_tasks}")
    else:
        # Draw the batch ONCE for the epoch — every sibling evaluates on the same tasks.
        batch_queue_before_proposal = list(state._batch_queue)
        batch_rng_state_before_proposal = state._batch_rng.getstate()
        batch_tasks = state.next_batch()
        batch_tasks_list = _parse_tasks_csv(batch_tasks)
        if batch_tasks:
            logger.info(f"batch: {batch_tasks}")

        # Keep the proposer trace in a stable checkpoint location so a restarted
        # run can skip reproposal and still recover the same staged candidates.
        staging_trace = _proposal_trace_path(state, epoch_idx)
        staging_trace.parent.mkdir(parents=True, exist_ok=True)
        _maybe_commit_modal_experience_volume(f"before-proposer-epoch-{epoch_idx}")

        proposer_result = invoke_proposer(
            staging_dir=state.staging_dir,
            experience_dir=state.experience_dir,
            bench_name=state.run_name,
            trace_path=staging_trace,
            model=args.proposer_model,
            harness=target.name,
            proposer_cli=args.proposer_cli,
            max_turns=args.proposer_max_turns,
            parent_seed_dir=None,
            surface_lock=args.surface_lock,
            candidates_per_iter=k,
            benchmark_path=args.benchmark,
        )
        if not proposer_result.success:
            logger.info(
                f"Proposer failed at epoch {epoch_idx}; retrying without consuming epoch"
            )
            state._batch_queue = batch_queue_before_proposal
            state._batch_rng.setstate(batch_rng_state_before_proposal)
            _clear_proposal_checkpoint(state, epoch_idx)
            return False
        proposer_run = proposer_result.run
        _write_proposal_checkpoint(
            state,
            epoch_idx=epoch_idx,
            batch_tasks_csv=batch_tasks,
            batch_tasks_list=batch_tasks_list,
            proposer_trace_path=staging_trace,
            proposer_run=proposer_run,
        )

    staged = _resolve_candidates_from_staging(state.staging_dir, epoch_idx, target)
    if not staged:
        logger.info(
            f"No candidates discovered in staging dir for epoch {epoch_idx}; "
            "retrying without consuming epoch"
        )
        if resume_checkpoint is None:
            state._batch_queue = batch_queue_before_proposal
            state._batch_rng.setstate(batch_rng_state_before_proposal)
        _clear_proposal_checkpoint(state, epoch_idx)
        return False

    print(f"\n  [2/4] Validating {len(staged)} candidate(s)...")
    valid: list[tuple[StagedCandidate, Path, Optional[dict[str, Any]]]] = []
    for sc in staged:
        config_path = (
            sc.staging_subdir if target.is_file_based
            else sc.staging_subdir / target.module_filename
        )
        vr = validate_config(config_path, bench_type=bench.type, harness=target.name)
        if not vr:
            # Persist the failure so the next proposer can see what broke and
            # avoid repeating the same mistake. The candidate dir contains the
            # actual harness code the proposer wrote, a validation_error.json
            # with the error message + traceback, and a scores.json stub
            # marked `validation_failed: true` so `meta-agent list` surfaces it.
            candidate_dir = state.experience_dir / sc.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if resume_checkpoint is not None:
                _clear_eval_artifacts(candidate_dir)
            _persist_staging_to_candidate(sc.staging_subdir, candidate_dir, target, config_path)
            _copy_optional_proposal_notes(sc.staging_subdir, candidate_dir)
            _copy_optional_proposal_manifest(sc, candidate_dir)
            (candidate_dir / "validation_error.json").write_text(json.dumps({
                "error": vr.error or "unknown validation failure",
                "traceback": vr.traceback,
                "harness": target.name,
                "bench_type": bench.type,
            }, indent=2))
            (candidate_dir / "scores.json").write_text(json.dumps({
                "name": sc.name,
                "validation_failed": True,
                "validation_error": vr.error,
                "n_tasks": 0,
                "n_passed": 0,
                "pass_rate": 0.0,
                "mean_reward": None,
                "total_cost_usd": None,
            }, indent=2))
            print(f"    {sc.name}: FAILED validation ({vr.error})")
            continue

        candidate_dir = state.experience_dir / sc.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if resume_checkpoint is not None:
            _clear_eval_artifacts(candidate_dir)
        _persist_staging_to_candidate(sc.staging_subdir, candidate_dir, target, config_path)
        proposal_notes = _copy_optional_proposal_notes(sc.staging_subdir, candidate_dir)
        _copy_optional_proposal_manifest(sc, candidate_dir)
        if proposer_run is not None:
            # Share the single proposer session's cost across all staged siblings
            # (regardless of whether each passed validation). `len(staged)` is
            # the N the proposer was asked to produce.
            write_proposer_cost_sidecar(
                candidate_dir, proposer_run, shared_across=len(staged),
            )

        reproducibility_manifest = {
            **state.run_repro_manifest,
            "epoch": epoch_idx,
            "candidate_name": sc.name,
            "parent_name": parent_name,
            "captured_at": import_time(),
        }
        (candidate_dir / "reproducibility.json").write_text(
            json.dumps(reproducibility_manifest, indent=2)
        )

        valid.append((sc, candidate_dir, proposal_notes))
        print(f"    {sc.name}: ok")

    if not valid:
        print(f"\n  EPOCH {epoch_idx} RESULT: all candidates failed validation")
        _clear_proposal_checkpoint(state, epoch_idx)
        return True

    # Smoke-gate: runtime check on each shape-valid candidate. Catches
    # hook output shapes that the CLI's Zod schema rejects, infinite
    # loops in hook callbacks, and any other runtime failure that
    # shape-only ``validate_config`` can't surface. Rejected candidates
    # are persisted with ``validation_error.json`` + a
    # ``validation_failed=true`` scores.json stub so the next proposer
    # reads their failure the same way it reads shape-validation
    # failures.
    print(f"\n  [2.5/4] Smoke-testing {len(valid)} candidate(s)...")
    survived: list[tuple[StagedCandidate, Path, Optional[dict[str, Any]]]] = []
    smoke_results = _run_smokes_parallel(
        candidates=valid,
        target=target,
        bench=bench,
        model=args.model,
    )
    for sc, cdir, notes, sr in smoke_results:
        if sr.ok:
            survived.append((sc, cdir, notes))
            print(f"    {sc.name}: smoke passed ({sr.duration_s:.1f}s)")
        else:
            print(f"    {sc.name}: SMOKE FAILED ({(sr.error or '')[:120]})")
    valid = survived

    if not valid:
        print(f"\n  EPOCH {epoch_idx} RESULT: all candidates failed smoke")
        _clear_proposal_checkpoint(state, epoch_idx)
        return True

    # Copy the proposer trace into the first valid candidate's dir; siblings
    # reference it via `proposer_trace_ref` in their epoch_meta.
    first_sc, first_dir, _ = valid[0]
    primary_trace_path = first_dir / "proposer_trace.jsonl"
    if staging_trace.exists():
        shutil.copy2(staging_trace, primary_trace_path)
    primary_trace_ref = rel_to_workspace(primary_trace_path)
    sibling_names = [sc.name for sc, _, _ in valid]

    if resume_checkpoint is not None:
        for _, candidate_dir, _ in valid:
            _clear_eval_artifacts(candidate_dir)

    print(f"\n  [3/4] Evaluating {len(valid)} candidate(s) on batch...")
    eval_inputs = [(sc.name, cdir) for sc, cdir, _ in valid]
    all_scores = _run_evals_parallel(
        candidate_dirs=eval_inputs,
        target=target,
        args=args,
        batch_tasks=batch_tasks,
        experience_dir=state.experience_dir,
    )

    # Record each candidate's result (search-side only for now; holdout runs
    # on the best-on-search at the end).
    results: list[tuple[StagedCandidate, Path, Optional[dict[str, Any]], Optional[dict[str, Any]]]] = []
    for (sc, candidate_dir, proposal_notes), scores in zip(valid, all_scores):
        epoch_meta = _build_epoch_meta(
            epoch_idx=epoch_idx,
            candidate_name=sc.name,
            parent_name=parent_name,
            batch_tasks_csv=batch_tasks,
            batch_tasks_list=batch_tasks_list,
            args=args,
            proposer_trace_ref=primary_trace_ref,
            proposal_notes=proposal_notes,
            siblings=[n for n in sibling_names if n != sc.name],
        )

        if scores:
            search_reward = primary_reward(scores)
            cost = _score_cost(scores)
            epoch_meta["status"] = "search_passed"
            epoch_meta["search"] = {
                "reward": reward_or_none(scores),
                "pass_rate": scores.get("pass_rate"),
                "n_passed": scores.get("n_passed"),
                "n_tasks": scores.get("n_tasks"),
                "cost_usd": scores.get("total_cost_usd"),
            }
            epoch_meta["search_delta"] = _compute_score_delta(scores, parent_scores)

            state.history.append({
                "name": sc.name,
                "reward": search_reward,
                "pass_rate": scores["pass_rate"],
                "n_passed": scores["n_passed"],
                "n_tasks": scores["n_tasks"],
                "cost_usd": cost,
                "total_cost_with_proposer_usd": _combined_score_cost(scores),
                "timestamp": import_time(),
            })
            print(
                f"    {sc.name}: search={search_reward:.1%}  cost={_fmt_cost(cost)}"
            )
        else:
            epoch_meta["status"] = "search_failed"
            print(f"    {sc.name}: FAILED (no scores)")

        results.append((sc, candidate_dir, proposal_notes, scores))
        state.iterations_since_skill_evolve.append(sc.name)

    state.write_history()

    accept_on_holdout = getattr(state.args, "accept_on_holdout", False)

    # If every sibling's search eval failed, write failure meta and exit.
    any_search_ok = any(scores is not None for _, _, _, scores in results)
    if not any_search_ok:
        for sc, candidate_dir, _, _ in results:
            epoch_meta = _build_epoch_meta(
                epoch_idx=epoch_idx,
                candidate_name=sc.name,
                parent_name=parent_name,
                batch_tasks_csv=batch_tasks,
                batch_tasks_list=batch_tasks_list,
                args=args,
                proposer_trace_ref=primary_trace_ref,
                proposal_notes=None,
                siblings=[n for n in sibling_names if n != sc.name],
            )
            epoch_meta["status"] = "search_failed"
            _write_epoch_meta(candidate_dir, epoch_meta)
        print(f"\n  EPOCH {epoch_idx} RESULT: FAILED (no valid scores)")
        _clear_proposal_checkpoint(state, epoch_idx)
        return True

    # --- Fan out holdout evals on every search-passed sibling ---------------
    # Running holdout on ONLY the best-on-search candidate defeats the
    # overfitting-prevention purpose of --accept-on-holdout at k>1: a sibling
    # that's worse on search but better on holdout can never be observed.
    # We pay k× holdout cost (same wall time — evals run in parallel) to get
    # a full holdout row for every sibling so the acceptance gate can pick
    # the real champion.
    holdout_scores_by_name: dict[str, Optional[dict[str, Any]]] = {}
    if state.holdout_dir and args.holdout_benchmark:
        passed_candidates = [
            (sc.name, candidate_dir)
            for (sc, candidate_dir, _, scores) in results
            if scores is not None
        ]
        if resume_checkpoint is not None:
            for cand_name, _ in passed_candidates:
                _clear_eval_artifacts(state.holdout_dir / f"{cand_name}_holdout")
        holdout_inputs = [
            (f"{name}_holdout", candidate_dir)
            for (name, candidate_dir) in passed_candidates
        ]
        print(f"\n  [4/4] Holdout on {len(holdout_inputs)} sibling(s) in parallel...")
        holdout_scores_list = _run_evals_parallel(
            candidate_dirs=holdout_inputs,
            target=target,
            args=args,
            batch_tasks=None,
            experience_dir=state.holdout_dir,
            benchmark_path=args.holdout_benchmark,
            split=getattr(args, "holdout_split", None),
        )
        for (cand_name, _), ho in zip(passed_candidates, holdout_scores_list):
            holdout_scores_by_name[cand_name] = ho
            if ho is None:
                print(f"    {cand_name}: holdout FAILED")
                continue
            ho_reward = primary_reward(ho)
            ho_cost = _score_cost(ho)
            print(f"    {cand_name}: holdout={ho_reward:.1%}  cost={_fmt_cost(ho_cost)}")
            # Amend the sibling's history row with its own holdout numbers.
            for row in state.history:
                if row.get("name") == cand_name:
                    row["holdout_reward"] = ho_reward
                    row["holdout_cost"] = ho_cost
                    row["holdout_n_passed"] = ho.get("n_passed", 0)
                    row["holdout_n_tasks"] = ho.get("n_tasks", 0)
                    row["holdout_pass_rate"] = ho.get("pass_rate", 0)
                    break

    # --- Pick the winner --------------------------------------------------
    # Default gate: best-on-search. --accept-on-holdout: best-on-holdout
    # across siblings (with holdout actually run on every sibling now).
    def _search_reward_of(scores: Optional[dict[str, Any]]) -> float:
        return primary_reward(scores) if scores else float("-inf")

    def _holdout_reward_of(cand_name: str) -> float:
        ho = holdout_scores_by_name.get(cand_name)
        return primary_reward(ho) if ho else float("-inf")

    if accept_on_holdout and holdout_scores_by_name:
        winner_idx = max(
            range(len(results)),
            key=lambda i: _holdout_reward_of(results[i][0].name),
        )
    else:
        if accept_on_holdout and not holdout_scores_by_name:
            logger.warning(
                "--accept-on-holdout was set but no holdout evals produced "
                "a reward; falling back to best-on-search for champion."
            )
        winner_idx = max(range(len(results)), key=lambda i: _search_reward_of(results[i][3]))

    best_sc, best_dir, _, best_scores = results[winner_idx]
    best_reward = _search_reward_of(best_scores)
    winner_holdout = holdout_scores_by_name.get(best_sc.name)
    holdout_reward: Optional[float] = (
        primary_reward(winner_holdout) if winner_holdout else None
    )

    # --- Write epoch_meta per sibling -------------------------------------
    for (sc, candidate_dir, proposal_notes, scores) in results:
        epoch_meta = _build_epoch_meta(
            epoch_idx=epoch_idx,
            candidate_name=sc.name,
            parent_name=parent_name,
            batch_tasks_csv=batch_tasks,
            batch_tasks_list=batch_tasks_list,
            args=args,
            proposer_trace_ref=primary_trace_ref,
            proposal_notes=proposal_notes,
            siblings=[n for n in sibling_names if n != sc.name],
        )
        if scores:
            epoch_meta["status"] = "search_passed"
            epoch_meta["search"] = {
                "reward": reward_or_none(scores),
                "pass_rate": scores.get("pass_rate"),
                "n_passed": scores.get("n_passed"),
                "n_tasks": scores.get("n_tasks"),
                "cost_usd": scores.get("total_cost_usd"),
            }
            epoch_meta["search_delta"] = _compute_score_delta(scores, parent_scores)
        else:
            epoch_meta["status"] = "search_failed"

        this_holdout = holdout_scores_by_name.get(sc.name)
        if this_holdout:
            epoch_meta["status"] = "completed"
            epoch_meta["holdout"] = {
                "reward": reward_or_none(this_holdout),
                "pass_rate": this_holdout.get("pass_rate"),
                "n_passed": this_holdout.get("n_passed"),
                "n_tasks": this_holdout.get("n_tasks"),
                "cost_usd": this_holdout.get("total_cost_usd"),
            }
            epoch_meta["holdout_delta"] = _compute_score_delta(
                this_holdout, parent_holdout_scores,
            )
        if sc.name == best_sc.name:
            epoch_meta["holdout_is_winner"] = this_holdout is not None

        _write_epoch_meta(candidate_dir, epoch_meta)

    # --- Acceptance gate + best-rate update ---------------------------------
    if accept_on_holdout and holdout_reward is not None:
        accept_reward = holdout_reward
        gate_label = "holdout"
    else:
        if accept_on_holdout and holdout_reward is None:
            logger.warning(
                "--accept-on-holdout was set but winner has no holdout "
                "reward; falling back to search reward."
            )
        accept_reward = best_reward
        gate_label = "search"

    is_best = accept_reward > state.best_rate
    if is_best:
        state.best_rate = accept_reward
    arrow = " *** NEW BEST ***" if is_best else ""

    print(f"\n  {'─'*50}")
    summary_line = f"  EPOCH {epoch_idx} WINNER: {best_sc.name}  search={best_reward:.1%}"
    if holdout_reward is not None:
        summary_line += f"  holdout={holdout_reward:.1%}"
    summary_line += arrow
    print(summary_line)
    print(f"  Best so far ({gate_label}): {state.best_rate:.1%}")

    state.write_history()
    write_epoch_report(state, epoch_idx, [sc.name for sc, _, _, scores in results if scores is not None])

    display_values = [_display_reward_from_row(h, accept_on_holdout) for h in state.history]
    rates = " -> ".join(f"{v:.0%}" for v in display_values[-8:])
    spark = _spark(display_values)
    print(f"  History ({gate_label}): {rates}  {spark}")
    print(f"  {'─'*50}")
    _clear_proposal_checkpoint(state, epoch_idx)
    return True
