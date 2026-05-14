"""One-off baseline-trajectory cache builder for Stage-2 reward signal.

Produces a JSONL pool keyed by ``task_id`` with one vanilla-actor rollout
per search task. The Stage-2 tau3 adapter loads this pool at the start of
every evaluation (see ``benchmarks/tau3/stage2.py::load_baseline_pool``)
and uses each baseline trajectory as the "B" side of the pairwise reward
pair when judging the proposer's candidate.

Why a frozen cache? The baseline is *part of the reward definition* —
randomness across re-runs would smear the signal. The plan (§6) specs
"same baseline trajectory → same judge comparison target". Spending $10
once for 35 deterministic rollouts is strictly better than regenerating
every epoch.

Output JSONL schema (one record per task)::

    {
      "task_id": "12",
      "domain": "airline",
      "actor_model": "claude-haiku-4-5",
      "actor_config": "harnesses/claude_vanilla",
      "gold_reward": 1.0 | 0.0,
      "conversation": [Message.model_dump() for m in tau2_trajectory],
      "num_messages": 24,
      "duration_s": 137.2,
      "cost_usd": 0.081,
      "generated_at": "2026-04-22T04:15:00+00:00"
    }

CLI::

    python -m benchmarks.tau3.cache_baseline_trajectories \\
        --benchmark benchmarks/tau3/benchmark.yaml:search-judge-v1 \\
        --config harnesses/claude_vanilla \\
        --out experience/.cache/tau3_baseline_cache_v1.jsonl \\
        --model claude-haiku-4-5 \\
        --concurrency 20

Runs locally against ``./experience/...`` or on Modal via
``modal_runner.py::cache_baseline_tau3`` (writes to the mounted
``meta-agent-experience`` volume).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


async def _cache_one_task(
    *,
    sem: asyncio.Semaphore,
    domain: str,
    task_id: str,
    config_path: str,
    model: str,
    user_model: Optional[str],
    timeout_s: int,
) -> Optional[dict[str, Any]]:
    """Run one rollout; return a JSONL record (or None on irrecoverable error)."""
    from benchmarks.tau3 import sdk_adapter

    async with sem:
        try:
            r = await asyncio.wait_for(
                sdk_adapter.run_tau_task_sdk(
                    domain=domain,
                    task_id=task_id,
                    config_path=config_path,
                    model=model,
                    user_model=user_model,
                ),
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — log and move on
            print(
                f"  [cache] ERROR task={domain}_{task_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

    return {
        "task_id": str(task_id),
        "domain": domain,
        "actor_model": model,
        "actor_config": config_path,
        "gold_reward": float(r.gold_reward),
        "passed": bool(r.passed),
        "conversation": list(r.tau2_conversation),
        "num_messages": len(r.tau2_conversation),
        "num_turns": r.num_turns,
        "duration_s": r.duration_s,
        "cost_usd": r.cost_usd,
        "session_id": r.session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def cache_baseline(
    *,
    task_list: list[tuple[str, str]],  # (domain, task_id) pairs
    config_path: str,
    model: str,
    out_path: Path,
    user_model: Optional[str] = None,
    concurrency: int = 20,
    timeout_s: int = 300,
) -> dict[str, int]:
    """Run the baseline actor on every task in ``task_list``; stream to JSONL.

    Returns a stats dict ``{"n_rollouts", "n_passed", "n_failed", "n_errors"}``.
    Rollouts stream to disk as they complete so partial progress survives
    a crash. Idempotent callers pass ``--resume`` / filter task_list
    externally; this function does not dedupe against existing output.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"n_rollouts": 0, "n_passed": 0, "n_failed": 0, "n_errors": 0}
    n_total = len(task_list)
    start = time.time()

    print(
        f"  [cache] Generating baseline cache: {n_total} tasks, "
        f"concurrency={concurrency}, model={model}, out={out_path}",
        flush=True,
    )

    with out_path.open("w") as out_f:
        async def _run_and_write(domain: str, task_id: str, idx: int) -> None:
            rec = await _cache_one_task(
                sem=sem,
                domain=domain,
                task_id=task_id,
                config_path=config_path,
                model=model,
                user_model=user_model,
                timeout_s=timeout_s,
            )
            async with lock:
                stats["n_rollouts"] += 1
                if rec is None:
                    stats["n_errors"] += 1
                    return
                if rec["passed"]:
                    stats["n_passed"] += 1
                else:
                    stats["n_failed"] += 1
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                mark = "PASS" if rec["passed"] else "FAIL"
                print(
                    f"  [cache] [{stats['n_rollouts']:>3}/{n_total}] {mark}  "
                    f"{domain}_{task_id:<5} turns={rec['num_turns']:<3} "
                    f"cost=${rec.get('cost_usd') or 0:.3f}  "
                    f"{rec['duration_s']:.0f}s",
                    flush=True,
                )

        await asyncio.gather(*[
            _run_and_write(d, tid, i)
            for i, (d, tid) in enumerate(task_list)
        ])

    elapsed = time.time() - start
    print(
        f"  [cache] Done: {stats['n_rollouts']} rollouts "
        f"({stats['n_passed']} pass / {stats['n_failed']} fail / "
        f"{stats['n_errors']} error) in {elapsed:.0f}s",
        flush=True,
    )
    return stats


def _load_task_list(
    benchmark_ref: str,
    task_filter: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Resolve a benchmark.yaml:split ref into (domain, task_id) pairs."""
    from meta_agent.core.benchmark import load_benchmark

    from benchmarks.tau3.adapter import parse_backend

    if ":" in benchmark_ref:
        path, split = benchmark_ref.split(":", 1)
    else:
        path, split = benchmark_ref, None

    bench = load_benchmark(path, split=split)
    backend = parse_backend(bench)
    if not backend.task_ids:
        raise ValueError(
            f"Benchmark {benchmark_ref!r} has no task_ids; cache requires "
            "an explicit task list."
        )

    domains = backend.domains or ["airline"]
    domain = domains[0]
    if len(domains) > 1:
        print(
            f"  [cache] WARN multiple domains in backend ({domains}); "
            f"caching against {domain!r} only",
            flush=True,
        )

    tasks = [(domain, tid) for tid in backend.task_ids]
    if task_filter:
        wanted = set(task_filter)
        tasks = [t for t in tasks if t[1] in wanted]
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a frozen baseline-trajectory cache for Stage-2.",
    )
    parser.add_argument(
        "--benchmark", required=True,
        help="benchmark.yaml ref (path or path:split) whose task_ids define the cache.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Baseline actor harness dir or harness.py (e.g. harnesses/claude_vanilla).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSONL path (e.g. experience/.cache/tau3_baseline_cache_v1.jsonl).",
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5",
        help="Actor model (default: claude-haiku-4-5).",
    )
    parser.add_argument(
        "--user-model", default="gpt-4.1",
        help=(
            "User-simulator model. Default: gpt-4.1 — MUST match the user_model "
            "used to build the Stage-1 judge training pool "
            "(`benchmarks/tau3_trajectory_judge/build_pool.py --user-model gpt-4.1`) "
            "so cached baseline trajectories are in-distribution with what the "
            "frozen judge was trained on. Override only if you regenerate the "
            "Stage-1 pool with a different user simulator."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-task timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Optional comma-separated subset of task_ids to cache.",
    )
    args = parser.parse_args()

    task_filter = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else None
    )
    task_list = _load_task_list(args.benchmark, task_filter=task_filter)
    if not task_list:
        print("  [cache] No tasks resolved from --benchmark; aborting.", file=sys.stderr)
        return 2

    stats = asyncio.run(cache_baseline(
        task_list=task_list,
        config_path=args.config,
        model=args.model,
        out_path=Path(args.out),
        user_model=args.user_model,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
    ))
    return 0 if stats["n_rollouts"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
