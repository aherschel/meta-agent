"""Sweep one harness over a full benchmark; persist a candidate.

`run(...)` is the in-process entry point used by `meta-agent eval` and the
loop package. It resolves the right `BenchmarkAdapter` for `bench.type`,
dispatches the run, and hands the results to `experience.write_candidate`.

All benchmark-specific logic lives in `benchmarks/<family>/adapter.py`; this
file is a 3-step pipeline.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, List, Optional

from meta_agent.core import adapters, experience
from meta_agent.core.benchmark import load_benchmark
from meta_agent.utils.logging import get_logger
from meta_agent.task_runner import TaskResult

logger = get_logger("eval")


def _select_tasks(
    bench_tasks: List[Any], fast: bool, fast_tasks: List[str], tasks: Optional[str],
) -> tuple[List[Any], Optional[List[str]]]:
    """Pick the subset of tasks for this run, plus the matching task_filter list."""
    if fast:
        return [t for t in bench_tasks if t.name in fast_tasks], list(fast_tasks)
    if tasks:
        names = [n.strip() for n in tasks.split(",")]
        return [t for t in bench_tasks if t.name in set(names)], names
    return list(bench_tasks), None


def _print_summary(*, name: str, scores: dict[str, Any], results: List[TaskResult],
                   elapsed: float, candidate_dir: Path) -> None:
    n_passed = scores["n_passed"]
    n_tasks = scores["n_tasks"]
    pass_rate = scores["pass_rate"]
    reward = scores.get("mean_reward")
    total_cost = scores.get("total_cost_usd") or 0

    print(f"\n{'='*60}")
    if isinstance(reward, (int, float)):
        print(f"  {name}  —  reward={reward:.1%}  pass={n_passed}/{n_tasks} ({pass_rate:.0%})")
    else:
        print(f"  {name}  —  {n_passed}/{n_tasks} ({pass_rate:.0%})")
    print(f"{'='*60}\n")
    for r in sorted(results, key=lambda x: x.task_name):
        mark = "PASS" if r.passed else "FAIL"
        cost = f"${r.cost_usd:.3f}" if r.cost_usd else "  N/A"
        turns = f"{r.num_turns:>3}" if r.num_turns else "N/A"
        dur = f"{(r.duration_ms / 1000):.0f}s" if r.duration_ms else "N/A"
        print(f"  {mark}  {r.task_name:<30}  {turns} turns  {cost}  {dur}")
    print()
    cost_str = f"${total_cost:.4f}" if total_cost else "N/A"
    turns_str = str(scores.get("median_turns", "N/A"))
    print(f"  Total cost: {cost_str}  |  Median turns: {turns_str}  |  Wall time: {elapsed:.0f}s")
    print(f"  Saved to: {candidate_dir}\n")


def run(
    *,
    benchmark_path: str,
    config_path: str,
    name: str,
    split: Optional[str] = None,
    model: str = "gpt-5.4",
    fast: bool = False,
    tasks: Optional[str] = None,
    concurrency: int = 4,
    keep_workspaces: bool = False,
    keep_failed: bool = False,
    dry_run: bool = False,
    experience_dir: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Run one evaluation; return the parsed scores dict, or None for dry-run.

    `benchmark_path` may include a ``:split`` suffix or you can pass `split`
    separately. Callers (e.g. the loop package) should prefer `run()` over
    spawning a subprocess.
    """
    bench = load_benchmark(benchmark_path, split=split)
    exp_dir = experience_dir or experience.candidates_dir(bench.name)

    selected, task_filter = _select_tasks(bench.tasks, fast, bench.fast_tasks, tasks)

    task_display = task_filter or [t.name for t in selected] or "all"
    logger.info(f"Benchmark: {bench.name} (type={bench.type})")
    logger.info(f"Config: {config_path}")
    logger.info(f"Name: {name}")
    logger.info(f"Model: {model}")
    logger.info(f"Tasks: {task_display}")
    logger.info(f"Concurrency: {concurrency}")

    if dry_run:
        logger.info("Dry run — exiting.")
        return None

    eval_start = time.time()
    logger.info(f"Starting at {time.strftime('%H:%M:%S')}...")
    print()

    adapter = adapters.get(bench.type)
    adapters.assert_target_supported(adapter, config_path)
    results = asyncio.run(adapter.run(
        benchmark=bench,
        config_path=config_path,
        model=model,
        concurrency=concurrency,
        task_filter=task_filter,
        keep_workspaces=keep_workspaces,
        keep_failed=keep_failed,
    ))

    elapsed = time.time() - eval_start
    candidate_dir = experience.write_candidate(
        candidates_root=exp_dir,
        name=name,
        config_path=config_path,
        model=model,
        results=results,
    )

    # Generic sidecar: write per-category scores if the adapter emitted
    # compatible traces. A no-op for benchmarks whose traces lack
    # `category` + `passed` fields — so this is safe for every adapter.
    from meta_agent.task_runner.judge_stats import write_category_scores
    write_category_scores(candidate_dir)

    # Optional adapter-specific post-processing (e.g. RewardBench 2 Ties
    # subset score override, which requires paired ref/tied sample
    # aggregation that can't fit the generic pass_rate shape).
    if adapter.post_process_scores is not None:
        adapter.post_process_scores(candidate_dir)

    import json
    scores = json.loads((candidate_dir / "scores.json").read_text())
    experience.rewrite_summary(candidate_dir)
    _print_summary(
        name=name, scores=scores, results=results,
        elapsed=elapsed, candidate_dir=candidate_dir,
    )
    return scores
