"""Benchmark adapter registry.

Each `benchmarks/<family>/adapter.py` calls `register(...)` at import time
with an async `run` function and (optionally) a `task_pool` extractor.
`get(name)` lazy-imports the matching adapter module on first lookup, so
optional deps (tau2, datasets, ...) never fire
unless that benchmark type is actually used.

The built-in `local` adapter — in-process per-task workspace runs — lives
at the bottom of this file because it needs no optional deps and every
researcher uses it.
"""
from __future__ import annotations

import asyncio
import importlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from meta_agent.core.benchmark import Benchmark, Task
from meta_agent.core.targets import detect_target
from meta_agent.task_runner import TaskResult, run_command, run_task_with_runtime


# Async function signature: (*, benchmark, config_path, model, concurrency,
# task_filter, **kwargs) -> list[TaskResult]. **kwargs absorbs adapter-
# specific flags like keep_workspaces (local) without forcing every adapter
# to declare them.
RunFn = Callable[..., Awaitable[List[TaskResult]]]
TaskPoolFn = Callable[[Benchmark], List[str]]
# Post-process hook: given a candidate_dir, mutate category_scores.json
# (or add a sidecar) with benchmark-specific aggregation. Runs after the
# generic per-category pass_rate scores are written. No-op by default.
PostProcessScoresFn = Callable[[Path], None]


@dataclass(frozen=True)
class BenchmarkAdapter:
    """One benchmark family's dispatch contract.

    `supported_targets` is an optional allowlist of harness targets this
    adapter can run with. Empty = all allowed. Adapters that inject MCP
    tools (for example tau3) set this to
    ``frozenset({"claude_agent_sdk"})`` so a codex-only config fails at
    dispatch with a readable error instead of a mysterious crash.
    """

    name: str
    run: RunFn
    task_pool: Optional[TaskPoolFn] = None
    post_process_scores: Optional[PostProcessScoresFn] = None
    supported_targets: frozenset[str] = frozenset()


def assert_target_supported(adapter: "BenchmarkAdapter", config_path: str) -> None:
    """Verify the harness detected at ``config_path`` is allowed by the adapter.

    Raises ``ValueError`` on mismatch with a message that names both sides.
    No-op when the adapter's allowlist is empty.
    """
    if not adapter.supported_targets:
        return
    target = detect_target(Path(config_path))
    if target.name not in adapter.supported_targets:
        raise ValueError(
            f"Adapter {adapter.name!r} requires harness target in "
            f"{sorted(adapter.supported_targets)} but detected {target.name!r} "
            f"at {config_path}"
        )


_REGISTRY: Dict[str, BenchmarkAdapter] = {}

# Benchmark type -> dotted module to import on first `get(name)`.
# Adapters self-register at module import; this map only tells the
# registry where to look.
_LAZY_MODULES: Dict[str, str] = {
    "tau":                   "benchmarks.tau3.adapter",
    "tau3":                  "benchmarks.tau3.adapter",
    "plan_rewardbench":      "benchmarks.plan_rewardbench.adapter",
    "tau3_trajectory_judge": "benchmarks.tau3_trajectory_judge.adapter",
    "tau3_domain_transfer_judge": "benchmarks.tau3_domain_transfer_judge.adapter",
    "terminal_bench":        "benchmarks.terminal_bench.adapter",
}


def register(adapter: BenchmarkAdapter) -> None:
    """Register an adapter. Last writer wins (allows reload during tests)."""
    _REGISTRY[adapter.name] = adapter


def get(name: str) -> BenchmarkAdapter:
    """Look up an adapter by name; lazy-import its module if missing."""
    if name not in _REGISTRY:
        module_path = _LAZY_MODULES.get(name)
        if module_path:
            importlib.import_module(module_path)
    if name not in _REGISTRY:
        known = sorted(set(_REGISTRY) | set(_LAZY_MODULES))
        raise ValueError(
            f"No adapter registered for benchmark type {name!r}. Known: {known}"
        )
    return _REGISTRY[name]


def known_types() -> List[str]:
    """All benchmark types this build can dispatch (registered or lazy)."""
    return sorted(set(_REGISTRY) | set(_LAZY_MODULES))


# --- Built-in: in-process local adapter ---------------------------------

async def _run_local(
    *,
    benchmark: Benchmark,
    config_path: str,
    model: str,
    concurrency: int,
    task_filter: Optional[List[str]] = None,
    keep_workspaces: bool = False,
    keep_failed: bool = False,
    **_unused: Any,
) -> List[TaskResult]:
    selected = (
        [t for t in benchmark.tasks if t.name in set(task_filter)]
        if task_filter else list(benchmark.tasks)
    )
    runtime = detect_target(Path(config_path)).default_runtime
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(task: Task) -> TaskResult:
        async with sem:
            tmp = Path(tempfile.mkdtemp(prefix=f"task_{task.name}_"))
            work_dir = tmp / task.name
            shutil.copytree(task.workspace, str(work_dir))
            if task.setup:
                run_command(task.setup, cwd=work_dir, timeout=task.timeout)
            result = await run_task_with_runtime(
                task=task,
                config_dir=config_path,
                model=model,
                work_dir=work_dir,
                runtime=runtime,
            )
            if not keep_workspaces and not (keep_failed and not result.passed):
                shutil.rmtree(tmp, ignore_errors=True)
            return result

    return list(await asyncio.gather(*[_run_one(t) for t in selected]))


def _local_task_pool(bench: Benchmark) -> List[str]:
    return [t.name for t in bench.tasks]


register(BenchmarkAdapter(name="local", run=_run_local, task_pool=_local_task_pool))
