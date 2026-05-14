"""Run one task from a benchmark. The narrow "does this work?" entrypoint.

Unlike `eval_runner` (which sweeps a whole benchmark and persists to the
experience store), this runs exactly one task, prints the result, and
leaves artifacts in a predictable workspace for inspection.

Public surface: `run(...) -> TaskResult`. Invoked by `meta-agent run-task`.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from meta_agent.core.benchmark import load_benchmark
from meta_agent.utils.logging import get_logger
from meta_agent.core.targets import detect_target
from meta_agent.task_runner import TaskResult, run_command, run_task_with_runtime

logger = get_logger("one-task")


def run(
    *,
    benchmark_path: str,
    config_path: str,
    task_name: Optional[str] = None,
    split: Optional[str] = None,
    model: str = "gpt-5.4",
    keep_workspace: bool = False,
) -> TaskResult:
    """Run one task from `benchmark_path` against `config_path`.

    `benchmark_path` may include a ``:split`` suffix or you can pass `split`
    separately. Harness target is detected from `config_path`.
    """
    bench = load_benchmark(benchmark_path, split=split)

    if not bench.tasks:
        raise ValueError(f"Benchmark {bench.name!r} has no tasks")

    if task_name is None:
        task = bench.tasks[0]
    else:
        matched = [t for t in bench.tasks if t.name == task_name]
        if not matched:
            known = [t.name for t in bench.tasks]
            raise ValueError(f"Task {task_name!r} not found. Known tasks: {known}")
        task = matched[0]

    tmp_root = Path(tempfile.mkdtemp(prefix=f"task_{task.name}_"))
    work_dir = tmp_root / task.name
    shutil.copytree(task.workspace, str(work_dir))

    if task.setup:
        run_command(task.setup, cwd=work_dir, timeout=task.timeout)

    runtime = detect_target(Path(config_path)).default_runtime
    logger.info(f"{bench.name}/{task.name} (runtime={runtime}, model={model})")
    logger.info(f"workspace: {work_dir}")

    result = asyncio.run(run_task_with_runtime(
        task=task,
        config_dir=config_path,
        model=model,
        work_dir=work_dir,
        runtime=runtime,
    ))

    status = "PASS" if result.passed else "FAIL"
    cost = f"${result.cost_usd:.4f}" if result.cost_usd else "N/A"
    turns = str(result.num_turns) if result.num_turns is not None else "N/A"
    print()
    logger.info(f"{status}  reward={result.reward:.2f}  cost={cost}  turns={turns}")
    logger.info(f"artifacts: {work_dir}")
    if not result.passed and result.verify_output:
        logger.warning(f"verifier output:\n{result.verify_output.rstrip()}")

    if result.passed and not keep_workspace:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return result
