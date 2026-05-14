"""Optional final-test evaluation after search completes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meta_agent.core import experience
from meta_agent.core.benchmark import load_benchmark
from meta_agent.utils.logging import get_logger
from meta_agent.loop.state import LoopState, _build_frontier

logger = get_logger("loop")


def _selected_final_candidates(state: LoopState) -> list[str]:
    args = state.args
    include_baseline = bool(getattr(args, "final_test_baseline", False))
    include_frontier = bool(getattr(args, "final_test_frontier", False))
    include_current_best = bool(getattr(args, "final_test_current_best", False))
    if not (include_baseline or include_frontier or include_current_best):
        # Match the Stanford text-classification reference default: baselines
        # plus validation frontier are tested after search. Also include the
        # accepted current best even if it is cost-dominated on the frontier.
        include_baseline = True
        include_frontier = True
        include_current_best = True

    names: list[str] = []
    seen: set[str] = set()

    def add(name: Any) -> None:
        if not isinstance(name, str) or not name or name in seen:
            return
        seen.add(name)
        names.append(name)

    if include_baseline:
        add("baseline")

    frontier = _build_frontier(
        state.history,
        run_name=state.run_name,
        accept_on_holdout=bool(getattr(args, "accept_on_holdout", False)),
        include_holdout=True,
    )
    if include_frontier:
        for row in frontier.get("pareto", []):
            add(row.get("name"))
    if include_current_best:
        add(frontier.get("current_best"))
    return names


def run_final_eval(state: LoopState) -> dict[str, Any] | None:
    """Evaluate selected candidates on a final held-out benchmark."""
    benchmark_ref = getattr(state.args, "final_test_benchmark", None)
    if not benchmark_ref:
        return None

    final_split = getattr(state.args, "final_test_split", None)
    final_bench = load_benchmark(benchmark_ref, split=final_split)
    final_dir = experience.candidates_dir(f"{state.run_name}__{final_bench.name}")
    final_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_final_candidates(state)
    logger.info(
        f"final-test: evaluating {len(selected)} candidate(s) on {final_bench.name}"
    )

    from meta_agent.loop.epoch import run_evaluation

    rows: list[dict[str, Any]] = []
    for candidate_name in selected:
        candidate_dir = state.experience_dir / candidate_name
        row: dict[str, Any] = {
            "candidate": candidate_name,
            "candidate_path": str(candidate_dir),
            "final_name": f"{candidate_name}_final_test",
            "ok": False,
            "scores": None,
            "error": None,
        }
        if not candidate_dir.exists():
            row["error"] = "candidate directory missing"
            rows.append(row)
            continue

        config_path = (
            candidate_dir
            if state.bench_target.is_file_based
            else candidate_dir / state.bench_target.module_filename
        )
        scores = run_evaluation(
            config_path=config_path,
            name=row["final_name"],
            model=state.args.model,
            benchmark_path=benchmark_ref,
            split=final_split,
            fast=False,
            tasks=None,
            concurrency=state.args.concurrency,
            experience_dir=final_dir,
        )
        row["ok"] = scores is not None
        row["scores"] = scores
        rows.append(row)

    summary = {
        "run_name": state.run_name,
        "benchmark": final_bench.name,
        "benchmark_ref": benchmark_ref,
        "split": final_split,
        "experience_dir": str(final_dir),
        "selected": selected,
        "results": rows,
    }
    out_path = state.history_path.parent / "final_test_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"final-test: wrote {out_path}")
    return summary
