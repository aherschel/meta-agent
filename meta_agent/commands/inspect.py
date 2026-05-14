"""Presentation layer for the experience store.

Functions here are imported by the unified `meta-agent` CLI (`__main__.py`).
The on-disk layout lives in `meta_agent.core.experience`; this module is print-only.

Public surface (`__all__`) is consumed by the unified CLI handlers and tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meta_agent.core import experience

# Re-exported so existing call sites and test patches keep working.
load_scores = experience.load_scores
load_per_task = experience.load_per_task


def _format_row(scores: dict[str, Any]) -> str:
    name = scores.get("name", "?")
    if scores.get("validation_failed"):
        err = scores.get("validation_error") or "validation failed"
        # Truncate long tracebacks to keep the table readable.
        err_short = err.split("\n", 1)[0][:60]
        return f"{name:<25} {'FAILED':<10} (validation error: {err_short})"
    if scores.get("eval_failed"):
        err = scores.get("eval_error") or "eval failed"
        err_short = str(err).split("\n", 1)[0][:60]
        return f"{name:<25} {'FAILED':<10} (eval error: {err_short})"
    n_passed = scores.get("n_passed", 0)
    n_tasks = scores.get("n_tasks", 0)
    reward = scores.get("mean_reward")
    rate = scores.get("pass_rate", 0)
    cost = scores.get("total_cost_usd")
    proposer_cost = scores.get("proposer_cost_usd")
    turns = scores.get("median_turns")
    reward_str = f"{reward:.1%}" if reward is not None else "N/A"
    if cost is not None and proposer_cost is not None:
        cost_str = f"${cost:.4f} (+${proposer_cost:.4f})"
    elif cost is not None:
        cost_str = f"${cost:.4f}"
    else:
        cost_str = "N/A"
    turns_str = str(turns) if turns is not None else "N/A"
    return f"{name:<25} {reward_str:<10} {n_passed}/{n_tasks} ({rate:.0%}){'':<2} {cost_str:<22} {turns_str:<8}"


def _print_table(candidates: list[experience.Candidate]) -> None:
    print(f"{'Name':<25} {'Reward':<10} {'Pass Rate':<12} {'Cost (+proposer)':<22} {'Turns':<8}")
    print("-" * 85)
    for c in candidates:
        print(_format_row(c.scores))


def _load_candidate_index(experience_dir: Path) -> dict[str, Any] | None:
    index_path = experience_dir.parent / "candidate_index.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_frontier(experience_dir: Path) -> dict[str, Any] | None:
    frontier_path = experience_dir.parent / "frontier.json"
    if not frontier_path.is_file():
        return None
    try:
        payload = json.loads(frontier_path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _fmt_pct(value: Any) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "N/A"


def _scores_overview(scores: dict[str, Any]) -> str:
    reward = scores.get("mean_reward")
    pass_rate = scores.get("pass_rate")
    n_passed = scores.get("n_passed", 0)
    n_tasks = scores.get("n_tasks", 0)
    return (
        f"Current scores.json: reward={_fmt_pct(reward)}  "
        f"pass={n_passed}/{n_tasks} ({_fmt_pct(pass_rate)})"
    )


def _format_index_row(row: dict[str, Any], scores: dict[str, Any] | None) -> str:
    name = str(row.get("name") or "?")
    accept = row.get("accept_reward")
    search = row.get("search_reward")
    holdout = row.get("holdout_reward")
    pass_rate = row.get("search_pass_rate")
    n_passed = row.get("search_n_passed", 0)
    n_tasks = row.get("search_n_tasks", 0)
    cost = row.get("search_total_cost_with_proposer_usd") or row.get("search_cost_usd")
    gap = row.get("search_holdout_gap")
    turns = scores.get("median_turns") if scores else None
    marker = "*" if row.get("is_current_best") else " "
    cost_str = f"${float(cost):.4f}" if isinstance(cost, (int, float)) else "N/A"
    gap_str = f"{float(gap):+.1%}" if isinstance(gap, (int, float)) else "N/A"
    turns_str = str(turns) if turns is not None else "N/A"
    return (
        f"{marker}{name:<24} {_fmt_pct(accept):<10} {_fmt_pct(search):<10} "
        f"{_fmt_pct(holdout):<10} {n_passed}/{n_tasks} ({_fmt_pct(pass_rate):<6}) "
        f"{gap_str:<8} {cost_str:<12} {turns_str:<8}"
    )


def _print_index_table(
    candidate_index: dict[str, Any], candidates: list[experience.Candidate],
) -> bool:
    rows = candidate_index.get("candidates")
    if not isinstance(rows, list):
        return False

    candidates_by_name = {
        str(candidate.scores.get("name") or candidate.name): candidate
        for candidate in candidates
    }
    print("Ranking by acceptance reward from candidate_index.json.")
    print("* marks the current accepted best.\n")
    print(f"{'Name':<25} {'Accept':<10} {'Search':<10} {'Holdout':<10} {'Pass Rate':<18} {'Gap':<8} {'Cost':<12} {'Turns':<8}")
    print("-" * 106)
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        candidate = candidates_by_name.get(name)
        print(_format_index_row(item, candidate.scores if candidate else None))
        seen.add(name)

    leftovers = [candidate for candidate in candidates if candidate.name not in seen]
    leftovers.sort(
        key=lambda c: (c.reward, -(c.scores.get("total_cost_usd") or 999)),
        reverse=True,
    )
    if leftovers:
        print("\nNot in candidate_index.json:")
        for candidate in leftovers:
            print(_format_row(candidate.scores))
    return True


def list_candidates(experience_dir: Path) -> None:
    """Print all candidates in `experience_dir` ranked by reward."""
    candidates = experience.list_candidates(experience_dir)
    if not candidates:
        if not experience_dir.exists():
            print("No experience store found.")
        else:
            print("No candidates found.")
        return

    candidate_index = _load_candidate_index(experience_dir)
    if candidate_index and _print_index_table(candidate_index, candidates):
        return

    candidates.sort(
        key=lambda c: (c.reward, -(c.scores.get("total_cost_usd") or 999)),
        reverse=True,
    )
    _print_table(candidates)


def show_candidate(experience_dir: Path, name: str) -> None:
    """Print the summary.md (or raw scores.json) for one candidate."""
    candidate_dir = experience_dir / name
    if not candidate_dir.exists():
        print(f"Candidate '{name}' not found.")
        return

    scores = experience.load_scores(candidate_dir)
    if scores:
        print(_scores_overview(scores))
        print()

    summary_path = candidate_dir / "summary.md"
    if summary_path.exists():
        print(summary_path.read_text())
        return

    if scores:
        print(json.dumps(scores, indent=2))


def diff_candidates(experience_dir: Path, name1: str, name2: str) -> None:
    """Show which tasks flipped between two candidates."""
    dir1 = experience_dir / name1
    dir2 = experience_dir / name2

    if not dir1.exists():
        print(f"Candidate '{name1}' not found.")
        return
    if not dir2.exists():
        print(f"Candidate '{name2}' not found.")
        return

    tasks1 = experience.load_per_task(dir1)
    tasks2 = experience.load_per_task(dir2)
    all_tasks = sorted(set(tasks1.keys()) | set(tasks2.keys()))

    if not all_tasks:
        print("No per-task data found for comparison.")
        return

    scores1 = experience.load_scores(dir1) or {}
    scores2 = experience.load_scores(dir2) or {}
    print(
        f"Comparing: {name1} reward={_fmt_pct(scores1.get('mean_reward'))} "
        f"pass={_fmt_pct(scores1.get('pass_rate'))} vs "
        f"{name2} reward={_fmt_pct(scores2.get('mean_reward'))} "
        f"pass={_fmt_pct(scores2.get('pass_rate'))}"
    )
    print()

    flipped_to_pass: list[str] = []
    flipped_to_fail: list[str] = []
    both_pass: list[str] = []
    both_fail: list[str] = []

    for task in all_tasks:
        passed1 = tasks1.get(task, {}).get("passed", False)
        passed2 = tasks2.get(task, {}).get("passed", False)

        if not passed1 and passed2:
            flipped_to_pass.append(task)
        elif passed1 and not passed2:
            flipped_to_fail.append(task)
        elif passed1 and passed2:
            both_pass.append(task)
        else:
            both_fail.append(task)

    if flipped_to_pass:
        print(f"Gained ({len(flipped_to_pass)} tasks — failed in {name1}, passed in {name2}):")
        for t in flipped_to_pass:
            print(f"  + {t}")
    if flipped_to_fail:
        print(f"\nLost ({len(flipped_to_fail)} tasks — passed in {name1}, failed in {name2}):")
        for t in flipped_to_fail:
            print(f"  - {t}")
    if both_pass:
        print(f"\nBoth pass: {len(both_pass)} tasks")
    if both_fail:
        print(f"Both fail: {len(both_fail)} tasks")

    cost1 = scores1.get("total_cost_usd", 0) or 0
    cost2 = scores2.get("total_cost_usd", 0) or 0
    if cost1 > 0 and cost2 > 0:
        print(f"\nCost: ${cost1:.4f} → ${cost2:.4f} ({'+' if cost2 > cost1 else ''}{cost2 - cost1:.4f})")


def pareto_frontier(experience_dir: Path) -> None:
    """Show the Pareto frontier (accuracy vs cost)."""
    candidates = [c for c in experience.iter_candidates(experience_dir)
                  if c.scores.get("total_cost_usd") is not None]

    if not candidates:
        if not experience_dir.exists():
            print("No experience store found.")
        else:
            print("No candidates with cost data found.")
        return

    frontier_payload = _load_frontier(experience_dir)
    rows = frontier_payload.get("pareto") if isinstance(frontier_payload, dict) else None
    if isinstance(rows, list):
        candidates_by_name = {
            str(candidate.scores.get("name") or candidate.name): candidate
            for candidate in candidates
        }
        print(f"Pareto frontier by acceptance reward ({len(rows)}/{len(candidates)} candidates):\n")
        print(f"{'Name':<25} {'Accept':<10} {'Search':<10} {'Holdout':<10} {'Pass Rate':<18} {'Gap':<8} {'Cost':<12} {'Turns':<8}")
        print("-" * 106)
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            candidate = candidates_by_name.get(name)
            print(_format_index_row(row, candidate.scores if candidate else None))
        return

    frontier: list[experience.Candidate] = []
    for c in candidates:
        cost = c.scores.get("total_cost_usd", float("inf"))
        dominated = any(
            other.reward >= c.reward
            and (other.scores.get("total_cost_usd", float("inf"))) <= cost
            and (
                other.reward > c.reward
                or other.scores.get("total_cost_usd", float("inf")) < cost
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(c)

    frontier.sort(key=lambda c: c.reward, reverse=True)

    print(f"Pareto frontier ({len(frontier)}/{len(candidates)} candidates):\n")
    _print_table(frontier)

    if len(frontier) >= 2:
        best = frontier[0]
        cheapest = frontier[-1]
        print(f"\nBest accuracy: {best.name} ({best.reward:.1%}, ${best.scores.get('total_cost_usd', 0):.4f})")
        print(f"Cheapest on frontier: {cheapest.name} ({cheapest.reward:.1%}, ${cheapest.scores.get('total_cost_usd', 0):.4f})")


def candidate_failures(experience_dir: Path, name: str) -> None:
    """List failed tasks for one candidate, with a short trace summary."""
    candidate_dir = experience_dir / name
    if not candidate_dir.exists():
        print(f"Candidate '{name}' not found.")
        return

    tasks = experience.load_per_task(candidate_dir)
    failed = {tname: data for tname, data in tasks.items() if not data.get("passed", False)}

    if not failed:
        print(f"No failures found for '{name}'.")
        return

    print(f"Failed tasks for {name} ({len(failed)}/{len(tasks)}):\n")

    for tname, data in sorted(failed.items()):
        cost = data.get("cost_usd")
        turns = data.get("num_turns")
        cost_str = f"${cost:.4f}" if cost else "N/A"
        turns_str = str(turns) if turns else "N/A"

        trace_path = candidate_dir / "per_task" / f"{tname}_trace.jsonl"
        summary = ""
        if trace_path.exists():
            try:
                lines = trace_path.read_text().strip().split("\n")
                for line in reversed(lines):
                    record = json.loads(line)
                    if record.get("type") == "ResultMessage":
                        result_text = record.get("result", "")
                        if result_text:
                            summary = result_text[:120]
                        break
            except (json.JSONDecodeError, KeyError):
                pass

        print(f"  {tname}  (cost={cost_str}, turns={turns_str})")
        if summary:
            print(f"    Last output: {summary}")
        print()


__all__ = [
    "load_scores",
    "load_per_task",
    "list_candidates",
    "show_candidate",
    "diff_candidates",
    "pareto_frontier",
    "candidate_failures",
]
