"""Generic per-category stats for any benchmark that emits categorized traces.

Reads per-task trace JSONLs from a candidate directory and computes
per-category pass rates. Writes the result as `category_scores.json` next
to `scores.json`. The proposer reads this to decide what to target.

Any adapter whose first JSON line has both a string `category` and a
boolean `passed` field is automatically supported. No benchmark-name
lookup required — adapters declare their categories at trace-write time
and this module aggregates generically. Candidates without compatible
traces get no file (returns None).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compute_category_scores(candidate_dir: Path) -> dict[str, dict[str, Any]]:
    """Walk per_task/*_trace.jsonl, aggregate {category: {passed, total, rate}}.

    Returns an empty dict when no trace files carry both `category` (str)
    and `passed` (bool) fields — the universal "this is a categorized
    judgment-style trace" signal. Agnostic to benchmark name and trace type.
    """
    per_task_dir = candidate_dir / "per_task"
    if not per_task_dir.exists():
        return {}

    totals: dict[str, int] = {}
    passed: dict[str, int] = {}
    for trace_path in per_task_dir.glob("*_trace.jsonl"):
        record = _first_json_line(trace_path)
        if not record:
            continue
        category = record.get("category")
        if not isinstance(category, str):
            continue
        if "passed" not in record:
            continue
        totals[category] = totals.get(category, 0) + 1
        if record.get("passed"):
            passed[category] = passed.get(category, 0) + 1

    return {
        cat: {
            "n_passed": passed.get(cat, 0),
            "n_tasks": totals[cat],
            "pass_rate": (passed.get(cat, 0) / totals[cat]) if totals[cat] else 0.0,
        }
        for cat in sorted(totals)
    }


def write_category_scores(candidate_dir: Path) -> Path | None:
    """Compute category scores and persist as `category_scores.json`.

    Returns the written path, or None when no categorized traces were found.
    Safe to call on any candidate dir — it's a no-op for non-judge benchmarks.
    """
    scores = compute_category_scores(candidate_dir)
    if not scores:
        return None
    out_path = candidate_dir / "category_scores.json"
    out_path.write_text(json.dumps(scores, indent=2))
    return out_path


def _first_json_line(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    except OSError:
        return None
    return None
