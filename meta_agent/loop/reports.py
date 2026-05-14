"""Deterministic short reports for completed evolution epochs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meta_agent.loop.state import LoopState


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _fmt_pct(value: Any) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "n/a"


def _fmt_delta(value: float) -> str:
    return f"{value:+.1%}"


def _bucket_rates(category_scores: dict[str, Any]) -> dict[str, float]:
    summary = category_scores.get("_summary")
    buckets = summary.get("buckets") if isinstance(summary, dict) else None
    if isinstance(buckets, dict):
        return {
            str(category): float(rate)
            for category, rate in buckets.items()
            if isinstance(rate, (int, float))
        }
    rates: dict[str, float] = {}
    for category, payload in category_scores.items():
        if str(category).startswith("_") or not isinstance(payload, dict):
            continue
        rate = payload.get("pass_rate")
        if isinstance(rate, (int, float)):
            rates[str(category)] = float(rate)
    return rates


def _candidate_section(
    state: LoopState,
    name: str,
    baseline_buckets: dict[str, float],
) -> list[str]:
    cdir = state.experience_dir / name
    scores = _load_json(cdir / "scores.json")
    category_scores = _load_json(cdir / "category_scores.json")
    notes = _load_json(cdir / "proposal_notes.json")
    manifest = _load_json(cdir / "proposal_manifest.json")
    epoch_meta = _load_json(cdir / "epoch_meta.json")

    lines = [
        f"## {name}",
        "",
        f"- Candidate: `{cdir}`",
        f"- Search reward: `{_fmt_pct(scores.get('mean_reward'))}`",
        f"- Pass rate: `{_fmt_pct(scores.get('pass_rate'))}` ({scores.get('n_passed', 'n/a')}/{scores.get('n_tasks', 'n/a')})",
    ]
    holdout = epoch_meta.get("holdout") if isinstance(epoch_meta.get("holdout"), dict) else {}
    if holdout:
        lines.append(
            f"- Holdout reward: `{_fmt_pct(holdout.get('reward'))}` "
            f"({holdout.get('n_passed', 'n/a')}/{holdout.get('n_tasks', 'n/a')})"
        )
    if notes:
        lines.append(f"- Hypothesis: {notes.get('hypothesis') or notes.get('rationale') or notes}")
        if notes.get("lever"):
            lines.append(f"- Lever: `{notes['lever']}`")
    elif manifest:
        lines.append(f"- Hypothesis: {manifest.get('hypothesis') or manifest}")
        if manifest.get("lever") or manifest.get("axis"):
            lines.append(f"- Lever/axis: `{manifest.get('lever') or manifest.get('axis')}`")

    if category_scores:
        shown = []
        candidate_buckets = _bucket_rates(category_scores)
        for category, rate in sorted(candidate_buckets.items()):
            delta = ""
            if category in baseline_buckets:
                delta = f" ({_fmt_delta(rate - baseline_buckets[category])} vs baseline)"
            shown.append(f"{category}={_fmt_pct(rate)}{delta}")
            if len(shown) >= 8:
                break
        if shown:
            lines.append("- Category scores: " + ", ".join(shown))

    failed = scores.get("tasks_failed") if isinstance(scores.get("tasks_failed"), list) else []
    if failed:
        lines.append("- Example failed traces:")
        for task in failed[:5]:
            lines.append(f"  - `per_task/{task}_trace.jsonl`")
    lines.append("")
    return lines


def write_epoch_report(state: LoopState, epoch_idx: int, candidate_names: list[str]) -> Path:
    """Write a concise report for one epoch's evaluated candidates."""
    reports_dir = state.history_path.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"evo_{epoch_idx:03d}.md"
    lines = [
        f"# Epoch {epoch_idx:03d} Report",
        "",
        f"- Run: `{state.run_name}`",
        f"- Candidates: {', '.join(f'`{name}`' for name in candidate_names)}",
        "",
    ]
    baseline_buckets = _bucket_rates(_load_json(state.experience_dir / "baseline" / "category_scores.json"))
    for name in candidate_names:
        lines.extend(_candidate_section(state, name, baseline_buckets))
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path
