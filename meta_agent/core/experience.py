"""Experience store — single owner of the candidate directory layout.

Layout (one per candidate):

    experience/<bench>/candidates/<name>/
        scores.json                     # mean_reward, pass_rate, n_tasks, cost
        summary.md
        per_task/
            {task}.json                 # reward, passed, cost_usd, num_turns
            {task}_trace.jsonl          # agent execution trace
            {task}_agent_result.json    # runtime metadata
            {task}_judge_feedback.md    # judge reasoning (if available)
        [harness files: AGENTS.md, .codex/, harness.py, ...]

Every read/write of this layout goes through this module. Adding new fields
to scores or new sidecar artifacts happens here, not in five places.
"""
from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

from meta_agent.core.benchmark import primary_reward
from meta_agent.core.paths import get_benchmark_candidates_dir


@dataclass(frozen=True)
class Candidate:
    """One candidate harness on disk + its parsed scores.json."""

    dir: Path
    scores: dict[str, Any]

    @property
    def name(self) -> str:
        value = self.scores.get("name")
        return value if isinstance(value, str) else self.dir.name

    @property
    def reward(self) -> float:
        return primary_reward(self.scores)

    def per_task(self) -> dict[str, dict[str, Any]]:
        return load_per_task(self.dir)


# --- Layout ---------------------------------------------------------------

def candidates_dir(bench_name: str) -> Path:
    """Canonical candidates directory for one benchmark."""
    return get_benchmark_candidates_dir(bench_name)


# --- Reads ----------------------------------------------------------------

def load_scores(candidate_dir: Path) -> Optional[dict[str, Any]]:
    """Read scores.json from a candidate dir; None on missing or bad JSON."""
    path = candidate_dir / "scores.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def load_per_task(candidate_dir: Path) -> dict[str, dict[str, Any]]:
    """All per-task result files for one candidate, keyed by short_name."""
    per_task_dir = candidate_dir / "per_task"
    if not per_task_dir.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(per_task_dir.glob("*.json")):
        if f.name.endswith("_agent_result.json"):
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        short_name = data.get("short_name", f.stem)
        out[short_name] = data
    return out


def iter_candidates(candidates_root: Path) -> Iterator[Candidate]:
    """Yield every candidate in `candidates_root` that has a scores.json."""
    if not candidates_root.exists():
        return
    for d in sorted(candidates_root.iterdir()):
        if not d.is_dir():
            continue
        scores = load_scores(d)
        if scores is not None:
            yield Candidate(dir=d, scores=scores)


def list_candidates(candidates_root: Path) -> List[Candidate]:
    """Materialized list of every candidate in `candidates_root`."""
    return list(iter_candidates(candidates_root))


def best_candidate(candidates_root: Path) -> Optional[Candidate]:
    """Highest-reward candidate, or None if the store is empty."""
    return max(
        iter_candidates(candidates_root),
        key=lambda c: c.reward,
        default=None,
    )


def has_candidate(candidates_root: Path, name: str) -> bool:
    return (candidates_root / name / "scores.json").exists()


# --- Writes ---------------------------------------------------------------

def write_candidate(
    *,
    candidates_root: Path,
    name: str,
    config_path: str,
    model: str,
    results: List[Any],
) -> Path:
    """Persist a full candidate: config files + per-task artifacts + scores + summary.

    `results` is a list of `TaskResult` (kept as `Any` to avoid an import
    cycle with `task_runner`). Returns the candidate directory.
    """
    candidate_dir = candidates_root / name
    per_task_dir = candidate_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    _copy_config_into(Path(config_path), candidate_dir)

    trials = [_persist_trial(r, per_task_dir) for r in results]

    scores = _build_scores(name=name, config_path=config_path, model=model, trials=trials)

    # Fold proposer cost into scores if the candidate carries a sidecar. This
    # makes `meta-agent list` and `meta-agent pareto` reflect the true cost
    # of producing + evaluating the candidate, not just the eval piece.
    proposer_cost_payload = _load_proposer_cost(candidate_dir)
    if proposer_cost_payload is not None:
        scores["proposer_cost_usd"] = proposer_cost_payload.get("cost_usd")
        scores["proposer_num_turns"] = proposer_cost_payload.get("num_turns")
        scores["proposer_cli"] = proposer_cost_payload.get("cli")
        scores["proposer_model"] = proposer_cost_payload.get("model")
        eval_cost = scores.get("total_cost_usd") or 0.0
        prop_cost = proposer_cost_payload.get("cost_usd") or 0.0
        scores["total_cost_with_proposer_usd"] = eval_cost + prop_cost

    (candidate_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    (candidate_dir / "summary.md").write_text(
        _render_summary(name=name, model=model, config_path=config_path, scores=scores, trials=trials)
    )
    return candidate_dir


def rewrite_summary(candidate_dir: Path) -> None:
    """Rewrite summary.md from current scores.json after adapter post-processing."""
    scores = load_scores(candidate_dir)
    if scores is None:
        return
    trials = list(load_per_task(candidate_dir).values())
    name = str(scores.get("name") or candidate_dir.name)
    model = str(scores.get("model") or "")
    config_path = str(scores.get("config_path") or "")
    (candidate_dir / "summary.md").write_text(
        _render_summary(
            name=name,
            model=model,
            config_path=config_path,
            scores=scores,
            trials=trials,
        )
    )


def _load_proposer_cost(candidate_dir: Path) -> Optional[dict[str, Any]]:
    """Read the `proposer_cost.json` sidecar if present. None if missing/bad."""
    path = candidate_dir / "proposer_cost.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


# --- Internal -------------------------------------------------------------

def _copy_config_into(src: Path, candidate_dir: Path) -> None:
    """Copy a config dir-or-file into the candidate dir; no-op when paths overlap."""
    if src.is_dir():
        if src.resolve() == candidate_dir.resolve():
            return
        for item in src.iterdir():
            dest = candidate_dir / item.name
            if item.is_dir():
                if dest.resolve() != item.resolve():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
            elif dest.resolve() != item.resolve():
                shutil.copy2(item, dest)
        return
    dest = candidate_dir / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)


_TRACE_SIDECARS: tuple[tuple[str, str], ...] = (
    ("trace.jsonl", "_trace.jsonl"),
    ("trace.raw.jsonl", "_trace.raw.jsonl"),
    ("events.jsonl", "_events.jsonl"),
    ("action_sequence.jsonl", "_action_sequence.jsonl"),
    ("tau2_conversation.jsonl", "_tau2_conversation.jsonl"),
    ("result.json", "_agent_result.json"),
    ("judge_feedback.md", "_judge_feedback.md"),
    ("task.json", "_task.json"),
    ("final_response.txt", "_final_response.txt"),
    ("answer.txt", "_answer.txt"),
    ("artifact.html", "_artifact.html"),
    ("screenshot_1.png", "_screenshot_1.png"),
    ("screenshot_2.png", "_screenshot_2.png"),
    ("screenshot_3.png", "_screenshot_3.png"),
)


def _persist_trial(r: Any, per_task_dir: Path) -> dict[str, Any]:
    """Serialize one TaskResult to disk and return the trial dict."""
    trial: dict[str, Any] = {
        "task_name": r.task_name,
        "short_name": r.task_name,
        "reward": r.reward,
        "passed": r.passed,
        "cost_usd": r.cost_usd,
        "num_turns": r.num_turns,
        "duration_ms": r.duration_ms,
        "wall_time_s": r.wall_time_s,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cache_tokens": r.cache_tokens,
        "session_id": r.session_id,
        "trial_dir": r.work_dir,
    }
    (per_task_dir / f"{r.task_name}.json").write_text(json.dumps(trial, indent=2))

    work_dir = Path(r.work_dir)
    for src_name, dst_suffix in _TRACE_SIDECARS:
        src = work_dir / src_name
        if src.exists():
            shutil.copy2(src, per_task_dir / f"{r.task_name}{dst_suffix}")
    return trial


def _build_scores(
    *, name: str, config_path: str, model: str, trials: list[dict[str, Any]],
) -> dict[str, Any]:
    n_tasks = len(trials)
    n_passed = sum(1 for t in trials if t["passed"])
    rewards = [t["reward"] for t in trials if t["reward"] is not None]
    costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
    turns = [t["num_turns"] for t in trials if t["num_turns"] is not None]
    return {
        "name": name,
        "config_path": config_path,
        "model": model,
        "n_tasks": n_tasks,
        "n_passed": n_passed,
        "pass_rate": n_passed / n_tasks if n_tasks else 0.0,
        "mean_reward": statistics.mean(rewards) if rewards else None,
        "mean_cost_usd": statistics.mean(costs) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
        "median_turns": statistics.median(turns) if turns else None,
        "tasks_passed": [t["short_name"] for t in trials if t["passed"]],
        "tasks_failed": [t["short_name"] for t in trials if not t["passed"]],
    }


def _render_summary(
    *,
    name: str,
    model: str,
    config_path: str,
    scores: dict[str, Any],
    trials: list[dict[str, Any]],
) -> str:
    reward = scores.get("mean_reward")
    metric = scores.get("plan_rewardbench_metric") or "mean_reward"
    reward_line = (
        f"**Reward ({metric}):** {reward:.1%}"
        if isinstance(reward, (int, float))
        else "**Reward:** N/A"
    )
    lines = [
        f"# {name}",
        "",
        f"**Model:** {model}",
        f"**Config:** {config_path}",
        reward_line,
        f"**Pass rate:** {scores['n_passed']}/{scores['n_tasks']} ({scores['pass_rate']:.1%})",
        f"**Total cost:** ${scores['total_cost_usd']:.4f}" if scores["total_cost_usd"] else "**Total cost:** N/A",
        f"**Median turns:** {scores['median_turns']}" if scores["median_turns"] else "**Median turns:** N/A",
        "",
        "## Per-task results",
        "",
        "| Task | Result | Cost | Turns |",
        "|------|--------|------|-------|",
    ]
    for t in sorted(trials, key=lambda x: x["short_name"]):
        status = "PASS" if t["passed"] else "FAIL"
        cost = f"${t['cost_usd']:.4f}" if t["cost_usd"] else "N/A"
        turns = str(t["num_turns"]) if t["num_turns"] else "N/A"
        lines.append(f"| {t['short_name']} | {status} | {cost} | {turns} |")
    return "\n".join(lines) + "\n"
