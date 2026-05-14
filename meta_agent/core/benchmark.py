"""Benchmark + Task pydantic models, YAML loader, and reward helpers.

## Benchmark YAML schema (family + splits)

A benchmark YAML describes a **family** of related splits of one dataset, not
a single configuration of one run. Harness/runtime live on the agent side
(detected from the config dir) and do not appear in benchmark YAMLs.

Non-local example:

    name: tau3-trajectory-judge
    type: tau3_trajectory_judge
    backend:                      # family-level protocol defaults
      pool_path: data/pool.jsonl
      timeout: 180
    splits:
      train:
        task_split: train
      val:
        task_split: val

Local-tasks example (unchanged from today, no splits required):

    name: example
    tasks:
      - name: fix-fibonacci
        instruction: "..."
        workspace: ./workspaces/fibonacci
        verify: ["python", "-c", "..."]

## Resolving a split

`load_benchmark(ref)` returns a `Benchmark` with the split already resolved —
`backend` is the shallow merge of family-level backend ⊕ split backend. Each
adapter's `parse_backend(bench)` keeps validating `bench.backend` against its
own pydantic model, so the adapter contract is unchanged.

`ref` can be:

* `"path/to/family.yaml"` — loads the file. If it defines exactly one split,
  that split is picked. Local-tasks files don't require splits.
* `"path/to/family.yaml:split-name"` — loads the file and picks the split.
* `"path/or/dir/:split-name"` — loads `<dir>/benchmark.yaml` and picks the split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Benchmark",
    "Task",
    "load_benchmark",
    "parse_benchmark_ref",
    "primary_reward",
    "reward_or_none",
]

_REWARD_KEYS: tuple[str, ...] = ("mean_reward", "reward", "pass_rate")


def primary_reward(scores: Dict[str, Any]) -> float:
    """Canonical reward reader. Returns 0.0 when no numeric score is present.

    Checks keys in order: mean_reward → reward → pass_rate. Works for both
    scores.json dicts (mean_reward/pass_rate) and history rows (reward/pass_rate).
    Distinct from the old `x or y` pattern which mis-handled 0.0 as falsy.
    """
    for key in _REWARD_KEYS:
        value = scores.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def reward_or_none(scores: Optional[Dict[str, Any]]) -> Optional[float]:
    """Like `primary_reward` but returns None when no numeric score is present.

    Use this only when `None` is semantically meaningful (e.g. computing deltas
    between candidates where one may not have a score yet).
    """
    if not isinstance(scores, dict):
        return None
    for key in _REWARD_KEYS:
        value = scores.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class Task(BaseModel):
    name: str
    instruction: str
    workspace: str
    verify: Union[str, List[str]]
    setup: Optional[Union[str, List[str]]] = None
    timeout: int = 300


class Benchmark(BaseModel):
    """Resolved benchmark — exactly one (family, split) pair.

    `backend` is already the merged view of family defaults ⊕ split overrides,
    so adapters keep calling `MyBackend.model_validate(bench.backend or {})`
    with no change to their contract.
    """

    model_config = ConfigDict(extra="forbid")

    name: str                                              # full display name (e.g. "tau3-trajectory-judge-train")
    family: str = ""                                       # benchmark family (empty for local)
    split: str = ""                                        # "train-2k" (empty for local)
    type: str = "local"
    description: str = ""
    backend: Dict[str, Any] = Field(default_factory=dict)  # merged family ⊕ split
    tasks: List[Task] = Field(default_factory=list)
    fast_tasks: List[str] = Field(default_factory=list)


# --- Ref parsing + loading ----------------------------------------------------


def parse_benchmark_ref(ref: str) -> Tuple[str, Optional[str]]:
    """Parse a benchmark ref string into ``(path, split_name)``.

    Accepted forms:
      * ``"path/to/file.yaml"``           -> (path, None)
      * ``"path/to/file.yaml:split"``     -> (path, "split")
      * ``"path/to/dir"``                 -> (path/benchmark.yaml, None)
      * ``"path/to/dir:split"``           -> (path/benchmark.yaml, "split")

    A bare directory is resolved to ``<dir>/benchmark.yaml``.
    """
    if ":" in ref:
        # The ref may be a Windows path with a drive letter. We only split on
        # the *last* colon to be friendly to macOS/Linux-style paths. On this
        # codebase we don't support Windows, but a single-colon rule is still
        # safer than splitting on the first colon.
        path_part, _, split_name = ref.rpartition(":")
        split_name = split_name or None
    else:
        path_part, split_name = ref, None

    path = Path(path_part).expanduser()
    if path.is_dir():
        path = path / "benchmark.yaml"
    return str(path), split_name


def load_benchmark(ref: str, split: Optional[str] = None) -> Benchmark:
    """Load a family YAML, resolve a split, and return a `Benchmark`.

    `ref` may include a ``:split`` suffix (see `parse_benchmark_ref`). The
    optional `split` argument takes precedence over any `:split` in the ref.
    """
    yaml_path, ref_split = parse_benchmark_ref(ref)
    chosen_split = split if split is not None else ref_split

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark YAML not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark YAML is not a mapping: {path}")

    family_name = str(data.get("name") or "").strip()
    if not family_name:
        raise ValueError(f"Benchmark YAML must set `name:` ({path})")
    family_type = str(data.get("type") or "local").strip()
    description = str(data.get("description") or "")
    family_backend: Dict[str, Any] = data.get("backend") or {}
    if not isinstance(family_backend, dict):
        raise ValueError(f"`backend:` must be a mapping in {path}")
    splits: Dict[str, Any] = data.get("splits") or {}
    if not isinstance(splits, dict):
        raise ValueError(f"`splits:` must be a mapping in {path}")

    # --- Local (tasks inline) ------------------------------------------------
    tasks_raw = data.get("tasks") or []
    if family_type == "local":
        if splits:
            raise ValueError(
                f"Local benchmark {family_name!r} cannot define `splits:`; "
                "use inline `tasks:` instead"
            )
        tasks = [Task.model_validate(t) for t in tasks_raw]
        if not tasks:
            raise ValueError(f"Local benchmark {family_name!r} has no tasks")

        base_dir = path.parent
        for task in tasks:
            task.workspace = str((base_dir / task.workspace).resolve())
            if not Path(task.workspace).is_dir():
                raise ValueError(f"Workspace not found: {task.workspace}")

        names = [t.name for t in tasks]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Benchmark {family_name!r} has duplicate task names"
            )

        fast_tasks_raw = data.get("fast_tasks") or []
        fast_tasks = list(fast_tasks_raw) if fast_tasks_raw else names

        return Benchmark(
            name=family_name,
            family=family_name,
            split="",
            type="local",
            description=description,
            backend=family_backend,
            tasks=tasks,
            fast_tasks=fast_tasks,
        )

    # --- Non-local (splits required) -----------------------------------------
    if not splits:
        raise ValueError(
            f"Non-local benchmark {family_name!r} (type={family_type!r}) "
            "must define at least one split under `splits:`"
        )

    if chosen_split is None:
        if len(splits) == 1:
            chosen_split = next(iter(splits))
        else:
            raise ValueError(
                f"Benchmark family {family_name!r} defines multiple splits "
                f"({sorted(splits)}); pass one via --split or ref suffix "
                f"'{path}:<split>'"
            )

    if chosen_split not in splits:
        raise ValueError(
            f"Unknown split {chosen_split!r} in family {family_name!r}; "
            f"available: {sorted(splits)}"
        )

    split_backend = splits[chosen_split] or {}
    if not isinstance(split_backend, dict):
        raise ValueError(
            f"Split {chosen_split!r} must be a mapping of backend overrides"
        )
    merged = {**family_backend, **split_backend}

    full_name = f"{family_name}-{chosen_split}"
    fast_tasks_raw = data.get("fast_tasks") or []

    return Benchmark(
        name=full_name,
        family=family_name,
        split=chosen_split,
        type=family_type,
        description=description,
        backend=merged,
        tasks=[],
        fast_tasks=list(fast_tasks_raw),
    )
