"""Build frozen Plan-RewardBench search/val/test manifests.

Rows are grouped by normalized user task before splitting. Plan-RewardBench has
many sibling rows for the same task UUID/query; splitting rows directly leaks
near-duplicate scenarios across search/validation/test. We balance by primary
rubric bucket and by a structural difficulty proxy so that long multi-turn /
tool-heavy grouped scenarios are spread across splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

DATASET = "wyy1112/Plan-RewardBench"
SPLIT = "train"
BUCKETS: tuple[str, ...] = (
    "irrelevance_unavailable",
    "planning_multi_easy",
    "planning_multi_hard",
    "planning_robustness",
    "planning_single_easy",
    "planning_single_hard",
    "refusal",
)


DEFAULT_DERIVED_BUCKET_LIMIT = 40
TINY_BUCKETS: tuple[str, ...] = (
    "irrelevance_unavailable",
    "planning_multi_easy",
    "planning_multi_hard",
    "planning_robustness",
    "refusal",
)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def _normalize_task(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _group_key(row: dict[str, Any]) -> str:
    query_key = _normalize_task(row.get("query"))
    if query_key:
        return f"query:{query_key}"
    uuid = str(row.get("uuid") or "").strip()
    if uuid:
        return f"uuid:{uuid}"
    return f"pair:{row['_pair_id']}"


def _group_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _pair_id(row: dict[str, Any], row_index: int) -> str:
    bucket = str(row.get("_lcp_bucket") or "unknown").strip() or "unknown"
    uuid = str(row.get("uuid") or "").strip()
    return f"{_safe_id(bucket)}__{uuid}__{row_index}"


def _messages(trajectory: Any) -> list[dict[str, Any]]:
    if not isinstance(trajectory, dict):
        return []
    messages = trajectory.get("messages")
    return [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []


def _content_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(str(content))
    return total


def _difficulty(row: dict[str, Any]) -> float:
    chosen_messages = _messages(row.get("chosen"))
    reject_messages = _messages(row.get("reject"))
    max_turns = max(len(chosen_messages), len(reject_messages))
    max_chars = max(_content_chars(chosen_messages), _content_chars(reject_messages))
    tools = row.get("tools")
    n_tools = len(tools) if isinstance(tools, list) else 0
    tool_events = sum(
        1
        for message in chosen_messages + reject_messages
        if str(message.get("role") or "").startswith("tool")
    )
    return max_turns + (max_chars / 2000.0) + (n_tools * 0.5) + (tool_events * 0.25)


def _difficulty_bins(rows: list[dict[str, Any]], *, n_bins: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (float(row["_difficulty"]), str(row["_pair_id"])))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for index, row in enumerate(ordered):
        bin_index = min(index * n_bins // max(1, len(ordered)), n_bins - 1)
        bins[bin_index].append(row)
    return [bucket for bucket in bins if bucket]


def _difficulty_bins_for_groups(groups: list[dict[str, Any]], *, n_bins: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(groups, key=lambda group: (float(group["_difficulty"]), str(group["_group_hash"])))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for index, group in enumerate(ordered):
        bin_index = min(index * n_bins // max(1, len(ordered)), n_bins - 1)
        bins[bin_index].append(group)
    return [bucket for bucket in bins if bucket]


def _split_counts(n_items: int, split_fractions: list[tuple[str, float]]) -> dict[str, int]:
    counts = {name: round(n_items * fraction) for name, fraction in split_fractions}
    while sum(counts.values()) > n_items:
        name = max(counts, key=lambda split_name: (counts[split_name], split_name != "search"))
        counts[name] -= 1
    while sum(counts.values()) < n_items:
        counts["search"] = counts.get("search", 0) + 1

    active = [name for name, fraction in split_fractions if fraction > 0]
    if n_items >= len(active):
        for name in active:
            if counts.get(name, 0) == 0:
                donor = max(active, key=lambda split_name: counts.get(split_name, 0))
                if counts.get(donor, 0) > 1:
                    counts[donor] -= 1
                    counts[name] = counts.get(name, 0) + 1
    return counts


def _split_groups(
    groups: list[dict[str, Any]],
    *,
    rng: random.Random,
    split_fractions: list[tuple[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    """Assign whole task groups while balancing pair counts per stratum.

    Plan-RB groups are uneven: a normalized task/query can contribute anywhere
    from one to five pair rows. Balancing only by group count can therefore make
    one split over- or under-represent a bucket/difficulty stratum. We keep the
    split unit as the whole group, but greedily fill pair-count targets so the
    benchmark metric sees roughly the requested 60/20/20 mix within each
    bucket+difficulty band.
    """
    shuffled = list(groups)
    rng.shuffle(shuffled)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in split_fractions}
    active = [name for name, fraction in split_fractions if fraction > 0]
    if not shuffled or not active:
        return out

    total_pairs = sum(len(group["rows"]) for group in shuffled)
    targets = _split_counts(total_pairs, split_fractions)
    assigned_pairs = {name: 0 for name in out}

    # Largest groups are hardest to place; assign them first. The prior shuffle
    # gives deterministic seed-controlled tie breaking without sorting by row id.
    shuffled.sort(key=lambda group: len(group["rows"]), reverse=True)
    for group in shuffled:
        group_pairs = len(group["rows"])

        def fill_ratio(split_name: str) -> tuple[float, int, str]:
            target = max(1, targets.get(split_name, 0))
            return (
                (assigned_pairs[split_name] + group_pairs) / target,
                assigned_pairs[split_name],
                split_name,
            )

        split_name = min(active, key=fill_ratio)
        out[split_name].append(group)
        assigned_pairs[split_name] += group_pairs
    return out


def _primary_bucket(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("_lcp_bucket") or "unknown").strip() or "unknown" for row in rows)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_splits(
    *,
    seed: int,
    search_fraction: float,
    val_fraction: float,
    shadow_fraction: float,
    n_bins: int,
) -> dict[str, list[dict[str, Any]]]:
    test_fraction = 1.0 - search_fraction - val_fraction - shadow_fraction
    if test_fraction < -1e-9:
        raise ValueError("search_fraction + val_fraction + shadow_fraction must be <= 1.0")
    split_fractions: list[tuple[str, float]] = [
        ("search", search_fraction),
        ("val", val_fraction),
    ]
    if shadow_fraction > 0:
        split_fractions.append(("shadow", shadow_fraction))
    split_fractions.append(("test", max(0.0, test_fraction)))

    ds = load_dataset(DATASET, split=SPLIT)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, raw_row in enumerate(ds):
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        bucket = str(row.get("_lcp_bucket") or "unknown").strip() or "unknown"
        row["_pair_id"] = _pair_id(row, row_index)
        row["_difficulty"] = _difficulty(row)
        row["_group_key"] = _group_key(row)
        row["_group_hash"] = _group_hash(row["_group_key"])
        by_bucket[bucket].append(row)
        by_group_key[row["_group_key"]].append(row)

    missing = [bucket for bucket in BUCKETS if bucket not in by_bucket]
    if missing:
        raise ValueError(f"Missing expected buckets: {missing}")

    groups: list[dict[str, Any]] = []
    for key, rows in by_group_key.items():
        groups.append({
            "_group_key": key,
            "_group_hash": _group_hash(key),
            "_primary_bucket": _primary_bucket(rows),
            "_difficulty": sum(float(row["_difficulty"]) for row in rows) / len(rows),
            "rows": rows,
        })

    by_primary_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_primary_bucket[str(group["_primary_bucket"])].append(group)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in split_fractions}
    for bucket in BUCKETS:
        for difficulty_bin in _difficulty_bins_for_groups(by_primary_bucket[bucket], n_bins=n_bins):
            split_groups = _split_groups(
                difficulty_bin,
                rng=rng,
                split_fractions=split_fractions,
            )
            for split_name, assigned_groups in split_groups.items():
                for group in assigned_groups:
                    splits[split_name].extend(group["rows"])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: (str(row["_lcp_bucket"]), str(row["_pair_id"])))
    return splits


def _manifest(
    split: str,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    n_bins: int,
    source_manifest: str | None = None,
    selection: str | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    difficulty: dict[str, dict[str, float]] = {}
    for bucket in BUCKETS:
        bucket_rows = [row for row in rows if row.get("_lcp_bucket") == bucket]
        counts[bucket] = len(bucket_rows)
        scores = [float(row["_difficulty"]) for row in bucket_rows]
        if scores:
            difficulty[bucket] = {
                "min": min(scores),
                "mean": sum(scores) / len(scores),
                "max": max(scores),
            }
    payload = {
        "name": f"plan-rewardbench-{split}",
        "dataset": DATASET,
        "dataset_split": SPLIT,
        "split": split,
        "seed": seed,
        "grouping": "normalized_query_with_uuid_fallback",
        "n_groups": len({str(row["_group_hash"]) for row in rows}),
        "group_key_hashes": sorted({str(row["_group_hash"]) for row in rows}),
        "difficulty_proxy": "turns + chars/2000 + 0.5*tools + 0.25*tool_events",
        "difficulty_bins_per_bucket": n_bins,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(rows),
        "bucket_counts": counts,
        "bucket_difficulty": difficulty,
        "pair_ids": [str(row["_pair_id"]) for row in rows],
    }
    if source_manifest is not None:
        payload["source_manifest"] = source_manifest
    if selection is not None:
        payload["selection"] = selection
    return payload


def _stratified_subset(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    per_bucket_limit: int | None = None,
    fraction: float | None = None,
) -> list[dict[str, Any]]:
    if (per_bucket_limit is None) == (fraction is None):
        raise ValueError("Set exactly one of per_bucket_limit or fraction")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("_lcp_bucket") or "unknown")].append(row)

    for bucket in BUCKETS:
        bucket_rows = sorted(by_bucket.get(bucket, []), key=lambda row: str(row["_pair_id"]))
        rng.shuffle(bucket_rows)
        if per_bucket_limit is not None:
            count = min(per_bucket_limit, len(bucket_rows))
        else:
            count = max(1, round(len(bucket_rows) * float(fraction))) if bucket_rows else 0
        selected.extend(bucket_rows[:count])
    selected.sort(key=lambda row: (str(row["_lcp_bucket"]), str(row["_pair_id"])))
    return selected


def _tiny_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for bucket in TINY_BUCKETS:
        bucket_rows = sorted(
            [row for row in rows if row.get("_lcp_bucket") == bucket],
            key=lambda row: str(row["_pair_id"]),
        )
        if bucket_rows:
            selected.append(bucket_rows[0])
    return selected


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-fraction", type=float, default=0.60)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--shadow-fraction", type=float, default=0.0)
    parser.add_argument("--difficulty-bins", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/plan_rewardbench/data"))
    parser.add_argument("--prefix", default="plan_rb")
    parser.add_argument("--no-derived", action="store_true")
    args = parser.parse_args()

    splits = build_splits(
        seed=args.seed,
        search_fraction=args.search_fraction,
        val_fraction=args.val_fraction,
        shadow_fraction=args.shadow_fraction,
        n_bins=args.difficulty_bins,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_ids: list[str] = []
    all_groups: list[str] = []
    for split, rows in splits.items():
        manifest = _manifest(split, rows, seed=args.seed, n_bins=args.difficulty_bins)
        path = args.out_dir / f"{args.prefix}_{split}.json"
        _write_manifest(path, manifest)
        all_ids.extend(manifest["pair_ids"])
        all_groups.extend(manifest["group_key_hashes"])
        print(f"{split}: {manifest['n_pairs']} -> {path}")
        print(f"  bucket_counts: {manifest['bucket_counts']}")
        print(f"  mean_difficulty: { {k: round(v['mean'], 2) for k, v in manifest['bucket_difficulty'].items()} }")

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Generated splits are not disjoint")
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("Generated splits have group leakage")
    print(f"total unique ids: {len(set(all_ids))}")
    print(f"total unique groups: {len(set(all_groups))}")

    if not args.no_derived:
        derived_specs: list[tuple[str, list[dict[str, Any]], str, str]] = []
        if "search" in splits:
            source = args.out_dir / f"{args.prefix}_search.json"
            search_panel = _stratified_subset(
                splits["search"],
                seed=args.seed,
                per_bucket_limit=DEFAULT_DERIVED_BUCKET_LIMIT,
            )
            derived_specs.append((
                "search_panel",
                search_panel,
                str(source),
                f"fixed stratified panel: up to {DEFAULT_DERIVED_BUCKET_LIMIT} pair_ids per bucket from search",
            ))
            derived_specs.append((
                "search_panel_25",
                _stratified_subset(search_panel, seed=args.seed + 1, fraction=0.25),
                str(args.out_dir / f"{args.prefix}_search_panel.json"),
                "fixed stratified subset: 25% per bucket from search-panel",
            ))
            derived_specs.append((
                "tiny_search_5",
                _tiny_subset(search_panel),
                str(args.out_dir / f"{args.prefix}_search_panel.json"),
                "one representative pair from five buckets for loop smoke tests",
            ))
        for split in ("val", "shadow", "test"):
            if split not in splits:
                continue
            source = args.out_dir / f"{args.prefix}_{split}.json"
            derived_specs.append((
                f"{split}_50",
                _stratified_subset(splits[split], seed=args.seed + 2, fraction=0.50),
                str(source),
                f"fixed stratified subset: 50% per bucket from {split}",
            ))
        if "val" in splits:
            derived_specs.append((
                "tiny_val_5",
                _tiny_subset(splits["val"]),
                str(args.out_dir / f"{args.prefix}_val.json"),
                "one representative pair from five buckets for loop smoke tests",
            ))

        for split, rows, source_manifest, selection in derived_specs:
            manifest = _manifest(
                split.replace("_", "-"),
                rows,
                seed=args.seed,
                n_bins=args.difficulty_bins,
                source_manifest=source_manifest,
                selection=selection,
            )
            path = args.out_dir / f"{args.prefix}_{split}.json"
            _write_manifest(path, manifest)
            print(f"{split}: {manifest['n_pairs']} -> {path}")
            print(f"  bucket_counts: {manifest['bucket_counts']}")


if __name__ == "__main__":
    main()
