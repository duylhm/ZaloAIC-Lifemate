"""Create deterministic Unseen-App + Stratified In-Domain local validation splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import REPO_ROOT, read_jsonl, resolve_data_root  # noqa: E402


DEFAULT_HOLDOUT_APPS = [
    "bike_rental",
    "exchange_rate",
    "medication",
    "package_tracking",
]


def _sample_category(sample: dict) -> str:
    calls = (sample.get("label") or {}).get("tool_calls") or []
    if not calls:
        return "out_of_scope"
    if any(c.get("name") == "sys.ask_missing_info" for c in calls):
        return "ask_missing_info"
    return "single_call" if len(calls) == 1 else "multi_call"


def _sample_used_apps(sample: dict) -> set[str]:
    used: set[str] = set()
    for tc in (sample.get("label") or {}).get("tool_calls") or []:
        used.add(tc["name"].split(".")[0])
    for turn in sample.get("history") or []:
        for tc in turn.get("tool_calls") or []:
            used.add(tc["name"].split(".")[0])
    return used - {"sys", ""}


def build_splits(
    rows: list[dict],
    holdout_apps: list[str],
    indomain_size: int = 100,
    seed: int = 42,
) -> dict:
    """Split training data cleanly into train_ids, val_unseen_ids, and val_indomain_ids."""
    holdout_set = set(holdout_apps)
    unseen_rows: list[dict] = []
    pool_rows: list[dict] = []

    for row in rows:
        used = _sample_used_apps(row)
        if used & holdout_set:
            unseen_rows.append(row)
        else:
            pool_rows.append(row)

    rng = random.Random(seed)
    buckets: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for row in pool_rows:
        has_hist = bool(row.get("history"))
        buckets[(_sample_category(row), has_hist)].append(row["id"])

    # Allocate stratified counts proportional to bucket sizes cleanly.
    total_pool = len(pool_rows)
    indomain_ids: list[str] = []
    remainders: list[tuple[float, tuple[str, bool], list[str]]] = []

    for key in sorted(buckets.keys()):
        ids_in_bucket = sorted(buckets[key])
        rng.shuffle(ids_in_bucket)
        exact = indomain_size * len(ids_in_bucket) / max(1, total_pool)
        take = min(len(ids_in_bucket), int(exact))
        indomain_ids.extend(ids_in_bucket[:take])
        remainders.append((exact - take, key, ids_in_bucket[take:]))

    # Distribute remaining budget cleanly to largest fractional remainders.
    remainders.sort(key=lambda x: (-x[0], x[1]))
    idx = 0
    while len(indomain_ids) < min(indomain_size, total_pool) and remainders:
        frac, key, leftover = remainders[idx % len(remainders)]
        if leftover:
            indomain_ids.append(leftover.pop(0))
        idx += 1

    indomain_set = set(indomain_ids)
    unseen_ids = [r["id"] for r in unseen_rows]
    train_ids = [r["id"] for r in pool_rows if r["id"] not in indomain_set]

    assert not (set(train_ids) & indomain_set)
    assert not (set(train_ids) & set(unseen_ids))
    assert not (indomain_set & set(unseen_ids))
    assert len(train_ids) + len(indomain_ids) + len(unseen_ids) == len(rows)

    return {
        "seed": seed,
        "holdout_apps": sorted(holdout_apps),
        "train_ids": train_ids,
        "val_unseen_ids": unseen_ids,
        "val_indomain_ids": sorted(indomain_ids),
        "val_all_ids": sorted(set(unseen_ids) | indomain_set),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Unseen-App + Stratified In-Domain local splits")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "splits")
    ap.add_argument("--indomain-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = resolve_data_root(args.root, split="train")
    rows = read_jsonl(data_root / "train" / "train.jsonl")
    split_data = build_splits(
        rows,
        holdout_apps=DEFAULT_HOLDOUT_APPS,
        indomain_size=args.indomain_size,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val_unseen.json").write_text(
        json.dumps({"holdout_apps": split_data["holdout_apps"], "ids": split_data["val_unseen_ids"]}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "val_indomain.json").write_text(
        json.dumps({"holdout_apps": split_data["holdout_apps"], "ids": split_data["val_indomain_ids"]}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "val_all.json").write_text(
        json.dumps(
            {
                "seed": split_data["seed"],
                "holdout_apps": split_data["holdout_apps"],
                "ids": split_data["val_all_ids"],
                "val_unseen_ids": split_data["val_unseen_ids"],
                "val_indomain_ids": split_data["val_indomain_ids"],
                "train_ids": split_data["train_ids"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "total_train_rows": len(rows),
        "train_split_count": len(split_data["train_ids"]),
        "val_unseen_count": len(split_data["val_unseen_ids"]),
        "val_indomain_count": len(split_data["val_indomain_ids"]),
        "val_all_count": len(split_data["val_all_ids"]),
        "holdout_apps": split_data["holdout_apps"],
        "output_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
