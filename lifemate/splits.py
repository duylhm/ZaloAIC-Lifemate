"""Grouped holdout, with held-out target apps and disjoint pilot/confirm sets."""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import atomic_json, file_hash, read_jsonl

HOLDOUT_APPS = {"bike_rental", "exchange_rate", "medication", "package_tracking"}


def groups(rows, audio_base=None):
    parents = list(range(len(rows)))
    owner, hashes = {}, {}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        # Group repeated audio across current request AND history; hashes also catch renamed duplicates.
        for turn in [row, *row.get("history", [])]:
            rel = turn.get("audio")
            if not rel:
                continue
            key = rel.replace("\\", "/")
            if audio_base is not None:
                path = Path(audio_base) / rel
                if not path.is_file():
                    raise FileNotFoundError(f"Split creation needs all audio: {path}")
                if key not in hashes:
                    hashes[key] = file_hash(path)
                key = hashes[key]
            if key in owner:
                parents[find(i)] = find(owner[key])
            owner[key] = i
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[find(i)].append(row)
    return list(result.values())


def category(row):
    calls = row["label"]["tool_calls"]
    return "oos" if not calls else "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "multi" if len(calls) > 1 else "single"


def make_splits(rows, audio_base=None, seed=42, pilot_size=30, indomain_size=100):
    rng = random.Random(seed)
    grouped = groups(rows, audio_base)
    unseen, pool = [], []
    for group in grouped:
        targets = {c["name"].split(".")[0] for r in group for c in r["label"]["tool_calls"]}
        (unseen if targets & HOLDOUT_APPS else pool).append(group)
    rng.shuffle(pool)
    # Round robin among categories gives rare OOS/ask cases representation.
    buckets = defaultdict(list)
    for group in pool:
        buckets[category(group[0])].append(group)
    interleaved = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                interleaved.append(buckets[key].pop())
    validation = list(unseen)
    used = 0
    train_groups = []
    for group in interleaved:
        if used < indomain_size:
            validation.append(group)
            used += len(group)
        else:
            train_groups.append(group)
    # Stratify pilot by heldout membership + action category, keeping each group intact.
    buckets = defaultdict(list)
    for group in validation:
        is_unseen = any(c["name"].split(".")[0] in HOLDOUT_APPS for r in group for c in r["label"]["tool_calls"])
        buckets[(is_unseen, category(group[0]))].append(group)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    pilot, confirm = [], []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                group = buckets[key].pop()
                (pilot if len(pilot) < pilot_size else confirm).extend(group)
    train = [r for group in train_groups for r in group]
    if not train or not confirm or not pilot:
        raise ValueError("Insufficient independent groups for train/pilot/confirm; reduce split sizes")
    return {"train": train, "pilot": pilot, "confirm": confirm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pilot-size", type=int, default=30)
    p.add_argument("--indomain-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = Path(a.root)
    source = root / "train" / "train.jsonl"
    rows = read_jsonl(source)
    result = make_splits(rows, root / "train", a.seed, a.pilot_size, a.indomain_size)
    report = {"seed": a.seed, "dataset_sha256": file_hash(source), "grouping": "SHA256 of all current/history audio",
              "holdout_apps": sorted(HOLDOUT_APPS), "counts": {}, "categories": {}}
    for name, selected in result.items():
        payload = {"ids": [r["id"] for r in selected], "dataset_sha256": report["dataset_sha256"],
                   "seed": a.seed, "holdout_apps": sorted(HOLDOUT_APPS)}
        atomic_json(Path(a.output) / f"{name}.json", payload)
        report["counts"][name] = len(selected)
        report["categories"][name] = dict(Counter(category(r) for r in selected))
    atomic_json(Path(a.output) / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()
