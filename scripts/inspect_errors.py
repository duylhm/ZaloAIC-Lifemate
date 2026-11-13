"""Inspect failing samples from a LifeMate submission against labeled ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate  # noqa: E402
from src.data_utils import read_jsonl, resolve_data_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze LifeMate error cases by category")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--max-per-bucket", type=int, default=5)
    args = ap.parse_args()

    data_root = resolve_data_root(args.root, split=args.split)
    sys.path.insert(0, str(data_root / args.split))
    from zaic_tools import ToolEnv  # noqa: E402

    filter_ids, unseen_apps = evaluate._load_ids_and_apps(args.ids_file)
    fuzzy_map = evaluate.load_catalog_fuzzy_map(data_root / args.split / f"{args.split}_tools_catalog.json")
    subs = {r["id"]: r for r in read_jsonl(args.submission)}

    summary = evaluate.evaluate_predictions(
        data_root,
        args.split,
        args.submission,
        sample_ids=filter_ids,
        unseen_apps=unseen_apps,
    )
    print("=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in read_jsonl(data_root / args.split / f"{args.split}.jsonl"):
        qid = item["id"]
        if filter_ids is not None and qid not in filter_ids:
            continue

        gold_calls = item["label"]["tool_calls"]
        gold_env = ToolEnv(args.split, qid, root=data_root)
        for gc in gold_calls:
            gold_env.call(gc["name"], gc.get("arguments") or {})
        gold_hash = gold_env.get_state_hash()

        sub = subs.get(qid, {"tool_calls": [], "state_hash": ""})
        sub_calls = sub.get("tool_calls") or []
        sub_env = ToolEnv(args.split, qid, root=data_root)
        for sc in sub_calls:
            sub_env.call(sc.get("name", ""), sc.get("arguments") or {})
        sub_hash = sub_env.get_state_hash()

        res = evaluate.score_sample(sub_calls, sub_hash, gold_calls, gold_hash, fuzzy_map)
        reason = evaluate._classify_failure(sub_calls, gold_calls, res["state_match"], res["M"])
        if reason != "ok":
            buckets[reason].append({
                "id": qid,
                "score": round(res["score"], 3),
                "f1_uncond": round(res["f1_unconditional"], 3),
                "state_ok": res["state_match"],
                "audio_text": item.get("audio_text", ""),
                "gold_calls": gold_calls,
                "sub_calls": sub_calls,
            })

    print("\n=== SAMPLE FAILURES BY BUCKET ===")
    for reason, rows in sorted(buckets.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {reason} ({len(rows)} cases) ---")
        for ex in rows[: args.max_per_bucket]:
            print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
