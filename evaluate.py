"""Validate replay then compute local metrics on selected TRAIN rows."""
import argparse
import json
from pathlib import Path
from lifemate.data import data_root, load_runtime, read_jsonl, select_rows
from lifemate.metrics import evaluate, verify

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--ids")
    a = p.parse_args()
    root = data_root(a.root, a.split)
    rows = select_rows(root, a.split, a.ids)
    predictions = read_jsonl(a.prediction)
    runtime = load_runtime(root, a.split)
    verify(rows, predictions, runtime, root, a.split)
    if a.split == "train":
        catalog = json.loads((root / a.split / f"{a.split}_tools_catalog.json").read_text(encoding="utf-8"))
        metrics, _ = evaluate(rows, predictions, runtime, root, a.split, catalog, 30000)
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Verified {len(predictions)} IDs, call schemas and replay hashes")

if __name__ == "__main__":
    main()
