"""Paired score differences; the bootstrap is descriptive on a small pilot."""
import argparse
import json
import random
import statistics
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("left")
    p.add_argument("right")
    a = p.parse_args()
    def read(path):
        values = [json.loads(s) for s in Path(path).with_suffix(".scores.jsonl").read_text(encoding="utf-8").splitlines() if s]
        return {r["id"]: r for r in values}
    left, right = read(a.left), read(a.right)
    if left.keys() != right.keys():
        raise ValueError("A/B must use exactly the same IDs")
    delta = [100 * (right[k]["score"] - left[k]["score"]) for k in left]
    rng = random.Random(42)
    boot = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
    print(json.dumps({"n": len(delta), "right_minus_left": statistics.mean(delta),
                      "wins": sum(v > 0 for v in delta), "losses": sum(v < 0 for v in delta),
                      "bootstrap_95_percent_descriptive": [boot[50], boot[1949]]}, indent=2))

if __name__ == "__main__":
    main()
