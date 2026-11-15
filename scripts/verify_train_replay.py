"""CPU oracle plumbing check. This is NOT a model quality measurement."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lifemate.agent import Agent
from lifemate.asr import hydrate
from lifemate.data import atomic_json, load_runtime, read_jsonl
from lifemate.metrics import evaluate, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.root)
    runtime = load_runtime(root, "train")
    rows = read_jsonl(root / "train" / "train.jsonl")
    catalog = json.loads((root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    predictions, failures = [], []
    for row in rows:
        class Oracle:
            def __init__(self):
                self.calls = iter([*[json.dumps({"tool_calls": [c]}, ensure_ascii=False) for c in row["label"]["tool_calls"]], '{"tool_calls":[]}'])
            def generate(self, messages, deadline):
                return next(self.calls, '{"tool_calls":[]}')
        env = runtime.ToolEnv("train", row["id"], root=root)
        sample = hydrate(row, root, "train", "gold", None, time.monotonic() + 60)
        pred, trace = Agent(Oracle(), catalog, max_steps=32).run(sample, env, time.monotonic() + 60)
        predictions.append(pred)
        if pred["tool_calls"] != row["label"]["tool_calls"]:
            failures.append({"id": row["id"], "trace": trace})
    verify(rows, predictions, runtime, root, "train")
    scores, _ = evaluate(rows, predictions, runtime, root, "train", catalog)
    report = {"test": "TRAIN oracle replay, NOT language-model inference", "metrics": scores,
              "mismatched_sequences": len(failures), "failures": failures}
    atomic_json(a.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    # Review every mismatch. An invalid label reference must not weaken an inference guardrail.

if __name__ == "__main__":
    main()
