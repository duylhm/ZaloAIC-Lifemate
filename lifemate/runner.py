"""Resumable inference. Checkpoints contain predictions AND traces atomically."""
from __future__ import annotations

import argparse
import json
import time
from importlib import metadata
from pathlib import Path

from .agent import Agent
from .asr import Gipformer, PhoWhisper, hydrate
from .data import atomic_json, data_root, digest, file_hash, load_runtime, select_rows, write_jsonl
from .metrics import evaluate, verify
from .models import HFBackend, NoopBackend, offline_environment
from .health import abort_slow_train


def parser():
    p = argparse.ArgumentParser(description="LifeMate zero-shot baseline")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    p.add_argument("--ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision")
    p.add_argument("--asr", choices=["gold", "gipformer", "phowhisper"], default="gipformer")
    p.add_argument("--asr-model")
    p.add_argument("--asr-revision")
    p.add_argument("--backend", choices=["hf", "noop"], default="hf")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--seconds-per-sample", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate", action="store_true")
    return p


def model_identity(path):
    if path == "noop":
        return path
    base = Path(path)
    # Snapshot path pins remote revision; local deployment also records file sizes and config hashes.
    return {"path": str(base), "files": {p.name: {"bytes": p.stat().st_size,
        "sha256": file_hash(p) if p.suffix in (".json", ".txt") else None}
        for p in base.iterdir() if p.is_file()}}


def run(args, backend=None, asr=None):
    if args.seconds_per_sample <= 0 or args.max_steps < 1:
        raise ValueError("Time budget and max_steps must be positive")
    if args.asr == "gold" and args.split != "train":
        raise ValueError("Gold transcripts cannot be used for test inference")
    if args.evaluate and args.split != "train":
        raise ValueError("Evaluation requires TRAIN labels")
    if args.backend == "noop" and args.split != "train":
        raise ValueError("Noop is restricted to TRAIN infrastructure smoke tests")
    if args.offline:
        offline_environment()
    root = data_root(args.root, args.split)
    rows = select_rows(root, args.split, args.ids, args.limit)
    runtime = load_runtime(root, args.split)
    catalog_path = root / args.split / f"{args.split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    checkpoint = output.with_suffix(".run")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if backend is None:
        backend = NoopBackend() if args.backend == "noop" else HFBackend(args.model, args.revision, args.offline,
            not args.no_4bit, args.max_new_tokens, seed=args.seed)
    if asr is None and args.asr != "gold":
        cls = Gipformer if args.asr == "gipformer" else PhoWhisper
        opts = dict(revision=args.asr_revision, offline=args.offline)
        if args.asr_model:
            opts["model"] = args.asr_model
        asr = cls(**opts)
    code_root = Path(__file__).parent
    versions = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "sherpa-onnx", "soundfile", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    config = {k: v for k, v in vars(args).items() if k not in ("output", "evaluate")}
    manifest = {"config": config, "row_hash": digest(rows), "ids": [r["id"] for r in rows],
        "catalog_hash": file_hash(catalog_path), "versions": versions,
        "code": {p.name: file_hash(p) for p in code_root.glob("*.py")},
        "runtime": {p.relative_to(root / args.split).as_posix(): file_hash(p)
                    for p in (root / args.split / "zaic_tools").rglob("*.py")},
        "model": model_identity(backend.path), "asr": model_identity(asr.path) if asr else "gold"}
    manifest_file = checkpoint / "manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Run configuration/data/code changed. Use a NEW output filename; refusing mixed resume.")
    else:
        if list(checkpoint.glob("sample-*.json")) or output.exists():
            raise ValueError("Existing output lacks a matching manifest. Choose a new output filename")
        atomic_json(manifest_file, manifest)
    agent = Agent(backend, catalog, args.max_steps)
    results, traces = [], []
    for i, row in enumerate(rows):
        record_path = checkpoint / f"sample-{i:05d}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["prediction"]["id"] != row["id"]:
                raise ValueError("Checkpoint ID mismatch")
        else:
            start = time.monotonic()  # Includes environment setup + every audio read/ASR pass.
            deadline = start + args.seconds_per_sample
            env = runtime.ToolEnv(args.split, row["id"], root=root)
            try:
                sample = hydrate(row, root, args.split, args.asr, asr, deadline)
                prediction, trace = agent.run(sample, env, deadline)
                trace["transcripts"] = {"current": sample["text"], "history": [t.get("text") for t in sample["history"] if t.get("role") == "user"]}
            except (TimeoutError, ValueError) as error:
                prediction = {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
                trace = {"stop": "timeout" if isinstance(error, TimeoutError) else "input_or_context_error", "error": str(error)}
            # Missing model/audio/runtime errors intentionally propagate instead of creating a fake complete submission.
            prediction["time"] = round((time.monotonic() - start) * 1000, 3)
            if prediction["time"] > args.seconds_per_sample * 1000:
                trace["stop"] = "timeout"
            record = {"prediction": prediction, "trace": trace}
            atomic_json(record_path, record)
        results.append(record["prediction"])
        traces.append({"id": row["id"], **record["trace"]})
        print(f"[{i + 1}/{len(rows)}] {row['id']}: {record['trace']['stop']} "
              f"calls={len(record['prediction']['tool_calls'])} ms={record['prediction']['time']:.0f}", flush=True)
        if len(rows) > 5:
            abort_slow_train(args.split, traces)
    # Verify every call and hash independently before exporting a complete file.
    verify(rows, results, runtime, root, args.split)
    write_jsonl(output, results)
    write_jsonl(output.with_suffix(".traces.jsonl"), traces)
    times = sorted(p["time"] for p in results)
    summary = {"n": len(rows), "mean_ms": sum(times) / len(times), "max_ms": max(times),
               "p95_ms": times[min(len(times) - 1, int(len(times) * .95))],
               "stops": {s: sum(t["stop"] == s for t in traces) for s in sorted({t["stop"] for t in traces})}}
    if args.evaluate:
        failures = {t["id"] for t in traces if t["stop"] == "input_or_context_error"}
        metrics, details = evaluate(rows, results, runtime, root, args.split, catalog, args.seconds_per_sample * 1000, failures)
        summary.update(metrics)
        write_jsonl(output.with_suffix(".scores.jsonl"), details)
    atomic_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
