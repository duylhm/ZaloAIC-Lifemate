"""Compliant Multi-Turn ReAct Inference Pipeline for LifeMate (Local CV, Colab T4, Offline Docker)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from src.agent import (
    HFTransformersBackend,
    HeuristicBackend,
    LifeMateAgent,
    NoopBackend,
    OpenAICompatibleBackend,
)
from src.asr import OfflineASR, hydrate_sample_transcripts, load_asr_cache
from src.data_utils import REPO_ROOT, read_jsonl, resolve_data_root
from src.prompt_builder import load_catalog_dict


def _build_backend(args: argparse.Namespace):
    if args.backend == "noop":
        return NoopBackend()
    if args.backend == "heuristic":
        return HeuristicBackend()
    if args.backend == "openai":
        return OpenAICompatibleBackend(
            base_url=args.base_url,
            model=args.model,
        )
    return HFTransformersBackend(
        model_id=args.model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.offline,
    )


def _needs_live_asr(
    rows: list[dict],
    split_dir: Path,
    split_name: str,
    primary_cache: dict[str, str],
    no_cache: bool,
) -> bool:
    for r in rows:
        refs = [r.get("audio")] + [t.get("audio") for t in (r.get("history") or []) if t.get("role") == "user"]
        existing_texts = [r.get("audio_text")] + [
            t.get("audio_text") for t in (r.get("history") or []) if t.get("role") == "user"
        ]
        for rel, txt in zip(refs, existing_texts):
            if txt and not no_cache:
                continue
            if rel and (split_dir / rel).is_file():
                norm = rel.replace("\\", "/").lstrip("/")
                cached = primary_cache.get(f"{split_name}/{norm}") or primary_cache.get(norm)
                if no_cache or not cached:
                    return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LifeMate ReAct inference and output submission.jsonl")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--ids-file", type=Path, default=None, help="Optional JSON file restricting sample IDs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["hf", "openai", "noop", "heuristic"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter directory")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--offline", action="store_true", help="Enforce HF_HUB_OFFLINE=1 & local_files_only=True")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "submission.jsonl")
    ap.add_argument("--mirror-path", type=Path, default=None, help="Optional backup copy (e.g. Google Drive)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output instead of resuming")
    ap.add_argument("--no-cache", action="store_true", help="Transcribe raw audio live instead of using ASR cache")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    ap.add_argument("--asr-beams", type=int, default=3)
    ap.add_argument("--asr-cache", type=Path, default=None)
    ap.add_argument("--asr-alt-cache", type=Path, default=None)
    ap.add_argument("--no-alt-cache", action="store_true")
    ap.add_argument("--eval", action="store_true", help="Run evaluate.py automatically if split has labels")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_root = resolve_data_root(args.root, split=args.split)
    split_dir = data_root / args.split
    sys.path.insert(0, str(split_dir))
    import zaic_tools  # noqa: E402

    catalog_path = zaic_tools.catalog_path(args.split, data_root)
    catalog_tools = load_catalog_dict(catalog_path)
    rows = read_jsonl(split_dir / f"{args.split}.jsonl")

    unseen_apps = None
    if args.ids_file:
        id_payload = json.loads(args.ids_file.read_text(encoding="utf-8"))
        id_list = id_payload["ids"] if isinstance(id_payload, dict) else id_payload
        if isinstance(id_payload, dict):
            apps = id_payload.get("holdout_apps") or id_payload.get("unseen_apps")
            unseen_apps = set(apps) if apps else None
        id_set = set(id_list)
        rows = [r for r in rows if r["id"] in id_set]
    if args.limit is not None:
        rows = rows[: args.limit]

    # Resolve ASR caches & optional live ASR model.
    default_primary = REPO_ROOT / "asr_public_test_pho.json"
    default_alt = REPO_ROOT / "asr_public_test.json"
    primary_cache_path = args.asr_cache or (default_primary if args.split == "public_test" else None)
    alt_cache_path = None if (args.no_alt_cache or args.no_cache) else (
        args.asr_alt_cache or (default_alt if args.split == "public_test" else None)
    )

    primary_cache = {} if args.no_cache else load_asr_cache(primary_cache_path)
    alt_cache = {} if args.no_cache else load_asr_cache(alt_cache_path)

    asr_engine = None
    if _needs_live_asr(rows, split_dir, args.split, primary_cache, args.no_cache):
        print(f"[ASR] Initializing live OfflineASR: {args.asr_model} (offline={args.offline})")
        asr_engine = OfflineASR(
            model_id=args.asr_model,
            num_beams=args.asr_beams,
            local_files_only=args.offline,
        )
    elif primary_cache:
        print(f"[ASR] Using cached transcripts ({len(primary_cache)} entries, alt={len(alt_cache)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if not args.force and args.output.is_file():
        for rec in read_jsonl(args.output):
            completed[rec["id"]] = rec
        print(f"[Resume] Found {len(completed)}/{len(rows)} completed samples in {args.output}")
    elif args.force and args.output.is_file():
        args.output.unlink()

    pending = [r for r in rows if r["id"] not in completed]
    print(
        f"[Inference] split={args.split} total={len(rows)} pending={len(pending)} "
        f"backend={args.backend} model={args.model} adapter={args.adapter}"
    )

    if pending:
        backend = _build_backend(args)
        agent = LifeMateAgent(
            backend=backend,
            catalog_tools=catalog_tools,
            max_turns=args.max_turns,
        )

        t0 = time.time()
        with args.output.open("a", encoding="utf-8") as fout:
            for idx, sample in enumerate(pending, start=1):
                hydrated, asr_alt = hydrate_sample_transcripts(
                    sample,
                    split_dir=split_dir,
                    split_name=args.split,
                    asr_engine=asr_engine,
                    primary_cache=primary_cache,
                    alt_cache=alt_cache,
                    overwrite_existing=args.no_cache,
                )
                env = zaic_tools.ToolEnv(args.split, sample["id"], root=data_root)
                calls, state_hash = agent.run_sample(hydrated, env, asr_alt=asr_alt)

                rec = {
                    "id": sample["id"],
                    "tool_calls": calls,
                    "state_hash": state_hash,
                }
                completed[sample["id"]] = rec
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                if idx % 20 == 0 or idx == len(pending):
                    elapsed = time.time() - t0
                    rate = elapsed / idx
                    eta = rate * (len(pending) - idx) / 60.0
                    print(
                        f"  [{len(completed)}/{len(rows)}] {rate:.2f}s/sample | "
                        f"elapsed={elapsed/60:.1f}m | eta={eta:.1f}m"
                    )
                    if args.mirror_path:
                        args.mirror_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(args.output, args.mirror_path)

    # Rewrite in deterministic split order cleanly once complete.
    ordered_rows = [completed[r["id"]] for r in rows if r["id"] in completed]
    with args.output.open("w", encoding="utf-8") as fout:
        for rec in ordered_rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.mirror_path:
        args.mirror_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.mirror_path)

    print(f"[Saved] {len(ordered_rows)} rows -> {args.output}")

    # Run schema validation on every full non-filtered run, or validate calls directly.
    if args.limit is None and args.ids_file is None:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "zaic_tools",
            "--split",
            args.split,
            "--root",
            str(data_root),
            "--validate",
            str(args.output),
        ]
        env_vars = os.environ.copy()
        env_vars["PYTHONPATH"] = str(split_dir) + (os.pathsep + env_vars["PYTHONPATH"] if env_vars.get("PYTHONPATH") else "")
        proc = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
        print(f"[zaic_tools --validate] exit={proc.returncode}")
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip())
        if proc.returncode != 0:
            sys.exit(proc.returncode)

    if args.eval and rows and "label" in rows[0]:
        import evaluate

        summary = evaluate.evaluate_predictions(
            data_root=data_root,
            split=args.split,
            submission_path=args.output,
            limit=args.limit,
            sample_ids={r["id"] for r in rows} if args.ids_file else None,
            unseen_apps=unseen_apps,
        )
        print("=== LOCAL EVAL SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
