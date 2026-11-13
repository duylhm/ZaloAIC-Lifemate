"""Benchmark offline Vietnamese ASR models on `train` audio & optionally regenerate `public_test` cache."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import unicodedata
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.asr import OfflineASR, load_asr_cache, save_asr_cache  # noqa: E402
from src.data_utils import read_jsonl, resolve_data_root  # noqa: E402


def _normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().strip()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(cleaned.split())


def _edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer_wer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Compute micro-averaged CER (%) and WER (%) on normalized strings."""
    total_chars = 0
    char_errs = 0
    total_words = 0
    word_errs = 0
    for ref, hyp in zip(refs, hyps):
        rn = _normalize_text(ref)
        hn = _normalize_text(hyp)
        total_chars += max(1, len(rn))
        char_errs += _edit_distance(rn, hn)
        rw = rn.split()
        hw = hn.split()
        total_words += max(1, len(rw))
        word_errs += _edit_distance(rw, hw)
    return (100.0 * char_errs / total_chars, 100.0 * word_errs / total_words)


def inspect_wav_metadata(split_dir: Path, rows: list[dict], limit: int) -> dict:
    sample_rates: dict[int, int] = {}
    channels: dict[int, int] = {}
    checked = 0
    for r in rows[:limit]:
        rel = r.get("audio")
        if not rel:
            continue
        p = split_dir / rel
        if not p.is_file():
            continue
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            channels[ch] = channels.get(ch, 0) + 1
            checked += 1
        except Exception:
            pass
    return {"checked_wavs": checked, "sample_rates": sample_rates, "channels": channels}


def benchmark_models(
    data_root: Path,
    models: list[str],
    limit: int,
    num_beams: int,
) -> list[dict]:
    split_dir = data_root / "train"
    rows = read_jsonl(split_dir / "train.jsonl")
    print("[Audio Metadata]", json.dumps(inspect_wav_metadata(split_dir, rows, limit), ensure_ascii=False))

    targets: list[tuple[Path, str, str]] = []
    for r in rows:
        rel = r.get("audio")
        ref = r.get("audio_text", "")
        wav_path = (split_dir / rel) if rel else None
        if wav_path and wav_path.is_file() and ref:
            targets.append((wav_path, ref, r["id"]))
            if len(targets) >= limit:
                break

    if not targets:
        raise FileNotFoundError(
            f"No .wav files found under {split_dir}. Run this benchmark on Colab where audio files are mounted."
        )

    reports = []
    for model_id in models:
        print(f"\n[Benchmark] Loading {model_id} (num_beams={num_beams})...")
        asr = OfflineASR(model_id=model_id, num_beams=num_beams)
        refs: list[str] = []
        hyps: list[str] = []
        t0 = time.time()
        for wav_path, ref, _qid in targets:
            refs.append(ref)
            hyps.append(asr.transcribe_file(wav_path))
        dt = time.time() - t0
        cer, wer = compute_cer_wer(refs, hyps)
        rep = {
            "model": model_id,
            "samples": len(targets),
            "cer_pct": round(cer, 2),
            "wer_pct": round(wer, 2),
            "avg_sec_per_audio": round(dt / len(targets), 3),
        }
        print("[Benchmark Result]", json.dumps(rep, ensure_ascii=False))
        for i in range(min(3, len(targets))):
            print(f"  [{targets[i][2]}] REF: {refs[i]}")
            print(f"  [{targets[i][2]}] HYP: {hyps[i]}")
        reports.append(rep)

        del asr
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return reports


def regenerate_split_cache(
    data_root: Path,
    split: str,
    model_id: str,
    out_cache: Path,
    num_beams: int,
    force: bool = False,
) -> None:
    split_dir = data_root / split
    rows = read_jsonl(split_dir / f"{split}.jsonl")
    cache = {} if force else load_asr_cache(out_cache)

    audio_keys: list[tuple[str, Path]] = []
    seen = set()
    for r in rows:
        for rel in [r.get("audio")] + [t.get("audio") for t in (r.get("history") or [])]:
            if not rel:
                continue
            norm = rel.replace("\\", "/").lstrip("/")
            key = f"{split}/{norm}"
            if key in seen:
                continue
            seen.add(key)
            if not force and key in cache and cache[key]:
                continue
            wav_p = split_dir / norm
            if wav_p.is_file():
                audio_keys.append((key, wav_p))

    print(f"[Regen] Audio files pending transcription for `{split}`: {len(audio_keys)} (cached={len(cache)})")
    if not audio_keys:
        save_asr_cache(out_cache, cache)
        return

    asr = OfflineASR(model_id=model_id, num_beams=num_beams)
    t0 = time.time()
    for idx, (key, wav_p) in enumerate(audio_keys, start=1):
        cache[key] = asr.transcribe_file(wav_p)
        if idx % 50 == 0 or idx == len(audio_keys):
            save_asr_cache(out_cache, cache)
            elapsed = (time.time() - t0) / 60.0
            print(f"  [{idx}/{len(audio_keys)}] saved cache ({elapsed:.1f}m elapsed)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ASR models or regenerate split ASR cache")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=["vinai/PhoWhisper-large", "openai/whisper-large-v3-turbo"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--num-beams", type=int, default=3)
    ap.add_argument("--regen-split", default=None, help="If set (e.g. public_test), build full cache instead of benchmark")
    ap.add_argument("--regen-model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out-cache", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    target_split = args.regen_split or "train"
    data_root = resolve_data_root(args.root, split=target_split)

    if args.regen_split:
        out_path = args.out_cache or (Path(__file__).resolve().parent.parent / f"asr_{args.regen_split}_best.json")
        regenerate_split_cache(
            data_root=data_root,
            split=args.regen_split,
            model_id=args.regen_model,
            out_cache=out_path,
            num_beams=args.num_beams,
            force=args.force,
        )
    else:
        benchmark_models(data_root, args.models, limit=args.limit, num_beams=args.num_beams)


if __name__ == "__main__":
    main()
