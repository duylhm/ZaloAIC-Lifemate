"""ASR Module for LifeMate — Runs vinai/PhoWhisper on CUDA GPU or CPU.
Offline capable, reads directly from zip or disk.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
import numpy as np
import torch
from scipy.io import wavfile
from transformers import pipeline

class LifeMateASR:
    def __init__(self, model_name: str = "vinai/PhoWhisper-large", device: str | None = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[ASR] Loading {model_name} on {self.device}...")
        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
            torch_dtype=torch_dtype
        )
        print(f"[ASR] Model loaded ready.")

    def transcribe_audio_bytes(self, raw_bytes: bytes) -> str:
        sr, audio = wavfile.read(io.BytesIO(raw_bytes))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        res = self.pipe({"raw": audio, "sampling_rate": sr})
        return res.get("text", "").strip()

    def transcribe_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return self.transcribe_audio_bytes(f.read())

    def batch_transcribe_from_zip(
        self,
        zip_path: str,
        keys: list[str],
        cache_path: str | None = None
    ) -> dict[str, str]:
        cache: dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                print(f"[ASR] Loaded existing cache: {len(cache)} entries.")
            except Exception:
                cache = {}

        to_run = [k for k in keys if k not in cache]
        print(f"[ASR] Total keys: {len(keys)}, Cached: {len(cache)}, Need run: {len(to_run)}")

        if not to_run:
            return cache

        with zipfile.ZipFile(zip_path, "r") as zf:
            for idx, key in enumerate(to_run, 1):
                # Try path variants inside zip
                candidates = [
                    f"train+public_test/{key}",
                    key,
                    f"public_test/{key}"
                ]
                raw_bytes = None
                for c in candidates:
                    try:
                        raw_bytes = zf.read(c)
                        break
                    except KeyError:
                        continue

                if raw_bytes is None:
                    print(f"[ASR Warning] Key not found in zip: {key}")
                    cache[key] = ""
                    continue

                try:
                    text = self.transcribe_audio_bytes(raw_bytes)
                    cache[key] = text
                except Exception as e:
                    print(f"[ASR Error] Failed on {key}: {e}")
                    cache[key] = ""

                if idx % 20 == 0 or idx == len(to_run):
                    print(f"[ASR Progress] {idx}/{len(to_run)} done.")
                    if cache_path:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)

        return cache
