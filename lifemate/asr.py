"""Decode current AND historical user audio; no public transcript caches."""
from __future__ import annotations

import math
import time

from .data import audio_path, public_input
from .models import resolve_model


def waveform(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != 16000:
        common = math.gcd(sr, 16000)
        samples = resample_poly(samples, 16000 // common, sr // common)
    return np.ascontiguousarray(samples, dtype=np.float32)


class Gipformer:
    def __init__(self, model="g-group-ai-lab/gipformer1.5-68M-rnnt", revision=None, offline=False):
        from pathlib import Path
        import sherpa_onnx
        self.path = resolve_model(model, revision, offline,
                                  ["*.int8.onnx", "tokens.txt", "config.json", "README.md"])
        base = Path(self.path)
        required = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        for name in required:
            if not (base / name).is_file():
                raise FileNotFoundError(base / name)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(base / required[0]), decoder=str(base / required[1]),
            joiner=str(base / required[2]), tokens=str(base / required[3]),
            num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu")

    def transcribe(self, path):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, waveform(path))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class PhoWhisper:
    def __init__(self, model="vinai/PhoWhisper-large", revision=None, offline=False):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.path = resolve_model(model, revision, offline)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
        # CPU fallback avoids ASR+LLM VRAM pressure; benchmark its latency before choosing it.
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.path, local_files_only=True,
                                                              dtype=torch.float32).eval()

    def transcribe(self, path):
        samples = waveform(path)
        texts = []
        # Whisper's 30 second window: never silently drop audio after the first window.
        for start in range(0, len(samples), 30 * 16000):
            inputs = self.processor(samples[start:start + 30 * 16000], sampling_rate=16000, return_tensors="pt")
            with self.torch.inference_mode():
                tokens = self.model.generate(inputs.input_features, language="vi", task="transcribe", do_sample=False)
            texts.append(self.processor.batch_decode(tokens, skip_special_tokens=True)[0])
        return " ".join(texts).strip()


def hydrate(row, root, split, mode, asr, deadline):
    sample = public_input(row)
    pairs = [(sample, row)] + [(new, old) for new, old in zip(sample["history"], row.get("history", []))
                              if old.get("role") == "user"]
    for target, original in pairs:
        if mode == "gold":
            if split != "train":
                raise ValueError("Gold transcripts are only allowed for TRAIN diagnostics")
            if not isinstance(original.get("audio_text"), str):
                raise ValueError("Missing TRAIN gold transcript")
            target["text"] = original["audio_text"]
        else:
            if time.monotonic() >= deadline:
                raise TimeoutError("Deadline reached during ASR")
            if not original.get("audio"):
                raise ValueError("Missing user audio")
            target["text"] = asr.transcribe(audio_path(root, split, original["audio"]))
    return sample
