"""Offline Vietnamese ASR loader (Transformers / faster-whisper) with 16kHz resampling & dual-cache support."""

from __future__ import annotations

import copy
import json
import wave
from pathlib import Path
from typing import Any


def load_asr_cache(path: Path | None) -> dict[str, str]:
    """Load JSON `{audio_rel_path: transcript}` cache if present."""
    if not path or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_asr_cache(path: Path, cache: dict[str, str]) -> None:
    """Persist ASR cache atomically to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def lookup_cached_transcript(
    cache: dict[str, str] | None,
    rel_audio: str | None,
    split_name: str,
) -> str | None:
    """Resolve transcript against cache checking both `<split>/<rel>` and `<rel>` keys."""
    if not cache or not rel_audio:
        return None
    norm = rel_audio.replace("\\", "/").lstrip("/")
    for key in (f"{split_name}/{norm}", norm):
        if key in cache and cache[key]:
            return cache[key].strip()
    return None


def load_audio_16k_mono(wav_path: Path):
    """Load audio file as float32 16kHz mono NumPy array."""
    import numpy as np

    audio = None
    sr = 16000
    try:
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            raw = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        audio = raw.reshape(-1, n_channels) if n_channels > 1 else raw

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    if sr != 16000 and len(audio) > 0:
        try:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
        except Exception:
            target_len = int(round(len(audio) * 16000.0 / float(sr)))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=max(1, target_len), endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


class OfflineASR:
    """Unified offline ASR wrapper supporting HuggingFace Transformers and faster-whisper."""

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-large",
        backend: str = "auto",
        num_beams: int = 3,
        local_files_only: bool = False,
    ) -> None:
        import torch

        self.model_id = model_id
        self.num_beams = num_beams
        self.local_files_only = local_files_only
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if backend == "auto":
            use_fw = "faster-whisper" in model_id.lower() or model_id.lower().startswith("systran/")
        else:
            use_fw = backend == "faster-whisper"

        self.backend_type = "faster-whisper" if use_fw else "transformers"

        if use_fw:
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "int8"
            self.fw_model = WhisperModel(
                model_id,
                device=self.device,
                compute_type=compute_type,
                local_files_only=local_files_only,
            )
            self.pipe = None
        else:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ).to(self.device)
            model.eval()
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=0 if self.device == "cuda" else -1,
            )
            self.fw_model = None

    def transcribe_file(self, wav_path: Path) -> str:
        """Transcribe a single .wav file to Vietnamese text at 16kHz mono."""
        audio = load_audio_16k_mono(wav_path)
        if self.backend_type == "faster-whisper":
            segments, _ = self.fw_model.transcribe(
                audio,
                language="vi",
                beam_size=self.num_beams,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        res = self.pipe(
            {"array": audio, "sampling_rate": 16000},
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "num_beams": self.num_beams,
            },
        )
        return (res.get("text") or "").strip()


def hydrate_sample_transcripts(
    sample: dict,
    split_dir: Path,
    split_name: str,
    asr_engine: OfflineASR | None = None,
    primary_cache: dict[str, str] | None = None,
    alt_cache: dict[str, str] | None = None,
    overwrite_existing: bool = False,
) -> tuple[dict, str | None]:
    """Populate missing `audio_text` on request + history user turns via cache or live ASR."""
    out = copy.deepcopy(sample)

    def resolve_text(rel_audio: str | None, existing: str | None) -> str:
        if existing and not overwrite_existing:
            return existing.strip()
        cached = lookup_cached_transcript(primary_cache, rel_audio, split_name)
        if cached is not None:
            return cached
        if asr_engine is not None and rel_audio:
            wav_path = split_dir / rel_audio
            if wav_path.is_file():
                text = asr_engine.transcribe_file(wav_path)
                if primary_cache is not None:
                    norm_audio = rel_audio.replace("\\", "/").lstrip("/")
                    key = f"{split_name}/{norm_audio}"
                    primary_cache[key] = text
                return text
        return (existing or "").strip()

    out["audio_text"] = resolve_text(out.get("audio"), out.get("audio_text"))
    for turn in out.get("history") or []:
        if turn.get("role") == "user":
            turn["audio_text"] = resolve_text(turn.get("audio"), turn.get("audio_text"))

    asr_alt = lookup_cached_transcript(alt_cache, out.get("audio"), split_name)
    if asr_alt and asr_alt.strip().lower() == out["audio_text"].strip().lower():
        asr_alt = None
    return out, asr_alt
