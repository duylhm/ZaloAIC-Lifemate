"""Local Hugging Face generation and pinned snapshot resolution."""
from __future__ import annotations

import os
import time
from pathlib import Path


def compute_dtype(torch):
    # is_bf16_supported() may include emulation on pre-Ampere GPUs.
    native = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native else torch.float16


def resolve_model(model, revision=None, offline=False, allow_patterns=None):
    if Path(model).is_dir():
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision, local_files_only=offline, allow_patterns=allow_patterns)


class HFBackend:
    def __init__(self, model, revision=None, offline=False, quantize=True,
                 max_new_tokens=384, max_context=16384, seed=42):
        import random
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for the language model. Enable Kaggle GPU before running this cell.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.torch = torch
        self.max_new_tokens, self.max_context = max_new_tokens, max_context
        if not 0 < max_new_tokens < max_context <= 16384:
            raise ValueError("Require 0 < max_new_tokens < max_context <= 16384")
        self.path = resolve_model(model, revision, offline)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        dtype = compute_dtype(torch)
        options = dict(local_files_only=True, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
        if quantize:
            options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        if config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForCausalLM
            # Official text-only class handles the language_model checkpoint prefix.
            self.model = Qwen3_5ForCausalLM.from_pretrained(self.path, **options)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.path, **options)
        self.model.eval()
        print(f"Language model loaded: {self.path}; dtype={dtype}; 4bit={quantize}", flush=True)

    def generate(self, messages, deadline):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        length = inputs.input_ids.shape[1]
        if length + self.max_new_tokens > self.max_context:
            raise ValueError(f"Context overflow: {length}+{self.max_new_tokens}>{self.max_context}; no silent truncation")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Sample deadline reached before generation")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                max_time=remaining, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, length:], skip_special_tokens=True)


class NoopBackend:
    """Infrastructure smoke only; never a model benchmark."""
    path = "noop"

    def generate(self, messages, deadline):
        return '{"tool_calls":[]}'


def offline_environment():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
