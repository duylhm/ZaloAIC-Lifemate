"""QLoRA Multi-Turn Causal-DAG SFT trainer (App-Dropout + Prompt Loss Masking, Colab T4 safe)."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path

from src.data_utils import REPO_ROOT, read_jsonl, resolve_data_root
from src.prompt_builder import load_catalog_dict
from src.trajectory import (
    inject_safe_phonetic_noise,
    sample_dropout_apps,
    sample_to_chatml,
    tokenize_chatml_with_masking,
)


def build_sft_conversations(
    train_rows: list[dict],
    catalog_tools: dict[str, dict],
    holdout_apps: set[str],
    aug_copies: int = 1,
    max_drop_ratio: float = 0.7,
    asr_noise_prob: float = 0.25,
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """Generate multi-turn ChatML conversations with clean + App-Dropout/ASR-noise views."""
    rng = random.Random(seed)
    conversations: list[list[dict[str, str]]] = []

    for row in train_rows:
        # Clean view (still masks any holdout_apps in val mode, slight shuffle for robustness).
        conversations.append(
            sample_to_chatml(
                row,
                catalog_tools,
                drop_apps=set(holdout_apps) if holdout_apps else None,
                shuffle_rng=random.Random(rng.randint(0, 10**9)),
            )
        )

        # Augmented views: App-Dropout + schema shuffle + safe verb phonetic noise.
        for _ in range(aug_copies):
            noisy_row = inject_safe_phonetic_noise(row, rng, prob=asr_noise_prob)
            dyn_drop = sample_dropout_apps(noisy_row, rng, max_drop_ratio=max_drop_ratio)
            drop_apps = set(holdout_apps) | dyn_drop
            conversations.append(
                sample_to_chatml(
                    noisy_row,
                    catalog_tools,
                    drop_apps=drop_apps,
                    shuffle_rng=random.Random(rng.randint(0, 10**9)),
                )
            )

    rng.shuffle(conversations)
    return conversations


def _make_collate_fn(pad_token_id: int):
    import torch

    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune <5B SLM for LifeMate with QLoRA + loss masking")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="Base HF model (<5B params)")
    ap.add_argument("--split-mode", choices=["val", "full"], default="val")
    ap.add_argument("--splits-file", type=Path, default=REPO_ROOT / "splits" / "val_all.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "lifemate-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--max-drop-ratio", type=float, default=0.7)
    ap.add_argument("--asr-noise-prob", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Build & inspect SFT data without GPU training")
    args = ap.parse_args()

    data_root = resolve_data_root(args.root, split="train")
    catalog_tools = load_catalog_dict(data_root / "train" / "train_tools_catalog.json")
    all_rows = read_jsonl(data_root / "train" / "train.jsonl")

    holdout_apps: set[str] = set()
    if args.split_mode == "val":
        if not args.splits_file.is_file():
            raise FileNotFoundError(f"Missing splits file: {args.splits_file}. Run scripts/prepare_splits.py first.")
        sp = json.loads(args.splits_file.read_text(encoding="utf-8"))
        train_id_set = set(sp["train_ids"])
        holdout_apps = set(sp.get("holdout_apps") or [])
        selected_rows = [r for r in all_rows if r["id"] in train_id_set]
    else:
        selected_rows = all_rows

    conversations = build_sft_conversations(
        selected_rows,
        catalog_tools,
        holdout_apps=holdout_apps,
        aug_copies=args.aug_copies,
        max_drop_ratio=args.max_drop_ratio,
        asr_noise_prob=args.asr_noise_prob,
        seed=args.seed,
    )
    print(
        f"[SFT Data] split_mode={args.split_mode} base_samples={len(selected_rows)} "
        f"conversations={len(conversations)} holdout_apps={sorted(holdout_apps)}"
    )

    if args.dry_run:
        char_lens = [sum(len(m["content"]) for m in conv) for conv in conversations]
        turn_lens = [len(conv) for conv in conversations]
        print(
            f"[Dry Run] avg_chars={sum(char_lens)/len(char_lens):.0f} "
            f"max_chars={max(char_lens)} avg_turns={sum(turn_lens)/len(turn_lens):.2f}"
        )
        print("[Dry Run] sample[0] roles:", [m["role"] for m in conversations[0]])
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized: list[dict[str, list[int]]] = []
    skipped_masked = 0
    for conv in conversations:
        item = tokenize_chatml_with_masking(tokenizer, conv, max_length=args.max_seq_len)
        if any(lbl != -100 for lbl in item["labels"]):
            tokenized.append(item)
        else:
            skipped_masked += 1

    tok_lens = [len(x["input_ids"]) for x in tokenized]
    sup_toks = sum(sum(1 for l in x["labels"] if l != -100) for x in tokenized)
    print(
        f"[Tokenized] kept={len(tokenized)} skipped={skipped_masked} "
        f"max_tokens={max(tok_lens)} avg_tokens={sum(tok_lens)/len(tok_lens):.0f} "
        f"supervised_tokens={sup_toks}"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    steps_per_epoch = max(1, math.ceil(len(tokenized) / max(1, args.batch_size * args.grad_accum)))
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = max(5, int(0.05 * total_steps))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ta_kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "cosine",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    valid_ta_params = inspect.signature(TrainingArguments.__init__).parameters
    filtered_ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_ta_params}
    training_args = TrainingArguments(**filtered_ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": _make_collate_fn(tokenizer.pad_token_id),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    meta = {
        "base_model": args.model,
        "split_mode": args.split_mode,
        "train_samples": len(selected_rows),
        "conversations": len(tokenized),
        "holdout_apps": sorted(holdout_apps),
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (args.output_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Done] Saved LoRA adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
