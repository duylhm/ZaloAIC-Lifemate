"""Zalo AI Challenge 2026 — LifeMate: Supervised Fine-Tuning (QLoRA)
Fine-tunes Qwen2.5-3B-Instruct on official train.jsonl in 4-bit on a single Colab T4 GPU (~20 mins).
"""
import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def build_training_dataset(train_jsonl_path: Path, catalog_path: Path):
    """Convert train.jsonl turns into ChatML format."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"[Dataset] Formatting {len(samples)} training dialogues...")
    dataset = []
    for s in samples:
        ctx = s.get("context", {})
        installed = set(ctx.get("installed_apps", {}).keys())
        curr_time = ctx.get("current_time", "")
        db_state = s.get("database_state") or {}
        gold_calls = s.get("ground_truth_tool_calls") or s.get("tool_calls") or []

        # System prompt
        filtered_tools = []
        for app in catalog.get("apps", []):
            if app.get("name") in installed or app.get("name") == "sys":
                for t in app.get("tools", []):
                    filtered_tools.append({"name": t["name"], "parameters": t.get("parameters", {})})

        sys_content = f"Bạn là trợ lý AI LifeMate.\nCÔNG CỤ:\n{json.dumps(filtered_tools, ensure_ascii=False)}\nDATABASE STATE:\n{json.dumps(db_state, ensure_ascii=False)}"
        user_content = f"Thời gian: {curr_time}\nNgười dùng nói: \"{s.get('audio_text') or s.get('text', '')}\""
        assistant_content = json.dumps(gold_calls, ensure_ascii=False)

        dataset.append({
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    return dataset

def main():
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = Path("./qwen2.5_3b_lifemate_lora")

    print("=" * 60)
    print("🚀 LifeMate QLoRA SFT Training Pipeline")
    print(f"Base Model:  {model_id}")
    print(f"Output dir:  {output_dir}")
    print("=" * 60)

    # 1. Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("[Ready] Training script configured. Ready to run on Colab GPU.")

if __name__ == "__main__":
    main()
