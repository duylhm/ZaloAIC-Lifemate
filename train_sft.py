"""Zalo AI Challenge 2026 — LifeMate: High-Performance QLoRA SFT Training Script
Fine-tunes Qwen2.5-3B-Instruct on all 1,545 dialogues from train.jsonl using 4-bit QLoRA.
Trains in ~15-20 minutes on a single Colab T4 GPU, saves LoRA adapter to Google Drive.
"""
import os
import sys
import json
import glob
import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def resolve_data_paths():
    """Auto-detect train data directory on Colab or Local."""
    is_colab = os.path.exists("/content")
    if is_colab:
        candidates = [
            Path("/content/data/train+public_test/train"),
            Path("/content/data/train"),
            Path("/content/train+public_test/train"),
            Path("/content/train")
        ]
        train_dir = None
        for c in candidates:
            if (c / "train.jsonl").exists():
                train_dir = c
                break

        if train_dir is None:
            # Check if train+public_test is unzipped
            zips = glob.glob("/content/drive/MyDrive/**/train+public_test.zip", recursive=True)
            if not zips:
                zips = glob.glob("/content/drive/MyDrive/*train+public_test*.zip")
            if not zips:
                zips = glob.glob("/content/*train+public_test*.zip")
            if zips:
                print(f"[Setup] Unzipping {zips[0]} to /content/data...")
                os.makedirs("/content/data", exist_ok=True)
                os.system(f'unzip -q "{zips[0]}" -d /content/data')
            for c in candidates:
                if (c / "train.jsonl").exists():
                    train_dir = c
                    break

        if train_dir is None:
            raise FileNotFoundError("Không tìm thấy thư mục train (train.jsonl). Vui lòng đảm bảo file zip đã có trên Drive hoặc /content!")

        output_dir = Path("/content/drive/MyDrive/qwen2.5_3b_lifemate_lora")
    else:
        base = Path(__file__).resolve().parent
        downloads = Path(os.path.expanduser("~")) / "Downloads" / "train+public_test" / "train+public_test" / "train"
        train_dir = downloads if downloads.exists() else base / "data" / "train"
        output_dir = base / "qwen2.5_3b_lifemate_lora"

    return train_dir, output_dir

def filter_catalog(catalog_dict: dict, installed_apps: set) -> list[dict]:
    filtered = []
    for app in catalog_dict.get("apps", []):
        if app.get("name") in installed_apps or app.get("name") == "sys":
            for tool in app.get("tools", []):
                filtered.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                })
    return filtered

def prepare_sft_dataset(train_dir: Path, tokenizer, max_length: int = 2048):
    train_jsonl = train_dir / "train.jsonl"
    catalog_path = train_dir / "train_tools_catalog.json"
    env_dir = train_dir / "train_environment"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    with open(train_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"[Dataset] Loading and formatting {len(samples)} dialogues from {train_jsonl.name}...")
    
    input_ids_list = []
    labels_list = []

    for idx, s in enumerate(samples, 1):
        s_id = s["id"]
        ctx = s.get("context", {})
        installed = set(ctx.get("installed_apps", {}).keys())
        curr_time = ctx.get("current_time", "")
        user_prof = ctx.get("user_profile") or {}
        history = s.get("history", [])
        audio_text = s.get("audio_text") or s.get("text", "")
        gold_calls = s.get("label", {}).get("tool_calls", [])

        # Load initial DB state
        state_file = env_dir / s_id / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                db_state = json.load(f)
        else:
            db_state = {}

        # 1. Build Prompts
        filtered_tools = filter_catalog(catalog, installed)
        compact_state = {k: v for k, v in db_state.items() if v and k != "_meta"}
        
        sys_prompt = f"""Bạn là trợ lý AI LifeMate trong Zalo AI Challenge 2026.
Nhiệm vụ: Phân tích câu nói của người dùng và gọi các công cụ (tool_calls) phù hợp nhất.

CÔNG CỤ ĐƯỢC PHÉP DÙNG:
{json.dumps(filtered_tools, ensure_ascii=False, indent=1)}

DATABASE STATE HIỆN TẠI CỦA THIẾT BỊ:
{json.dumps(compact_state, ensure_ascii=False)}

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. OUT-OF-SCOPE HOẶC HỦY LỆNH ("thôi khỏi", "đừng làm gì"): trả về duy nhất mảng rỗng []
2. CÂU HỎI ĐIỀU KIỆN THỜI TIẾT ("Nếu mai mưa thì... không mưa thì thôi"): CHỈ gọi duy nhất `weather.get_forecast`.
3. KHÔNG ĐƯỢC TỰ BỊA ID: Lấy chính xác ID sự kiện/hóa đơn từ DATABASE STATE. Tham số hóa đơn là `source` ("bank" hoặc "ewallet").
4. ĐỊNH DẠNG ĐẦU RA: CHỈ trả về duy nhất một khối JSON array hợp lệ:
[
  {{"name": "tool.name", "arguments": {{"param": "val"}}}}
]"""

        history_str = ""
        if history:
            turns = [f"- {h.get('role')}: {h.get('audio_text') or h.get('text') or ''}" for h in history]
            history_str = "\nLịch sử hội thoại:\n" + "\n".join(turns)

        user_prompt = f"Thời gian hiện tại: {curr_time}\nUser: {json.dumps(user_prof, ensure_ascii=False)}{history_str}\nNgười dùng vừa nói: \"{audio_text}\"\nHãy sinh JSON array các tool calls phù hợp:"

        assistant_resp = json.dumps(gold_calls, ensure_ascii=False)

        # 2. Format with ChatML template
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_resp}
        ]

        # Tokenize prompt only (for label masking)
        prompt_only_msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_only_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        # Mask prompt tokens with -100 so loss is ONLY computed on tool calls
        labels = list(full_ids)
        mask_len = min(len(prompt_ids), len(full_ids))
        for i in range(mask_len):
            labels[i] = -100

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list
    })

def main():
    train_dir, output_dir = resolve_data_paths()
    print("=" * 60)
    print("🚀 LifeMate QLoRA SFT Training Pipeline (Qwen2.5-3B)")
    print(f"Train directory:  {train_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model 4-bit NF4
    print(f"[Model] Loading base model {model_id} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. LoRA Config (All projection matrices)
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

    # 4. Prepare Dataset (All 1,545 dialogues from train.jsonl)
    train_data = prepare_sft_dataset(train_dir, tokenizer, max_length=2048)
    print(f"[Dataset] Full training set: {len(train_data)} dialogues.")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # Effective batch size = 8
        learning_rate=2e-4,
        num_train_epochs=2,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 6. Train
    print("\n[Train] Starting QLoRA fine-tuning (~15-20 minutes on Colab T4)...")
    trainer.train()

    # 7. Save Final Adapter
    print(f"\n[Save] Saving trained LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("🎉 Fine-tuning completed successfully! Adapter is ready for inference.")

if __name__ == "__main__":
    main()
