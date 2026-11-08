"""Zalo AI Challenge 2026 — LifeMate: Professional Inference Runner
Runs offline SLM (Qwen2.5-3B-Instruct 4-bit NF4) with deterministic guardrails,
ToolEnv replay state calculation, and auto-resume checkpointing.
"""
import os
import sys
import json
import glob
import time
import shutil
import argparse
from pathlib import Path

# Enable memory fragmentation avoidance in PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import rapidfuzz

def resolve_environment():
    """Auto-detect paths on Colab or Local machine."""
    is_colab = os.path.exists("/content")
    
    if is_colab:
        # Check Drive mount
        drive_dir = Path("/content/drive/MyDrive")
        if not drive_dir.exists():
            print("[Info] Google Drive not mounted at /content/drive/MyDrive.")
        
        # Check if train+public_test is unzipped
        data_root = Path("/content/data/train+public_test")
        if not data_root.exists():
            # Find zip file on Drive
            zips = glob.glob("/content/drive/MyDrive/**/train+public_test.zip", recursive=True)
            if not zips:
                zips = glob.glob("/content/drive/MyDrive/*train+public_test*.zip")
            if zips:
                zip_path = zips[0]
                print(f"[Setup] Unzipping {zip_path} to /content/data...")
                os.makedirs("/content/data", exist_ok=True)
                os.system(f'unzip -q "{zip_path}" -d /content/data')
            else:
                # Look in current directory or /content
                local_zips = glob.glob("/content/*train+public_test*.zip")
                if local_zips:
                    os.system(f'unzip -q "{local_zips[0]}" -d /content/data')
        
        test_dir = data_root / "public_test" if data_root.exists() else Path("/content/data/public_test")
        output_path = Path("/content/submission.jsonl")
        drive_output = drive_dir / "submission.jsonl" if drive_dir.exists() else None
    else:
        # Local Windows/Linux path
        base = Path(__file__).resolve().parent
        data_root = base / "data" / "train+public_test"
        test_dir = base / "data" / "public_test" if not data_root.exists() else data_root / "public_test"
        output_path = base / "submission.jsonl"
        drive_output = None

    return {
        "is_colab": is_colab,
        "data_root": data_root,
        "test_dir": test_dir,
        "output_path": output_path,
        "drive_output": drive_output
    }

def load_asr_cache(base_dir: Path, drive_dir: Path | None) -> dict:
    """Load pre-computed ASR transcriptions to completely bypass audio model."""
    candidates = [
        base_dir / "asr_public_test_pho.json",
        Path("/content/ZaloAIC-Lifemate/asr_public_test_pho.json"),
        Path("/content/data/asr_public_test_pho.json"),
    ]
    if drive_dir and drive_dir.exists():
        candidates.append(drive_dir / "asr_cache_public_test.json")

    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                print(f"[ASR] Loaded {len(cache)} cached audio transcriptions from {p.name}")
                return cache
            except Exception as e:
                print(f"[ASR] Warning reading {p}: {e}")
    return {}

def filter_catalog(catalog_dict: dict, installed_apps: set) -> list[dict]:
    """Filter catalog to only tools belonging to installed apps or sys."""
    filtered = []
    for app in catalog_dict.get("apps", []):
        app_name = app.get("name", "")
        if app_name in installed_apps or app_name == "sys":
            for tool in app.get("tools", []):
                filtered.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                })
    return filtered

def build_system_prompt(filtered_tools: list[dict], current_state: dict) -> str:
    tools_str = json.dumps(filtered_tools, ensure_ascii=False, indent=1)
    compact_state = {k: v for k, v in current_state.items() if v}
    state_str = json.dumps(compact_state, ensure_ascii=False)
    
    return f"""Bạn là trợ lý AI LifeMate trong Zalo AI Challenge 2026.
Nhiệm vụ: Phân tích câu nói của người dùng và gọi các công cụ (tool_calls) phù hợp nhất.

CÔNG CỤ ĐƯỢC PHÉP DÙNG:
{tools_str}

DATABASE STATE HIỆN TẠI CỦA THIẾT BỊ:
{state_str}

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. OUT-OF-SCOPE HOẶC HỦY LỆNH ("thôi khỏi", "đừng làm gì"): trả về duy nhất mảng rỗng []
2. CÂU HỎI ĐIỀU KIỆN THỜI TIẾT ("Nếu mai mưa thì... không mưa thì thôi"): CHỈ gọi duy nhất `weather.get_forecast`.
3. KHÔNG ĐƯỢC TỰ BỊA ID: Lấy chính xác ID sự kiện/hóa đơn từ DATABASE STATE. Tham số hóa đơn là `source` ("bank" hoặc "ewallet"). Tham số nhắc nhở là `content` và `time`.
4. ĐỊNH DẠNG ĐẦU RA: CHỈ trả về duy nhất một khối JSON array hợp lệ:
[
  {{"name": "tool.name", "arguments": {{"param": "val"}}}}
]"""

def build_user_prompt(sample: dict, audio_text: str) -> str:
    ctx = sample.get("context", {})
    curr_time = ctx.get("current_time", "")
    user_prof = ctx.get("user_profile") or {}
    history = sample.get("history", [])
    
    history_str = ""
    if history:
        turns = [f"- {h.get('role')}: {h.get('audio_text') or h.get('text') or ''}" for h in history]
        history_str = "\nLịch sử hội thoại:\n" + "\n".join(turns)
        
    return f"""Thời gian hiện tại: {curr_time}
User: {json.dumps(user_prof, ensure_ascii=False)}{history_str}
Người dùng vừa nói: "{audio_text}"
Hãy sinh JSON array các tool calls phù hợp:"""

def parse_json_calls(raw_output: str) -> list[dict]:
    import re
    text = raw_output.strip()
    m_code = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if m_code:
        text = m_code.group(1).strip()
    else:
        m_arr = re.search(r"(\[[\s\S]*\])", text)
        if m_arr:
            text = m_arr.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {"name": str(c["name"]).strip(), "arguments": c.get("arguments") or {}}
                for c in data if isinstance(c, dict) and "name" in c
            ]
    except Exception:
        pass
    return []

def apply_guardrails(call: dict, current_state: dict) -> dict:
    """Correct common parameter hallucinations and align with database state."""
    name = call.get("name", "")
    args = call.get("arguments") or {}

    # 1. Bill payment source guardrail
    if name == "bill_payment.pay":
        src = str(args.get("source", "")).lower()
        if "bank" in src:
            args["source"] = "bank"
        elif "wallet" in src:
            args["source"] = "ewallet"
        else:
            args["source"] = "bank"

    # 2. Fuzzy match IDs in database state
    if "event_id" in args:
        events = current_state.get("calendar", {}).get("events", [])
        real_ids = [str(e.get("id")) for e in events if e.get("id")]
        if real_ids and str(args["event_id"]) not in real_ids:
            match = rapidfuzz.process.extractOne(str(args["event_id"]), real_ids)
            if match and match[1] >= 60:
                args["event_id"] = match[0]

    if "bill_id" in args:
        bills = current_state.get("bill_payment", {}).get("bills", [])
        real_ids = [str(b.get("id")) for b in bills if b.get("id")]
        if real_ids and str(args["bill_id"]) not in real_ids:
            match = rapidfuzz.process.extractOne(str(args["bill_id"]), real_ids)
            if match and match[1] >= 60:
                args["bill_id"] = match[0]

    call["arguments"] = args
    return call

def main():
    parser = argparse.ArgumentParser(description="Run LifeMate Offline Inference")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    env_info = resolve_environment()
    test_dir = env_info["test_dir"]
    data_root = env_info["data_root"]
    output_path = env_info["output_path"]
    drive_output = env_info["drive_output"]

    print("=" * 60)
    print("🚀 LifeMate Pro Inference Pipeline (Qwen2.5-3B 4-bit NF4)")
    print(f"Data directory:   {test_dir}")
    print(f"Output target:    {output_path}")
    if drive_output:
        print(f"Drive backup:     {drive_output}")
    print("=" * 60)

    # Add test_dir to sys.path for zaic_tools
    sys.path.insert(0, str(test_dir))
    try:
        from zaic_tools import ToolEnv, schema_validate
    except ImportError:
        print(f"[Error] Cannot import zaic_tools from {test_dir}. Make sure data is unzipped!")
        sys.exit(1)

    # 1. Load test samples
    test_jsonl = test_dir / "public_test.jsonl"
    with open(test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"[Data] Loaded {len(samples)} public test samples.")

    # 2. Load catalog
    catalog_path = test_dir / "public_test_tools_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_dict = json.load(f)

    # 3. Load pre-computed ASR cache (0 GPU memory, 0 wait time)
    base_dir = Path(__file__).resolve().parent
    drive_dir = Path("/content/drive/MyDrive") if env_info["is_colab"] else None
    asr_cache = load_asr_cache(base_dir, drive_dir)

    # 4. Resume check: read existing progress
    results = []
    done_ids = set()
    source_to_check = drive_output if (drive_output and drive_output.exists()) else output_path
    if source_to_check.exists():
        try:
            with open(source_to_check, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        results.append(row)
                        done_ids.add(row["id"])
            print(f"[Checkpoint] Found existing progress! Already completed: {len(done_ids)}/{len(samples)} samples.")
        except Exception as e:
            print(f"[Checkpoint] Error reading checkpoint: {e}")
            results, done_ids = [], set()

    remaining_samples = [s for s in samples if s["id"] not in done_ids]
    print(f"[Plan] Samples remaining to process: {len(remaining_samples)}")

    if not remaining_samples:
        print("🎉 All samples already completed! Nothing to run.")
        if drive_output and not drive_output.exists():
            shutil.copy(str(output_path), str(drive_output))
        sys.exit(0)

    # 5. Load Qwen2.5-3B in 4-bit NF4
    print(f"\n[Model] Loading {args.model_id} in 4-bit NF4 (Occupies only ~2.1 GB VRAM)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[Model] Ready! GPU VRAM in use: {vram_gb:.2f} GB (Over 12.5 GB free).\n")

    # 6. Run Inference Loop
    start_time = time.time()
    with open(output_path, "w", encoding="utf-8") as f_out:
        # Re-write finished samples
        for row in results:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()

        for idx, s in enumerate(samples, 1):
            if s["id"] in done_ids:
                continue

            rel = s.get("audio", "")
            text = asr_cache.get(f"public_test/{rel}", "")
            
            env = ToolEnv("public_test", s["id"], root=data_root)

            t_low = text.lower()
            oos_words = ["khách sạn", "hotel", "homestay", "resort", "hộ chiếu", "visa", "bảo hiểm", "karaoke", "spa", "massage"]
            oos = any(w in t_low for w in oos_words)
            cancel = any(w in t_low for w in ["thôi khỏi", "đừng làm gì", "không cần nữa"]) and not any(w in t_low for w in ["chỉ cần xem", "xem giúp"])

            if not oos and not cancel:
                installed = set(s.get("context", {}).get("installed_apps", {}).keys())
                filtered = filter_catalog(catalog_dict, installed)
                sys_p = build_system_prompt(filtered, env.get_state())
                usr_p = build_user_prompt(s, text)

                messages = [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": usr_p}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
                    resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                del inputs, out_ids
                pred_calls = parse_json_calls(resp_text)

                for c in pred_calls:
                    c = apply_guardrails(c, env.get_state())
                    val = env.validate(c["name"], c.get("arguments") or {})
                    if val.get("ok"):
                        env.call(c["name"], c.get("arguments") or {})

            row = {
                "id": s["id"],
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash()
            }
            results.append(row)
            done_ids.add(s["id"])
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            if len(done_ids) % 20 == 0 or len(done_ids) == len(samples):
                torch.cuda.empty_cache()
                if drive_output:
                    try:
                        shutil.copy(str(output_path), str(drive_output))
                    except Exception:
                        pass
                elapsed = time.time() - start_time
                print(f"[Progress] [{len(done_ids)}/{len(samples)}] samples | Elapsed: {elapsed/60:.1f}m")

    # Final sync
    if drive_output:
        shutil.copy(str(output_path), str(drive_output))
        print(f"\n[Done] Backup synced to Google Drive: {drive_output}")

    print(f"[Success] All {len(samples)} samples processed successfully -> {output_path}")

if __name__ == "__main__":
    main()
