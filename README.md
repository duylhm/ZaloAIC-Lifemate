# Zalo AI Challenge 2026 — LifeMate

Pipeline **Offline Voice-to-Action Agentic** (`SLM < 5B` + `Offline Vietnamese ASR`) tuân thủ 100% thể lệ BTC:

- **Không rò rỉ `state.json`:** Agent chỉ truy xuất trạng thái thiết bị qua các lượt gọi `read` tool trên `zaic_tools.ToolEnv`.
- **Guardrail kiểm tra trước khi gọi (`env.validate` → `env.call`):** Chặn và tự sửa tham số sai schema trước khi gọi `env.call(...)`, tránh làm bẩn `env.get_tool_calls()`, đồng thời trả `schema_validation_error` lại cho model tự sửa ở lượt kế tiếp.
- **Tổng quát hóa sang Unseen Apps:** Nén chữ ký tool có kiểu dữ liệu (`format_compact_tool`), phân rã chuỗi hành động theo Causal DAG, tăng cường **App-Dropout**, xáo trộn thứ tự tool và tính **Prompt Loss Masking** chỉ trên các lượt trả lời của assistant.

---

## Cấu trúc Thư mục

- `src/prompt_builder.py` — Chuẩn hóa `time_anchors`, giữ đầy đủ `history` (`calls` + `results`), hỗ trợ `asr_alt` và nén schema theo `installed_apps` + `sys.ask_missing_info`.
- `src/trajectory.py` — Gom `tool_calls` thành các bước nhân quả (`group_trajectory_into_steps`), App-Dropout, nhiễu âm vị tiếng Việt an toàn, chuyển ChatML và masking loss (`labels = -100`) trên prompt/tool results.
- `src/guardrails.py` — Lọc `<think>` / markdown fence, tách JSON, ép kiểu số nguyên tiếng Việt (`1.200.000` → `1200000`), chuẩn hóa `date` / `date-time` / `time` và bỏ tham số mặc định thừa.
- `src/agent.py` — Vòng lặp ReAct `LifeMateAgent` + backend `HFTransformersBackend` (4-bit NF4, LoRA, hỗ trợ Qwen3 `enable_thinking=False`), `OpenAICompatibleBackend`, `HeuristicBackend`, `NoopBackend`.
- `src/asr.py` — `OfflineASR` (`vinai/PhoWhisper-large` / `faster-whisper`) hỗ trợ resample 16kHz mono, nhận dạng cả lượt chính và lịch sử `_u*.wav`, kết hợp dual-cache.
- `src/data_utils.py` — Tự động dò thư mục dữ liệu trên máy local Windows / Google Colab và tiện ích đọc/ghi JSONL.
- `scripts/prepare_splits.py` — Tạo bộ kiểm thử nội bộ tại `splits/`: `val_unseen.json` (`142` mẫu thuộc 4 app ẩn: `bike_rental`, `exchange_rate`, `medication`, `package_tracking`), `val_indomain.json` (`100` mẫu phân tầng) và `val_all.json`.
- `scripts/benchmark_asr.py` — Benchmark CER/WER ASR tiếng Việt trên audio tập `train` và tái tạo cache `public_test` trên Colab T4.
- `scripts/inspect_errors.py` — Phân tích các ca lỗi theo từng nhóm bằng `evaluate.py`.
- `train_sft.py` — Huấn luyện QLoRA multi-turn SFT (`--split-mode val` và `--split-mode full`).
- `run_inference.py` — Suy luận ReAct có checkpoint/resume, tự chạy `zaic_tools --validate` và tùy chọn `--eval`.
- `docker/Dockerfile` & `docker/run.sh` — Đóng gói Docker chạy offline hoàn toàn (`HF_HUB_OFFLINE=1`, `--no-cache`) cho `private_test`.

---

## Quy trình Kiểm thử Tại Local (Windows PowerShell / CPU)

```powershell
python -m unittest tests/test_pipeline.py -v
python scripts/prepare_splits.py
python run_inference.py --split train --ids-file splits/val_all.json --backend noop --force --output outputs/val_noop.jsonl --eval
python run_inference.py --split train --ids-file splits/val_all.json --backend heuristic --force --output outputs/val_heuristic.jsonl --eval
```

---

## Quy trình Fine-tune & Suy luận Trên Google Colab (GPU T4)

Mở notebook [notebooks/LifeMate_Colab_Inference.ipynb](notebooks/LifeMate_Colab_Inference.ipynb) trên Google Colab (chọn Runtime **T4 GPU**), mount Google Drive chứa `train+public_test.zip` và chạy tuần tự các cell:

1. **Cell 1–2 (`setup-cell`, `config-cell`):** Clone/pull repo, cài thư viện, giải nén dữ liệu và chọn `MODEL_ID` (`Qwen/Qwen3-4B` hoặc `Qwen/Qwen2.5-3B-Instruct`).
2. **Cell 3 (`asr-bench-cell`, tùy chọn):** Đo CER/WER giữa `PhoWhisper-large` và `whisper-large-v3-turbo` trên 50 file `.wav` tập `train`.
3. **Cell 4–5 (`zeroshot-val-cell`, `train-eval-val-cell`):** Fine-tune QLoRA (`--split-mode val`) trên `1,303` mẫu, chấm điểm trên `142` mẫu Unseen-App + `100` mẫu In-Domain qua `evaluate.py` và soi lỗi với `inspect_errors.py`.
4. **Cell 6–7 (`full-train-infer-cell`, `download-cell`):** Fine-tune QLoRA (`--split-mode full`) trên toàn bộ `1,545` mẫu → infer `public_test` (`634` mẫu) → tự động backup sang Google Drive, kiểm tra `zaic_tools --validate` và tải `submission.jsonl` (hoặc chạy Cell 7 để tải lại nếu trình duyệt chặn popup).

---

## Quy trình Fine-tune & Suy luận Trên Kaggle (GPU T4 x2 / P100)

Mở/import notebook [notebooks/LifeMate_Kaggle_Inference.ipynb](notebooks/LifeMate_Kaggle_Inference.ipynb) trên Kaggle:

1. Bật **Accelerator: GPU T4 x2** (hoặc **GPU P100**), **Internet: On**, và đính kèm dataset dữ liệu cuộc thi qua **+ Add Input**.
2. Chạy tuần tự **Cell 1–2** (tự khóa `CUDA_VISIBLE_DEVICES=0` để tránh lỗi `DataParallel` với 4-bit `device_map="auto"`, tự tìm `/kaggle/input`), **Cell 5** (tùy chọn kiểm chứng Unseen-App CV) và **Cell 6–7** (train full `1,545` mẫu → xuất `/kaggle/working/submission.jsonl` + tạo `FileLink` tải trực tiếp).

---

## Build & Chạy Docker Offline (`private_test`)

Copy weights LoRA đã train vào thư mục `adapter/`, sau đó build và chạy container ở chế độ không mạng (`--network none`):

```bash
docker build -f docker/Dockerfile -t lifemate-agent .
docker run --rm --gpus all --network none -v /path/to/private_test:/data -v /path/to/out:/output lifemate-agent /data /output/submission.jsonl private_test
```
