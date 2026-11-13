# Zalo AI Challenge 2026 — LifeMate

Offline Voice-to-Action Agentic Pipeline (<5B SLM + Offline Vietnamese ASR), 100% compliant with BTC rules:
- **Zero `state.json` leakage**: device state is only accessed via multi-turn `read` tool calls against `zaic_tools.ToolEnv`.
- **Pre-call Guardrail**: validates every call with `env.validate(name, args)` before `env.call(name, args)` so invalid arguments never pollute `env.get_tool_calls()`, feeding validation feedback back to the agent for self-correction.
- **Unseen-App Generalization**: compact typed tool signatures + Causal DAG step decomposition + App-Dropout + schema shuffling + strict assistant-only loss masking.

## Structure
- `src/prompt_builder.py` — Compact schema serializer (`format_compact_tool`), temporal anchors, full `history` turn serialization.
- `src/trajectory.py` — Causal DAG step grouping (`group_trajectory_into_steps`), App-Dropout, safe phonetic verb-noise augmentation, ChatML formatting, and exact multi-turn assistant loss masking.
- `src/guardrails.py` — `<think>` / fence stripping, JSON extraction, schema-aware coercion (thousands separators, Vietnamese time/date normalization), and gold defaults.
- `src/agent.py` — `LifeMateAgent` ReAct loop + `HFTransformersBackend` (4-bit NF4 + LoRA + Qwen3 `enable_thinking=False`), `OpenAICompatibleBackend`, `HeuristicBackend`, `NoopBackend`.
- `src/asr.py` — `OfflineASR` (`vinai/PhoWhisper-large` / `faster-whisper`) with 16kHz mono resampling, request + `_u*` history transcription, and dual-cache fallback.
- `scripts/prepare_splits.py` — Generates `splits/val_unseen.json` (142 unseen-app samples across `bike_rental`, `exchange_rate`, `medication`, `package_tracking`), `splits/val_indomain.json` (100 stratified samples), and `splits/val_all.json`.
- `scripts/benchmark_asr.py` — Colab T4 CER/WER benchmark on `train` audio & cache regeneration.
- `scripts/inspect_errors.py` — Per-failure-bucket diagnostics via `evaluate.py`.
- `train_sft.py` — QLoRA trainer supporting `--split-mode val` and `--split-mode full`.
- `run_inference.py` — Resumable ReAct inference + `zaic_tools --validate` + optional `evaluate.py`.
- `docker/Dockerfile` & `docker/run.sh` — Offline (`HF_HUB_OFFLINE=1`, `--no-cache`) Private Test container.

## Local Workflow (Windows CPU)
```powershell
python -m unittest tests/test_pipeline.py -v
python scripts/prepare_splits.py
python run_inference.py --split train --ids-file splits/val_all.json --backend noop --force --output outputs/val_noop.jsonl --eval
python run_inference.py --split train --ids-file splits/val_all.json --backend heuristic --force --output outputs/val_heuristic.jsonl --eval
```

## Google Colab T4 Workflow
Open [notebooks/LifeMate_Colab_Inference.ipynb](notebooks/LifeMate_Colab_Inference.ipynb), mount Drive with `train+public_test.zip`, and run cells sequentially:
1. Validate QLoRA (`Qwen/Qwen3-4B` or `Qwen/Qwen2.5-3B-Instruct`) on Unseen-App CV (`splits/val_all.json`).
2. Train `--split-mode full` and generate `submission.jsonl` (`634/634` rows, schema-validated).

## Offline Docker Build & Run
Copy trained LoRA weights into `adapter/`, then build and run offline:
```bash
docker build -f docker/Dockerfile -t lifemate-agent .
docker run --rm --gpus all --network none -v /path/to/private_test:/data -v /path/to/out:/output lifemate-agent /data /output/submission.jsonl private_test
```
