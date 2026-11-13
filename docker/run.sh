#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-/data}"
OUTPUT_FILE="${2:-/output/submission.jsonl}"
SPLIT="${3:-private_test}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve whether INPUT_DIR is `<root>` or `<root>/<split>`
if [ -f "${INPUT_DIR}/${SPLIT}/${SPLIT}.jsonl" ]; then
  ROOT_DIR="${INPUT_DIR}"
elif [ -f "${INPUT_DIR}/${SPLIT}.jsonl" ]; then
  ROOT_DIR="$(dirname "${INPUT_DIR}")"
  SPLIT="$(basename "${INPUT_DIR}")"
elif [ -f "${INPUT_DIR}/public_test/public_test.jsonl" ]; then
  ROOT_DIR="${INPUT_DIR}"
  SPLIT="public_test"
else
  echo "ERROR: Cannot find ${SPLIT}.jsonl inside ${INPUT_DIR}" >&2
  exit 1
fi

ADAPTER_ARGS=()
if [ -f "/app/adapter/adapter_config.json" ]; then
  ADAPTER_ARGS=(--adapter /app/adapter)
fi

python /app/run_inference.py \
  --root "${ROOT_DIR}" \
  --split "${SPLIT}" \
  --backend hf \
  --model "${LIFEMATE_BASE_MODEL:-Qwen/Qwen3-4B}" \
  --asr-model "${LIFEMATE_ASR_MODEL:-vinai/PhoWhisper-large}" \
  --no-cache \
  --offline \
  --force \
  "${ADAPTER_ARGS[@]}" \
  --output "${OUTPUT_FILE}"
