#!/bin/bash
set -e

# Run offline inference pipeline
# Arguments passed by BTC: input_dir output_file
INPUT_DIR=${1:-"/data/test"}
OUTPUT_FILE=${2:-"/output/submission.jsonl"}

echo "Running LifeMate offline inference on $INPUT_DIR -> $OUTPUT_FILE"
python -m src.agent --input_dir "$INPUT_DIR" --output_file "$OUTPUT_FILE"
echo "Inference finished successfully."
