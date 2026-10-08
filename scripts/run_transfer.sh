#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: scripts/run_transfer.sh qwen3_4b|llama_3_2_3b [output-dir]}"
OUTPUT_DIR="${2:-outputs/${MODEL_ID}}"
case "$MODEL_ID" in
  qwen3_4b)
    BASE_MODEL="${QWEN3_4B_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
    ADAPTER="weights/qwen3_4b"
    ;;
  llama_3_2_3b)
    BASE_MODEL="${LLAMA_3_2_3B_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
    ADAPTER="weights/llama_3_2_3b"
    ;;
  *) echo "unknown model: $MODEL_ID" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
python scripts/run_inference.py --model-id "$MODEL_ID" --condition base --base-model "$BASE_MODEL" --output "$OUTPUT_DIR/base.jsonl"
python scripts/run_inference.py --model-id "$MODEL_ID" --condition adapted --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$OUTPUT_DIR/adapted.jsonl"
python scripts/score_paired.py --base "$OUTPUT_DIR/base.jsonl" --adapted "$OUTPUT_DIR/adapted.jsonl" --output-dir "$OUTPUT_DIR/scored"
