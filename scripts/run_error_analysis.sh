#!/bin/bash
set -e

BASE_DIR="./quantized_models"
OUTPUT_BASE="./error_analysis"

# SST-2
echo "=========================================="
echo "Running error analysis for SST-2"
echo "=========================================="
python scripts/error_analysis.py \
    --task sst2 \
    --model_dir ms7019/sst2-bert-base-uncased-seed42 \
    --quantized_model_dir ${BASE_DIR}/sst2_static_w8a8_calib500 \
    --output_dir ${OUTPUT_BASE}/sst2

echo ""
echo "=========================================="
echo "Running error analysis for MRPC"
echo "=========================================="
python scripts/error_analysis.py \
    --task mrpc \
    --model_dir ms7019/mrpc-bert-base-uncased-seed42 \
    --quantized_model_dir ${BASE_DIR}/mrpc_static_w8a8_calib500 \
    --output_dir ${OUTPUT_BASE}/mrpc

echo ""
echo "=========================================="
echo "Error analysis complete!"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "=========================================="

