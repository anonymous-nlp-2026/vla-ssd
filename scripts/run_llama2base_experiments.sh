#!/bin/bash
set -e

echo "[$(date)] Waiting for Llama-2 download to complete..."
while ls <HF_CACHE>/models--NousResearch--Llama-2-7b-hf/blobs/*.incomplete 2>/dev/null; do
    sleep 10
done
echo "[$(date)] Download complete!"

echo "[$(date)] Step 3a: Running cross_instruction_extract_dense (llama2base)..."
CUDA_VISIBLE_DEVICES=0 python3 ./scripts/cross_instruction_extract_dense.py \
  --condition llama2base --gpu 0 2>&1 | tee ./results/cross_instruction_dense/llama2base.log

echo "[$(date)] Step 3b: Running l1_attention_visualization (llama2base)..."
CUDA_VISIBLE_DEVICES=0 python3 ./scripts/l1_attention_visualization.py \
  --condition llama2base --gpu 0 2>&1 | tee ./results/attention/llama2base.log

echo "[$(date)] Step 4: Running classification..."
python3 ./scripts/cross_instruction_classify_dense.py \
  --condition llama2base 2>&1 | tee ./results/cross_instruction_dense/llama2base_classify.log

echo "[$(date)] All experiments complete!"
