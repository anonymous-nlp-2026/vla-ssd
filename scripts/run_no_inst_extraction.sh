#!/bin/bash
set -e
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh
conda activate base
cd .
export CUDA_VISIBLE_DEVICES=0

LOG=./logs/no_inst_extraction.log
mkdir -p ./logs

echo "=== START: $(date) ===" > $LOG

echo "--- Phase 1: Trained VLA, no instruction ---" >> $LOG 2>&1
python scripts/extract_features.py \
  --model_path ./checkpoints/openvla-7b \
  --data_dir ./data/libero/libero_goal/ \
  --output_dir ./features/trained_libero_goal_no_inst/ \
  --task_instruction "" \
  --max_demos 10 \
  --batch_size 4 \
  --gpu 0 >> $LOG 2>&1

echo "--- Phase 2: Untrained VLA, no instruction ---" >> $LOG 2>&1
python scripts/extract_features.py \
  --model_path ./checkpoints/openvla-7b-untrained \
  --data_dir ./data/libero/libero_goal/ \
  --output_dir ./features/untrained_libero_goal_no_inst/ \
  --task_instruction "" \
  --mode untrained \
  --max_demos 10 \
  --batch_size 4 \
  --gpu 0 >> $LOG 2>&1

echo "=== DONE: $(date) ===" >> $LOG
