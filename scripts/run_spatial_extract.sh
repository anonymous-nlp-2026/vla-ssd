#!/bin/bash
set -e
export HDF5_USE_FILE_LOCKING=FALSE

PYTHON=<PROJECT_ROOT>/miniconda3/bin/python3
SCRIPT=./scripts/cross_instruction_extract_layers.py
DATA_DIR=./data/libero/libero_spatial/
OUTPUT_DIR=./results/cross_instruction_spatial/
LAYERS="0-31"

echo "=== Starting LIBERO-Spatial cross-instruction extraction ==="
echo "Trained on cuda:0, Untrained on cuda:1"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --layers $LAYERS \
  --condition trained \
  --gpu 0 \
  --max_demos 50 \
  --frames_per_demo 1 \
  > ${OUTPUT_DIR}/extract_trained.log 2>&1 &
PID_TRAINED=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --layers $LAYERS \
  --condition untrained \
  --gpu 0 \
  --max_demos 50 \
  --frames_per_demo 1 \
  > ${OUTPUT_DIR}/extract_untrained.log 2>&1 &
PID_UNTRAINED=$!

echo "Trained PID: $PID_TRAINED, Untrained PID: $PID_UNTRAINED"
echo "Waiting for both to finish..."

wait $PID_TRAINED
STATUS_T=$?
echo "Trained finished with exit code $STATUS_T"

wait $PID_UNTRAINED
STATUS_U=$?
echo "Untrained finished with exit code $STATUS_U"

if [ $STATUS_T -eq 0 ] && [ $STATUS_U -eq 0 ]; then
  echo "=== Both extractions completed successfully ==="
  echo "Running classification..."
  $PYTHON ./scripts/cross_instruction_classify_spatial.py
else
  echo "ERROR: Extraction failed (trained=$STATUS_T, untrained=$STATUS_U)"
  echo "--- Trained log tail ---"
  tail -20 ${OUTPUT_DIR}/extract_trained.log
  echo "--- Untrained log tail ---"
  tail -20 ${OUTPUT_DIR}/extract_untrained.log
  exit 1
fi
