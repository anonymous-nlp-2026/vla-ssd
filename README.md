# Instruction Routing in Fine-Tuned Vision-Language-Action Models: A Cross-Architecture Analysis

This repository contains the analysis code for probing the internal representations of Vision-Language-Action (VLA) models, accompanying the paper submission to CoRL 2026.

## Overview

We investigate what VLA models learn internally by extracting hidden-state representations across all transformer layers and applying a suite of probing and similarity analyses. The codebase supports:

- **Feature extraction** from trained and untrained VLA models (OpenVLA, Phi-3V-based TracVLA) and vision encoders (DINOv2, SigLIP, CLIP)
- **Linear/MLP probing** for temporal distance, action prediction, subtask classification, and goal classification
- **Representational Similarity Analysis (RSA)** comparing neural representations to action-space and visual similarity
- **Attention analysis** across layers and heads
- **Counterfactual experiments** with instruction permutation and cross-instruction generalization
- **Cross-architecture comparisons** (OpenVLA vs. Phi-3V vs. Llama-2 backbone)
- **Ablation studies** (demo sensitivity, probe seed robustness, permutation null tests)

## Directory Structure

```
.
├── scripts/                    # Main analysis scripts
│   ├── extract_features.py     # Extract VLA hidden states from LIBERO demos
│   ├── extract_dinov2.py       # Extract DINOv2 features
│   ├── train_probes.py         # Train linear/MLP probes on extracted features
│   ├── analyze_results.py      # Aggregate and analyze probe results
│   ├── counterfactual_*.py     # Counterfactual instruction experiments
│   ├── cross_instruction_*.py  # Cross-instruction generalization
│   ├── phi3v_*.py              # Phi-3V / TracVLA experiments
│   ├── plot_*.py               # Plotting scripts for paper figures
│   ├── gen_fig_*.py            # Figure generation scripts
│   ├── sr_*.py                 # Success rate evaluation pipeline
│   └── ablations/              # Ablation studies (demo sensitivity, permutation null, etc.)
├── rsa_analysis.py             # RSA analysis (trained vs. untrained, per-layer)
├── rsa_analysis_no_inst.py     # RSA without instruction tokens
├── rsa_no_instruction.py       # No-instruction control
├── bootstrap_training_effect.py # Bootstrap tests for training effect significance
├── action_baselines.py         # Action prediction baselines
├── action_head_v2_*.py         # Action head decoding analysis
├── pca_linear_probe_gc.py      # PCA + linear probe for goal classification
├── paper_figures/              # Generated paper figures (PDF + PNG)
│   └── updated/                # Updated figure versions
├── artifacts/figures/paper/    # Final paper figures
├── results/                    # Pre-computed probe results (JSON + CSV)
└── requirements.txt
```

## Environment

- Python 3.10+
- PyTorch 2.1+ with CUDA support
- See `requirements.txt` for full dependencies

## Data Preparation

1. Download the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark datasets (LIBERO-Goal and/or LIBERO-10) and place them under `./data/libero/`.
2. Download model checkpoints:
   - [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b) (base and LIBERO-finetuned)
   - Phi-3V / TracVLA checkpoints (if running cross-architecture experiments)

## Reproducing Results

### Step 1: Extract Features

```bash
# Trained OpenVLA
python scripts/extract_features.py \
  --model_path ./checkpoints/openvla-7b-finetuned-libero-goal \
  --data_dir ./data/libero/libero_goal/ \
  --output_dir ./features/trained_libero_goal/ \
  --mode trained

# Untrained OpenVLA (randomly re-initialized LLM backbone)
python scripts/extract_features.py \
  --model_path ./checkpoints/openvla-7b \
  --data_dir ./data/libero/libero_goal/ \
  --output_dir ./features/untrained_libero_goal/ \
  --mode untrained

# DINOv2 features
python scripts/extract_dinov2.py \
  --data_dir ./data/libero/libero_goal/ \
  --output_dir ./features/dinov2_libero_goal/
```

### Step 2: Train Probes

```bash
# Temporal distance probes
python scripts/train_probes.py \
  --features_dir ./features/trained_libero_goal/ \
  --output_dir ./results/trained/temporal_distance/ \
  --probe_type temporal_distance

# Action prediction probes
python scripts/train_probes.py \
  --features_dir ./features/trained_libero_goal/ \
  --output_dir ./results/trained/action_delta/ \
  --probe_type action_delta \
  --data_dir ./data/libero/libero_goal/
```

### Step 3: RSA Analysis

```bash
python rsa_analysis.py
```

### Step 4: Generate Figures

Plotting scripts are in `scripts/plot_*.py` and `scripts/gen_fig_*.py`. Example:

```bash
python scripts/plot_action_r2_threecond.py
python scripts/plot_threecond_multiseed_v2.py
```

## Hardware Requirements

- Feature extraction: 1x GPU with >= 24GB VRAM (A5000/A6000/A100)
- Probe training and analysis: CPU-only is sufficient
