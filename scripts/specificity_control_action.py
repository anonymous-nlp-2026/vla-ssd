"""Specificity control: ablate L1 bottom-24 instruction-attention heads.
Compares with top-24 ablation (d=0.65) to test whether disruption is
specific to instruction-dominant heads or a generic effect of ablating
any 24 heads."""

import argparse
import gc
import glob
import json
import os
import sys
import time

sys.path.insert(0, "./data/LIBERO")

os.environ["MUJOCO_GL"] = "egl"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats

np.random.seed(42)
torch.manual_seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "./checkpoints/openvla-7b"
UNNORM_KEY = "bridge_orig"
DATA_DIR = "./data/libero/libero_goal"
RESULTS_DIR = "./results"

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS

# Bottom-24 heads sorted by instruction attention fraction (ascending).
# Includes control-8 [3,5,6,9,13,21,24,28] plus next 16 lowest.
BOTTOM_HEADS_24 = [
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20,
    21, 23, 24, 26, 28, 29, 30, 31,
]

TOP_24_COHENS_D = 0.65

DIM_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


# ── Model loading ───────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    return model, processor


# ── Manual autoregressive generation (bypasses generate() DynamicCache bug) ─
def predict_action_with_logits(model, input_ids, pixel_values, unnorm_key):
    action_dim = model.get_action_dim(unnorm_key)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)),
            dim=1,
        )
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values,
            past_key_values=None,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    generated_ids = []
    all_logits = []

    for _ in range(action_dim):
        all_logits.append(next_token_logits.float().cpu())
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.squeeze().item())
        attn_mask = torch.ones(
            (1, past_key_values[0][0].shape[2] + 1),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                pixel_values=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    predicted_action_token_ids = np.array(generated_ids)
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1
    )
    normalized_actions = model.bin_centers[discretized_actions]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    logits_tensor = torch.cat(all_logits, dim=0)
    return action, predicted_action_token_ids, logits_tensor


# ── Ablation hook (instance-level patch, transformers 4.40.1) ───────────────
def install_ablation_hook(model):
    import math
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    layer_attn = model.language_model.model.layers[1].self_attn
    layer_attn._ablate_heads = None
    original_forward = layer_attn.forward

    def ablated_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = layer_attn.q_proj(hidden_states)
        key_states = layer_attn.k_proj(hidden_states)
        value_states = layer_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, layer_attn.num_heads, layer_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)

        pkv = getattr(layer_attn, "past_key_value", past_key_value)
        cos, sin = layer_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if pkv is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = pkv.update(key_states, value_states, layer_attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, layer_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, layer_attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(layer_attn.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=layer_attn.attention_dropout, training=layer_attn.training)

        if layer_attn._ablate_heads is not None:
            kv_len = attn_weights.shape[-1]
            if INST_START < kv_len:
                attn_weights[:, layer_attn._ablate_heads, :, INST_START:] = 0.0
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn_weights = attn_weights / row_sums

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, layer_attn.hidden_size)
        attn_output = layer_attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, pkv

    layer_attn.forward = ablated_forward
    return original_forward, layer_attn


def remove_ablation_hook(layer_attn, original_forward):
    layer_attn._ablate_heads = None
    layer_attn.forward = original_forward


# ── KL divergence ───────────────────────────────────────────────────────────
def compute_kl_divergence(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = F.softmax(logits_p, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.numpy()


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_task_files(n_tasks):
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    return hdf5_files[:n_tasks]


def get_language_instruction(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        info = json.loads(f["data"].attrs["problem_info"])
    return info["language_instruction"]


def compute_task_metrics(task_b, task_a, task_b_ids, task_a_ids, task_kls):
    paired_mse = np.mean((task_b - task_a) ** 2, axis=1)
    t_res = stats.ttest_1samp(paired_mse, 0)
    cohens_d = np.mean(paired_mse) / (np.std(paired_mse, ddof=1) + 1e-12)

    per_dim_shift = np.abs(task_b - task_a)
    token_div = np.mean(task_b_ids != task_a_ids)

    return {
        "paired_mse": {
            "mean": float(np.mean(paired_mse)),
            "std": float(np.std(paired_mse)),
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "cohens_d": float(cohens_d),
        },
        "per_dim_shift": {
            "means": [float(x) for x in np.mean(per_dim_shift, axis=0)],
            "stds": [float(x) for x in np.std(per_dim_shift, axis=0)],
        },
        "token_divergence_rate": float(token_div),
        "kl_divergence": {
            "mean": float(np.mean(task_kls)),
            "std": float(np.std(task_kls)),
        },
        "n_samples": int(len(task_b)),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_tasks = 1 if args.dry_run else 3
    n_demos = 1 if args.dry_run else 10
    n_timesteps = 2 if args.dry_run else 50

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model...")
    model, processor = load_model()
    print("Model loaded.")

    original_forward, layer1_attn = install_ablation_hook(model)

    task_files = get_task_files(n_tasks)
    print(f"Tasks ({n_tasks}): {[os.path.basename(f) for f in task_files]}")
    print(f"Config: {n_demos} demos/task, {n_timesteps} timesteps/demo")
    print(f"Bottom-24 heads: {BOTTOM_HEADS_24}")

    all_baseline_actions = []
    all_ablated_actions = []
    all_gt_actions = []
    all_baseline_token_ids = []
    all_ablated_token_ids = []
    all_kl_divs = []
    per_task_data = {}

    total_passes = 0
    t_start = time.time()

    for task_idx, hdf5_path in enumerate(task_files):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        instruction = get_language_instruction(hdf5_path)
        print(f"\n[Task {task_idx+1}/{n_tasks}] {task_name}")
        print(f"  Instruction: {instruction}")

        task_baseline_actions = []
        task_ablated_actions = []
        task_gt_actions = []
        task_baseline_token_ids = []
        task_ablated_token_ids = []
        task_kl_divs = []

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(n_demos):
                demo_key = f"data/demo_{demo_idx}"
                actions_data = f[f"{demo_key}/actions"][:]
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(n_timesteps, len(actions_data))

                for step_idx in range(n_steps):
                    image_np = images_data[step_idx]
                    gt_action = actions_data[step_idx]

                    img = Image.fromarray(image_np)
                    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]

                    # Baseline (ablation disabled)
                    layer1_attn._ablate_heads = None
                    b_action, b_ids, b_logits = predict_action_with_logits(
                        model, input_ids, pixel_values, UNNORM_KEY
                    )

                    # Ablated (bottom-24 heads)
                    layer1_attn._ablate_heads = BOTTOM_HEADS_24
                    a_action, a_ids, a_logits = predict_action_with_logits(
                        model, input_ids, pixel_values, UNNORM_KEY
                    )

                    kl_per_token = compute_kl_divergence(b_logits, a_logits)

                    task_baseline_actions.append(b_action)
                    task_ablated_actions.append(a_action)
                    task_gt_actions.append(gt_action)
                    task_baseline_token_ids.append(b_ids)
                    task_ablated_token_ids.append(a_ids)
                    task_kl_divs.append(kl_per_token)

                    total_passes += 2
                    if total_passes % 100 == 0:
                        elapsed = time.time() - t_start
                        rate = total_passes / elapsed
                        eta = (n_tasks * n_demos * n_timesteps * 2 - total_passes) / rate
                        print(
                            f"  [{total_passes} passes, {elapsed:.0f}s, "
                            f"{rate:.1f} pass/s, ETA {eta:.0f}s]"
                        )

                    del b_logits, a_logits
                    torch.cuda.empty_cache()

        task_b = np.array(task_baseline_actions)
        task_a = np.array(task_ablated_actions)
        task_b_ids = np.array(task_baseline_token_ids)
        task_a_ids = np.array(task_ablated_token_ids)
        task_kls = np.array(task_kl_divs)

        per_task_data[task_name] = compute_task_metrics(
            task_b, task_a, task_b_ids, task_a_ids, task_kls
        )
        m = per_task_data[task_name]
        print(
            f"  >> paired_mse={m['paired_mse']['mean']:.6f}, "
            f"token_div={m['token_divergence_rate']:.4f}, "
            f"kl={m['kl_divergence']['mean']:.4f}"
        )

        all_baseline_actions.extend(task_baseline_actions)
        all_ablated_actions.extend(task_ablated_actions)
        all_gt_actions.extend(task_gt_actions)
        all_baseline_token_ids.extend(task_baseline_token_ids)
        all_ablated_token_ids.extend(task_ablated_token_ids)
        all_kl_divs.extend(task_kl_divs)

        gc.collect()
        torch.cuda.empty_cache()

    # ── Global metrics ──────────────────────────────────────────────────────
    remove_ablation_hook(layer1_attn, original_forward)

    all_b = np.array(all_baseline_actions)
    all_a = np.array(all_ablated_actions)
    all_gt = np.array(all_gt_actions)
    all_b_ids = np.array(all_baseline_token_ids)
    all_a_ids = np.array(all_ablated_token_ids)
    all_kls = np.array(all_kl_divs)

    paired_mse_all = np.mean((all_b - all_a) ** 2, axis=1)
    t_res_all = stats.ttest_1samp(paired_mse_all, 0)
    cohens_d_all = np.mean(paired_mse_all) / (np.std(paired_mse_all, ddof=1) + 1e-12)

    per_dim_shift_all = np.abs(all_b - all_a)
    token_div_all = np.mean(all_b_ids != all_a_ids)

    kl_per_position = np.mean(all_kls, axis=0)

    mse_gt_baseline = np.mean((all_b - all_gt) ** 2, axis=1)
    mse_gt_ablated = np.mean((all_a - all_gt) ** 2, axis=1)

    try:
        wilcoxon_res = stats.wilcoxon(paired_mse_all)
        wilcoxon_p = float(wilcoxon_res.pvalue)
    except Exception:
        wilcoxon_p = None

    bottom24_d = float(cohens_d_all)

    output = {
        "config": {
            "n_tasks": n_tasks,
            "n_demos_per_task": n_demos,
            "n_timesteps_per_demo": n_timesteps,
            "total_forward_passes": total_passes,
            "model": "openvla-7b",
            "ablation": "L1_bottom_24_heads_zeroed",
            "ablated_heads": BOTTOM_HEADS_24,
            "unnorm_key": UNNORM_KEY,
        },
        "bottom_24_heads": BOTTOM_HEADS_24,
        "main_metrics": {
            "paired_action_mse": {
                "mean": float(np.mean(paired_mse_all)),
                "std": float(np.std(paired_mse_all)),
                "t_stat": float(t_res_all.statistic),
                "p_value": float(t_res_all.pvalue),
                "cohens_d": bottom24_d,
            },
            "per_dim_action_shift": {
                "means": [float(x) for x in np.mean(per_dim_shift_all, axis=0)],
                "stds": [float(x) for x in np.std(per_dim_shift_all, axis=0)],
                "dim_names": DIM_NAMES,
            },
            "token_divergence_rate": float(token_div_all),
        },
        "secondary_metrics": {
            "kl_divergence": {
                "mean": float(np.mean(all_kls)),
                "std": float(np.std(all_kls)),
                "per_token_position": [float(x) for x in kl_per_position],
            },
            "mse_vs_gt_baseline": {
                "mean": float(np.mean(mse_gt_baseline)),
                "std": float(np.std(mse_gt_baseline)),
                "note": "OOD, secondary",
            },
            "mse_vs_gt_ablated": {
                "mean": float(np.mean(mse_gt_ablated)),
                "std": float(np.std(mse_gt_ablated)),
                "note": "OOD, secondary",
            },
            "wilcoxon_p": wilcoxon_p,
        },
        "comparison_with_top24": {
            "top24_d": TOP_24_COHENS_D,
            "bottom24_d": bottom24_d,
            "ratio": TOP_24_COHENS_D / (bottom24_d + 1e-12),
        },
        "per_task": per_task_data,
    }

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {total_passes} forward passes in {elapsed_total:.0f}s")
    print(
        f"Paired MSE: {np.mean(paired_mse_all):.6f} "
        f"(p={t_res_all.pvalue:.2e}, d={cohens_d_all:.3f})"
    )
    print(f"Token divergence: {token_div_all:.4f}")
    print(f"KL divergence: {np.mean(all_kls):.4f}")
    print(f"Comparison: top24_d={TOP_24_COHENS_D:.2f}, bottom24_d={bottom24_d:.3f}, ratio={TOP_24_COHENS_D/(bottom24_d+1e-12):.2f}")

    out_path = os.path.join(RESULTS_DIR, "specificity_control_action.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
