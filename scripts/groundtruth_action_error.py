"""Ground-truth action error analysis: baseline vs top-24 vs bottom-24 ablation."""

import gc
import glob
import json
import math
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

MODEL_PATH = "./checkpoints/openvla-7b"
UNNORM_KEY = "bridge_orig"
DATA_DIR = "./data/libero/libero_goal"
RESULTS_DIR = "./results"

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS

TOP_24_HEADS = [
    0, 1, 2, 4, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20,
    22, 23, 25, 26, 27, 29, 30, 31,
]
BOTTOM_24_HEADS = [
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20,
    21, 23, 24, 26, 28, 29, 30, 31,
]

DIM_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    return model, processor


def predict_action_with_logits(model, input_ids, pixel_values, unnorm_key):
    action_dim = model.get_action_dim(unnorm_key)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)),
            dim=1,
        )
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values, past_key_values=None, use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    generated_ids = []
    for _ in range(action_dim):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.squeeze().item())
        attn_mask = torch.ones(
            (1, past_key_values[0][0].shape[2] + 1),
            device=input_ids.device, dtype=input_ids.dtype,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token, attention_mask=attn_mask,
                pixel_values=None, past_key_values=past_key_values, use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    predicted_action_token_ids = np.array(generated_ids)
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    action = np.where(
        mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return action


def install_ablation_hook(model):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    layer_attn = model.language_model.model.layers[1].self_attn
    layer_attn._ablate_heads = None
    original_forward = layer_attn.forward

    def ablated_forward(
        hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, **kwargs,
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


def get_task_files(n_tasks):
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))[:n_tasks]


def get_language_instruction(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        info = json.loads(f["data"].attrs["problem_info"])
    return info["language_instruction"]


def main():
    n_tasks = 3
    n_demos = 10
    n_timesteps = 50

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model...")
    model, processor = load_model()
    print("Model loaded.")

    original_forward, layer_attn = install_ablation_hook(model)

    task_files = get_task_files(n_tasks)
    print(f"Tasks ({n_tasks}): {[os.path.basename(f) for f in task_files]}")

    conditions = {
        "original": None,
        "top24_ablated": TOP_24_HEADS,
        "bottom24_ablated": BOTTOM_24_HEADS,
    }

    all_data = {cond: {"actions": [], "gt_actions": []} for cond in conditions}
    per_task_data = {}

    total_passes = 0
    t_start = time.time()

    for task_idx, hdf5_path in enumerate(task_files):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        instruction = get_language_instruction(hdf5_path)
        print(f"\n{'='*60}")
        print(f"Task {task_idx}: {task_name}")
        print(f"  Instruction: {instruction}")

        task_cond_data = {cond: {"actions": [], "gt_actions": []} for cond in conditions}
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

                    for cond_name, heads in conditions.items():
                        layer_attn._ablate_heads = heads
                        action = predict_action_with_logits(model, input_ids, pixel_values, UNNORM_KEY)
                        task_cond_data[cond_name]["actions"].append(action.tolist())
                        task_cond_data[cond_name]["gt_actions"].append(gt_action.tolist())
                        total_passes += 1

                    if total_passes % 150 == 0:
                        elapsed = time.time() - t_start
                        rate = total_passes / elapsed
                        total_expected = n_tasks * n_demos * n_timesteps * len(conditions)
                        eta = (total_expected - total_passes) / rate
                        print(f"  [{total_passes}/{total_expected} passes, {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s]")

                    torch.cuda.empty_cache()

        task_metrics = {}
        for cond_name in conditions:
            pred = np.array(task_cond_data[cond_name]["actions"])
            gt = np.array(task_cond_data[cond_name]["gt_actions"])
            se = (pred - gt) ** 2
            ae = np.abs(pred - gt)
            mse_per_sample = np.mean(se, axis=1)
            mae_per_sample = np.mean(ae, axis=1)

            task_metrics[cond_name] = {
                "mse": {"mean": float(np.mean(mse_per_sample)), "std": float(np.std(mse_per_sample))},
                "mae": {"mean": float(np.mean(mae_per_sample)), "std": float(np.std(mae_per_sample))},
                "per_dim_mse": [float(x) for x in np.mean(se, axis=0)],
                "per_dim_mae": [float(x) for x in np.mean(ae, axis=0)],
                "n_samples": int(len(pred)),
            }

        orig_pred = np.array(task_cond_data["original"]["actions"])
        orig_gt = np.array(task_cond_data["original"]["gt_actions"])
        orig_mse = np.mean((orig_pred - orig_gt) ** 2, axis=1)

        for abl_name in ["top24_ablated", "bottom24_ablated"]:
            abl_pred = np.array(task_cond_data[abl_name]["actions"])
            abl_gt = np.array(task_cond_data[abl_name]["gt_actions"])
            abl_mse = np.mean((abl_pred - abl_gt) ** 2, axis=1)
            delta_mse = abl_mse - orig_mse
            try:
                wilcoxon_p = float(stats.wilcoxon(delta_mse).pvalue)
            except Exception:
                wilcoxon_p = None

            task_metrics[f"delta_{abl_name}"] = {
                "mean": float(np.mean(delta_mse)),
                "std": float(np.std(delta_mse)),
                "median": float(np.median(delta_mse)),
                "pct_increased": float(np.mean(delta_mse > 0)),
                "wilcoxon_p": wilcoxon_p,
            }

        per_task_data[task_name] = task_metrics
        print(f"  Original MSE vs GT: {task_metrics['original']['mse']['mean']:.6f}")
        print(f"  Top-24  MSE vs GT:  {task_metrics['top24_ablated']['mse']['mean']:.6f} (delta={task_metrics['delta_top24_ablated']['mean']:+.6f})")
        print(f"  Bot-24  MSE vs GT:  {task_metrics['bottom24_ablated']['mse']['mean']:.6f} (delta={task_metrics['delta_bottom24_ablated']['mean']:+.6f})")

        for cond_name in conditions:
            all_data[cond_name]["actions"].extend(task_cond_data[cond_name]["actions"])
            all_data[cond_name]["gt_actions"].extend(task_cond_data[cond_name]["gt_actions"])

        gc.collect()
        torch.cuda.empty_cache()

    layer_attn._ablate_heads = None
    remove_ablation_hook(layer_attn, original_forward)

    global_metrics = {}
    for cond_name in conditions:
        pred = np.array(all_data[cond_name]["actions"])
        gt = np.array(all_data[cond_name]["gt_actions"])
        se = (pred - gt) ** 2
        ae = np.abs(pred - gt)
        mse_per_sample = np.mean(se, axis=1)
        mae_per_sample = np.mean(ae, axis=1)

        global_metrics[cond_name] = {
            "mse": {"mean": float(np.mean(mse_per_sample)), "std": float(np.std(mse_per_sample))},
            "mae": {"mean": float(np.mean(mae_per_sample)), "std": float(np.std(mae_per_sample))},
            "per_dim_mse": {DIM_NAMES[i]: float(np.mean(se[:, i])) for i in range(7)},
            "per_dim_mae": {DIM_NAMES[i]: float(np.mean(ae[:, i])) for i in range(7)},
            "n_samples": int(len(pred)),
        }

    orig_pred = np.array(all_data["original"]["actions"])
    orig_gt = np.array(all_data["original"]["gt_actions"])
    orig_mse = np.mean((orig_pred - orig_gt) ** 2, axis=1)
    orig_mae = np.mean(np.abs(orig_pred - orig_gt), axis=1)
    orig_per_dim_se = (orig_pred - orig_gt) ** 2
    orig_per_dim_ae = np.abs(orig_pred - orig_gt)

    delta_tests = {}
    for abl_name in ["top24_ablated", "bottom24_ablated"]:
        abl_pred = np.array(all_data[abl_name]["actions"])
        abl_gt = np.array(all_data[abl_name]["gt_actions"])
        abl_mse = np.mean((abl_pred - abl_gt) ** 2, axis=1)
        abl_mae = np.mean(np.abs(abl_pred - abl_gt), axis=1)
        abl_per_dim_se = (abl_pred - abl_gt) ** 2
        abl_per_dim_ae = np.abs(abl_pred - abl_gt)

        delta_mse = abl_mse - orig_mse
        delta_mae = abl_mae - orig_mae

        try:
            wilcoxon_mse_p = float(stats.wilcoxon(delta_mse).pvalue)
            wilcoxon_mae_p = float(stats.wilcoxon(delta_mae).pvalue)
        except Exception:
            wilcoxon_mse_p = wilcoxon_mae_p = None

        per_dim_delta = {}
        for i, dim in enumerate(DIM_NAMES):
            d_se = abl_per_dim_se[:, i] - orig_per_dim_se[:, i]
            d_ae = abl_per_dim_ae[:, i] - orig_per_dim_ae[:, i]
            try:
                p_se = float(stats.wilcoxon(d_se).pvalue)
                p_ae = float(stats.wilcoxon(d_ae).pvalue)
            except Exception:
                p_se = p_ae = None
            per_dim_delta[dim] = {
                "delta_mse_mean": float(np.mean(d_se)),
                "delta_mae_mean": float(np.mean(d_ae)),
                "delta_mse_wilcoxon_p": p_se,
                "delta_mae_wilcoxon_p": p_ae,
            }

        delta_tests[abl_name] = {
            "delta_mse": {
                "mean": float(np.mean(delta_mse)), "std": float(np.std(delta_mse)),
                "median": float(np.median(delta_mse)),
                "pct_increased": float(np.mean(delta_mse > 0)),
                "wilcoxon_p": wilcoxon_mse_p,
            },
            "delta_mae": {
                "mean": float(np.mean(delta_mae)), "std": float(np.std(delta_mae)),
                "median": float(np.median(delta_mae)),
                "pct_increased": float(np.mean(delta_mae > 0)),
                "wilcoxon_p": wilcoxon_mae_p,
            },
            "per_dim_delta": per_dim_delta,
        }

    grouped = {}
    for cond_name in conditions:
        pred = np.array(all_data[cond_name]["actions"])
        gt = np.array(all_data[cond_name]["gt_actions"])
        se = (pred - gt) ** 2
        ae = np.abs(pred - gt)
        grouped[cond_name] = {
            "translation_mse": float(np.mean(se[:, :3])),
            "translation_mae": float(np.mean(ae[:, :3])),
            "rotation_mse": float(np.mean(se[:, 3:6])),
            "rotation_mae": float(np.mean(ae[:, 3:6])),
            "gripper_mse": float(np.mean(se[:, 6])),
            "gripper_mae": float(np.mean(ae[:, 6])),
        }

    output = {
        "config": {
            "n_tasks": n_tasks, "n_demos": n_demos, "n_timesteps": n_timesteps,
            "total_passes": total_passes, "model": "openvla-7b",
            "unnorm_key": UNNORM_KEY,
            "top24_heads": TOP_24_HEADS, "bottom24_heads": BOTTOM_24_HEADS,
            "tasks": [os.path.basename(f).replace("_demo.hdf5", "") for f in task_files],
            "note": "GT actions in LIBERO native space; model outputs use bridge_orig unnorm. Delta metrics cancel domain bias.",
        },
        "global_metrics": global_metrics,
        "delta_tests": delta_tests,
        "grouped_errors": grouped,
        "per_task": per_task_data,
    }

    out_path = os.path.join(RESULTS_DIR, "groundtruth_action_error.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    elapsed = time.time() - t_start
    print(f"Total: {total_passes} passes in {elapsed:.0f}s ({total_passes/elapsed:.1f}/s)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Condition':<20} {'MSE vs GT':>12} {'MAE vs GT':>12} {'d_MSE':>12} {'d_MAE':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cond in conditions:
        m = global_metrics[cond]
        if cond == "original":
            print(f"{cond:<20} {m['mse']['mean']:>12.6f} {m['mae']['mean']:>12.6f} {'0':>12} {'0':>12}")
        else:
            dt = delta_tests[cond]
            print(f"{cond:<20} {m['mse']['mean']:>12.6f} {m['mae']['mean']:>12.6f} {dt['delta_mse']['mean']:>+12.6f} {dt['delta_mae']['mean']:>+12.6f}")

    print(f"\nGrouped errors:")
    print(f"{'Condition':<20} {'Trans MSE':>12} {'Rot MSE':>12} {'Grip MSE':>12}")
    print("-" * 56)
    for cond in conditions:
        g = grouped[cond]
        print(f"{cond:<20} {g['translation_mse']:>12.6f} {g['rotation_mse']:>12.6f} {g['gripper_mse']:>12.6f}")


if __name__ == "__main__":
    main()
