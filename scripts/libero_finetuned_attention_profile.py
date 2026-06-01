"""Attention profile for openvla-7b-finetuned-libero-goal across all 32 layers.

Reuses the same pipeline as full_layer_attention_analysis.py but:
- Loads finetuned checkpoint via device_map (Blackwell GPU requirement)
- Generates comparison JSON against the 3 existing conditions
"""

import gc
import glob
import json
import os
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

NUM_IMAGE_TOKENS = 256
NUM_LAYERS = 32
NUM_HEADS = 32

FINETUNED_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal/"
OUTPUT_DIR = "./results/attention/"
RESULT_PATH = "./results/libero_finetuned_attention_profile.json"


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def load_model(device_str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        FINETUNED_PATH, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        FINETUNED_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
        device_map=device_str,
    )
    model.eval()
    print(f"Model loaded via device_map={device_str}")
    return model, processor


@torch.no_grad()
def extract_all_layers_attention(model, processor, image, instruction, device):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    attn_storage = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_storage[layer_idx] = output[1][0].cpu().float().numpy()
        return hook_fn

    hooks = []
    for i in range(NUM_LAYERS):
        layer_module = model.language_model.model.layers[i].self_attn
        h = layer_module.register_forward_hook(make_hook(i))
        hooks.append(h)

    outputs = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    first_attn = attn_storage[0]
    actual_seq_len = first_attn.shape[-1]

    bos_range = (0, 1)
    image_range = (1, 1 + NUM_IMAGE_TOKENS)
    text_range = (1 + NUM_IMAGE_TOKENS, actual_seq_len)
    last_preaction_idx = actual_seq_len - 1

    lp_attn_dict = {}
    for layer_idx, attn in attn_storage.items():
        lp_attn_dict[layer_idx] = attn[:, last_preaction_idx, :]

    token_info = {
        "seq_len": actual_seq_len,
        "bos_range": bos_range,
        "image_range": image_range,
        "text_range": text_range,
        "last_preaction_idx": last_preaction_idx,
    }

    del outputs, attn_storage
    return lp_attn_dict, token_info


def load_existing_conditions():
    conditions = {}
    for cond in ["untrained", "llama2base", "trained"]:
        path = os.path.join(OUTPUT_DIR, f"full_32layer_{cond}.json")
        if os.path.exists(path):
            with open(path) as f:
                conditions[cond] = json.load(f)
    return conditions


def build_comparison(finetuned_results, existing):
    comparison = {}
    for layer_idx in range(NUM_LAYERS):
        key = f"L{layer_idx}"
        entry = {
            "instruction_heads": {},
            "mean_text_attn": {},
        }
        for cond_name, cond_data in existing.items():
            if key in cond_data:
                label = {"trained": "openvla7b", "llama2base": "llama2base", "untrained": "untrained"}[cond_name]
                entry["instruction_heads"][label] = cond_data[key]["n_instruction_dominant"]
                entry["mean_text_attn"][label] = cond_data[key]["mean_text_attn"]
        if key in finetuned_results:
            entry["instruction_heads"]["libero_finetuned"] = finetuned_results[key]["n_instruction_dominant"]
            entry["mean_text_attn"]["libero_finetuned"] = finetuned_results[key]["mean_text_attn"]
        comparison[key] = entry
    return comparison


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 files in {DATA_DIR}")

    task_data = []
    for task_idx, hf_path in enumerate(hdf5_files[:10]):
        instruction = task_desc_from_filename(hf_path)
        with h5py.File(hf_path, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys()])
            demo = f["data"][demo_keys[0]]
            obs = demo["obs"]
            img_arr = obs["agentview_rgb"][0]
            img = Image.fromarray(img_arr)
        task_data.append({"task_idx": task_idx, "instruction": instruction, "image": img})

    print(f"Loaded {len(task_data)} tasks, extracting all {NUM_LAYERS} layers")
    print(f"Model: {FINETUNED_PATH}")

    model, processor = load_model(device)

    layer_accum = {i: [] for i in range(NUM_LAYERS)}

    t_start = time.time()
    for td in task_data:
        print(f"\n  [Task {td['task_idx']}] {td['instruction']}")
        t0 = time.time()
        lp_attn_dict, token_info = extract_all_layers_attention(
            model, processor, td["image"], td["instruction"], device
        )
        t1 = time.time()

        bos_start, bos_end = token_info["bos_range"]
        img_start, img_end = token_info["image_range"]
        txt_start, txt_end = token_info["text_range"]

        if td == task_data[0]:
            print(f"    seq_len={token_info['seq_len']}, text_range=({txt_start},{txt_end}), "
                  f"lp_idx={token_info['last_preaction_idx']}")
            print(f"    Forward pass took {t1-t0:.1f}s")

        for layer_idx in range(NUM_LAYERS):
            if layer_idx not in lp_attn_dict:
                continue
            lp_attn = lp_attn_dict[layer_idx]

            attn_to_bos = lp_attn[:, bos_start:bos_end].sum(axis=1)
            attn_to_image = lp_attn[:, img_start:img_end].sum(axis=1)
            attn_to_text = lp_attn[:, txt_start:txt_end].sum(axis=1)

            layer_accum[layer_idx].append(
                np.stack([attn_to_text, attn_to_image, attn_to_bos], axis=1)
            )

        for li in [0, 1, 8, 16, 31]:
            if li in lp_attn_dict:
                lp = lp_attn_dict[li]
                mt = lp[:, txt_start:txt_end].sum(axis=1).mean()
                mi = lp[:, img_start:img_end].sum(axis=1).mean()
                print(f"    L{li}: mean_text={mt:.4f}, mean_image={mi:.4f}")

        del lp_attn_dict
        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nAll tasks done in {elapsed:.1f}s")

    # Aggregate per-layer results
    per_layer = []
    condition_results = {}
    for layer_idx in range(NUM_LAYERS):
        if not layer_accum[layer_idx]:
            continue
        stacked = np.stack(layer_accum[layer_idx], axis=0)
        mean_per_head = stacked.mean(axis=0)

        n_instruction_dominant = 0
        per_head_list = []
        per_head_text_attn = []
        for h in range(NUM_HEADS):
            text_attn = float(mean_per_head[h, 0])
            image_attn = float(mean_per_head[h, 1])
            bos_attn = float(mean_per_head[h, 2])
            per_head_text_attn.append(round(text_attn, 4))
            if text_attn > 0.5:
                n_instruction_dominant += 1
            per_head_list.append({
                "head_id": h,
                "text_attn": round(text_attn, 4),
                "image_attn": round(image_attn, 4),
                "bos_attn": round(bos_attn, 4),
            })

        layer_entry = {
            "layer": layer_idx,
            "n_instruction_dominant_heads": n_instruction_dominant,
            "mean_text_attn": round(float(mean_per_head[:, 0].mean()), 4),
            "mean_image_attn": round(float(mean_per_head[:, 1].mean()), 4),
            "mean_bos_attn": round(float(mean_per_head[:, 2].mean()), 4),
            "per_head": per_head_list,
        }
        per_layer.append(layer_entry)

        condition_results[f"L{layer_idx}"] = {
            "n_instruction_dominant": n_instruction_dominant,
            "mean_text_attn": round(float(mean_per_head[:, 0].mean()), 4),
            "mean_image_attn": round(float(mean_per_head[:, 1].mean()), 4),
            "mean_bos_attn": round(float(mean_per_head[:, 2].mean()), 4),
            "per_head_text_attn": per_head_text_attn,
        }

    # Save in same format as other conditions (for plotting compatibility)
    compat_path = os.path.join(OUTPUT_DIR, "full_32layer_libero_finetuned.json")
    with open(compat_path, "w") as f:
        json.dump(condition_results, f, indent=2)
    print(f"Compatible format saved: {compat_path}")

    # Build comparison with existing conditions
    existing = load_existing_conditions()
    comparison = build_comparison(condition_results, existing)

    # Build final output
    output = {
        "model": "openvla-7b-finetuned-libero-goal",
        "checkpoint_path": FINETUNED_PATH,
        "n_tasks": len(task_data),
        "n_demos": len(task_data),
        "per_layer": per_layer,
        "comparison_with_3conditions": {
            "l1_instruction_heads": comparison.get("L1", {}).get("instruction_heads", {}),
            "l1_mean_text_attn": comparison.get("L1", {}).get("mean_text_attn", {}),
        },
        "full_layer_comparison": comparison,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved: {RESULT_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: libero_finetuned (all 32 layers)")
    print("=" * 70)
    print(f"{'Layer':<6} {'n_dom':>5} {'mean_text':>10} {'mean_image':>10} {'mean_bos':>10}")
    print("-" * 45)
    for i in range(NUM_LAYERS):
        key = f"L{i}"
        d = condition_results[key]
        print(f"{key:<6} {d['n_instruction_dominant']:>5}/32 "
              f"{d['mean_text_attn']:>10.4f} {d['mean_image_attn']:>10.4f} {d['mean_bos_attn']:>10.4f}")


if __name__ == "__main__":
    main()
