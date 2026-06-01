"""Attention profile for TracVLA-Phi3V across all 32 layers."""

import gc
import json
import os
import sys
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

NUM_LAYERS = 32
NUM_HEADS = 32

CHECKPOINT_PATH = "./checkpoints/tracevla-phi3v/"
DATA_DIR = "./data/libero/libero_goal/"
OUTPUT_DIR = "./results/attention/"
RESULT_PATH = "./results/attention/phi3v_attention_profile.json"


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def load_model(device_str):
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT_PATH, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
        device_map=device_str,
    )
    model.eval()
    print(f"Model loaded on {device_str}")
    return model, processor


def detect_image_boundaries(input_ids_list):
    """Detect image token start/end in Phi3V input_ids."""
    neg_pos = [i for i, tid in enumerate(input_ids_list) if tid < 0]
    if len(neg_pos) >= 50:
        return neg_pos[0], neg_pos[-1] + 1

    from collections import Counter
    counts = Counter(input_ids_list)
    for tid, count in counts.most_common():
        if 50 <= count <= 800:
            positions = [i for i, t in enumerate(input_ids_list) if t == tid]
            if positions[-1] - positions[0] + 1 == count:
                return positions[0], positions[-1] + 1

    raise ValueError(f"Cannot detect image tokens. Top counts: {counts.most_common(10)}")


@torch.no_grad()
def extract_attention(model, processor, image, instruction, device):
    prompt = f"<|user|>\n<|image_1|>\nWhat action should the robot take to {instruction}?<|end|>\n<|assistant|>\n"
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    input_ids = inputs["input_ids"][0].cpu().tolist()
    img_start, img_end = detect_image_boundaries(input_ids)
    n_img_tokens = img_end - img_start

    attn_storage = {}
    def make_hook(idx):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_storage[idx] = out[1][0].cpu().float().numpy()
        return hook_fn

    hooks = []
    for i in range(NUM_LAYERS):
        h = model.model.layers[i].self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    outputs = model(**inputs, output_attentions=True)
    for h in hooks:
        h.remove()

    actual_seq_len = attn_storage[0].shape[-1]
    if actual_seq_len != len(input_ids):
        expansion = actual_seq_len - len(input_ids)
        img_end = img_start + n_img_tokens + expansion
        n_img_tokens = img_end - img_start

    last_idx = actual_seq_len - 1
    lp_attn = {}
    for layer_idx, attn in attn_storage.items():
        lp_attn[layer_idx] = attn[:, last_idx, :]

    bounds = {"img_start": img_start, "img_end": img_end,
              "seq_len": actual_seq_len, "n_img_tokens": n_img_tokens}
    del outputs, attn_storage
    return lp_attn, bounds


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dry_run = "--dry-run" in sys.argv

    print("Loading TracVLA-Phi3V...")
    model, processor = load_model(device)

    hdf5_files = sorted(Path(DATA_DIR).glob("*_demo.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 in {DATA_DIR}")

    task_data = []
    for hdf5_path in hdf5_files[:5]:
        task_name = task_desc_from_filename(hdf5_path.name)
        with h5py.File(str(hdf5_path), "r") as f:
            demo_key = list(f["data"].keys())[0]
            obs = f["data"][demo_key]["obs"]
            for img_key in ["agentview_rgb", "agentview_image"]:
                if img_key in obs:
                    image = Image.fromarray(obs[img_key][0])
                    break
            else:
                raise KeyError(f"No image key in {list(obs.keys())}")
        task_data.append((task_name, image))
        print(f"Loaded: {task_name}")

    if dry_run:
        print("\n--- DRY RUN ---")
        task_name, image = task_data[0]
        prompt = f"<|user|>\n<|image_1|>\nWhat action should the robot take to {task_name}?<|end|>\n<|assistant|>\n"
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        print(f"input_ids length: {len(ids)}")
        print(f"First 20 IDs: {ids[:20]}")
        print(f"Last 10 IDs: {ids[-10:]}")
        print(f"Negative IDs: {sum(1 for t in ids if t < 0)}")
        from collections import Counter
        counts = Counter(ids)
        print(f"Top counts: {counts.most_common(10)}")
        img_s, img_e = detect_image_boundaries(ids)
        print(f"Image: [{img_s}, {img_e}) = {img_e - img_s} tokens")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        outputs = model(**inputs, output_attentions=True)
        if outputs.attentions:
            print(f"Attention[0] shape: {outputs.attentions[0].shape}")
        else:
            print("WARNING: attentions empty, checking hooks...")
        del outputs
        print("--- DRY RUN OK ---")
        return

    print(f"\nProcessing {len(task_data)} tasks x {NUM_LAYERS} layers...")
    layer_accum = {i: [] for i in range(NUM_LAYERS)}
    t_start = time.time()

    for task_idx, (task_name, image) in enumerate(task_data):
        print(f"\n[{task_idx+1}/{len(task_data)}] {task_name}")
        lp_attn, bounds = extract_attention(model, processor, image, task_name, device)
        img_s, img_e = bounds["img_start"], bounds["img_end"]

        for li in range(NUM_LAYERS):
            if li not in lp_attn:
                continue
            lp = lp_attn[li]
            attn_to_image = lp[:, img_s:img_e].sum(axis=1)
            attn_to_text = lp[:, :img_s].sum(axis=1) + lp[:, img_e:].sum(axis=1)
            layer_accum[li].append(np.stack([attn_to_text, attn_to_image], axis=1))

        for li in [0, 1, 8, 16, 31]:
            if li in lp_attn:
                lp = lp_attn[li]
                mt = lp[:, :img_s].sum(axis=1).mean() + lp[:, img_e:].sum(axis=1).mean()
                mi = lp[:, img_s:img_e].sum(axis=1).mean()
                print(f"    L{li}: text={mt:.4f}, image={mi:.4f}")

        del lp_attn
        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")

    condition_results = {}
    for li in range(NUM_LAYERS):
        if not layer_accum[li]:
            continue
        stacked = np.stack(layer_accum[li], axis=0).mean(axis=0)
        per_head_text = [round(float(stacked[h, 0]), 4) for h in range(NUM_HEADS)]
        n_dom = sum(1 for v in per_head_text if v > 0.5)
        condition_results[f"L{li}"] = {
            "n_instruction_dominant": n_dom,
            "mean_text_attn": round(float(stacked[:, 0].mean()), 4),
            "mean_image_attn": round(float(stacked[:, 1].mean()), 4),
            "per_head_text_attn": per_head_text,
        }

    compat_path = os.path.join(OUTPUT_DIR, "full_32layer_phi3v.json")
    with open(compat_path, "w") as f:
        json.dump(condition_results, f, indent=2)

    per_layer_out = {}
    for li in range(NUM_LAYERS):
        key = f"L{li}"
        if key in condition_results:
            d = condition_results[key]
            per_layer_out[key] = {
                "instruction_dominant_heads": d["n_instruction_dominant"],
                "mean_text_attention": d["mean_text_attn"],
                "per_head_text_attention": d["per_head_text_attn"],
            }

    l1 = condition_results.get("L1", {})
    output = {
        "model": "tracevla-phi3v",
        "layers": NUM_LAYERS,
        "heads": NUM_HEADS,
        "per_layer": per_layer_out,
        "summary": {
            "L1_instruction_dominant": l1.get("n_instruction_dominant", 0),
            "L1_mean_text_attn": l1.get("mean_text_attn", 0.0),
        },
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {RESULT_PATH}")

    print("\n" + "=" * 60)
    print("TracVLA-Phi3V Attention Profile (32 layers)")
    print("=" * 60)
    print(f"{'Layer':<6} {'n_dom':>5} {'text':>8} {'image':>8}")
    print("-" * 30)
    for i in range(NUM_LAYERS):
        d = condition_results[f"L{i}"]
        print(f"L{i:<5} {d['n_instruction_dominant']:>5}/32 "
              f"{d['mean_text_attn']:>8.4f} {d['mean_image_attn']:>8.4f}")
    print(f"\nL1: {l1.get('n_instruction_dominant', 'N/A')}/32 dominant, "
          f"mean_text={l1.get('mean_text_attn', 'N/A')}")


if __name__ == "__main__":
    main()
