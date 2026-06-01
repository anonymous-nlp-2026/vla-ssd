"""Attention profile for untrained (base Phi-3 LLM) in TracVLA-Phi3V architecture, 10 LIBERO-Goal tasks."""

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

TRACVLA_PATH = "./checkpoints/tracevla-phi3v/"
BASE_PHI3_PATHS = [
    "<HF_CACHE>/Phi-3-mini-4k-instruct",
    "<HF_CACHE>/models--microsoft--Phi-3-mini-4k-instruct",
]
DATA_DIR = "./data/libero/libero_goal/"
OUTPUT_DIR = "./results/attention/"
RESULT_PATH = "./results/attention/phi3v_attention_untrained_10tasks.json"


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def find_base_phi3():
    for p in BASE_PHI3_PATHS:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    for p in BASE_PHI3_PATHS:
        parent = Path(p)
        if parent.exists():
            for child in parent.rglob("config.json"):
                return str(child.parent)
    raise FileNotFoundError(f"Base Phi-3-mini not found in {BASE_PHI3_PATHS}")


def load_untrained_model(device_str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        TRACVLA_PATH, trust_remote_code=True, local_files_only=True
    )

    print("Loading TracVLA-Phi3V architecture (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        TRACVLA_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    base_path = find_base_phi3()
    print(f"Loading base Phi-3-mini from {base_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Replacing LLM decoder layer weights (32 layers)...")
    for i in range(NUM_LAYERS):
        model.model.layers[i].load_state_dict(base_model.model.layers[i].state_dict())

    try:
        model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        print("Replaced embed_tokens + lm_head")
    except RuntimeError as e:
        print(f"Skipping embed_tokens/lm_head (dim mismatch): {e}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device_str)
    model.eval()
    print(f"Untrained hybrid model on {device_str}")
    return model, processor


def detect_image_boundaries(input_ids_list):
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

    print("Loading untrained (base Phi-3 LLM) model...")
    model, processor = load_untrained_model(device)

    hdf5_files = sorted(Path(DATA_DIR).glob("*_demo.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {DATA_DIR}")
    print(f"Found {len(hdf5_files)} tasks")

    task_data = []
    for hf in hdf5_files:
        task_name = task_desc_from_filename(hf.name)
        with h5py.File(str(hf), "r") as f:
            demo_keys = sorted([k for k in f["data"].keys()])
            demo = f["data"][demo_keys[0]]
            frame = demo["obs"]["agentview_rgb"][0]
            image = Image.fromarray(frame)
        task_data.append((task_name, image))
        print(f"  {task_name}")

    if dry_run:
        print("\n--- DRY RUN: testing forward pass ---")
        task_name, image = task_data[0]
        prompt = f"<|user|>\n<|image_1|>\nWhat action should the robot take to {task_name}?<|end|>\n<|assistant|>\n"
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        print(f"Input IDs shape: {inputs['input_ids'].shape}, {inputs['input_ids'].shape[1]} tokens")
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

    compat_path = os.path.join(OUTPUT_DIR, "full_32layer_phi3v_untrained_10tasks.json")
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
        "model": "phi3v-untrained-base",
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
    print("Phi3V UNTRAINED (base Phi-3 LLM) Attention Profile - 10 tasks")
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
