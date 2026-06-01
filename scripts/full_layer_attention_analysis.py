"""full_layer_attention_analysis.py — Extract attention across all 32 layers for one condition.

One forward pass per task captures all 32 layers via hooks.
Token layout (Prismatic/OpenVLA): [BOS] [256 image tokens] [text tokens...]
Image tokens are injected by the model (not in input_ids).
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--condition", choices=["trained", "untrained", "llama2base"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default="./results/attention/")
    return p.parse_args()


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def load_model(model_path, condition, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    if condition == "untrained":
        pass  # randomize after moving to GPU

    elif condition == "llama2base":
        print("Replacing LLM backbone with Llama-2-7B base weights...")
        from transformers import AutoModelForCausalLM
        llama2_path = "<HF_CACHE>/models--NousResearch--Llama-2-7b-hf/snapshots/"
        snap_dirs = [d for d in os.listdir(llama2_path) if not d.startswith(".")]
        assert len(snap_dirs) >= 1, f"No Llama-2 snapshot found in {llama2_path}"
        llama2_full_path = os.path.join(llama2_path, snap_dirs[0])
        base_llm = AutoModelForCausalLM.from_pretrained(
            llama2_full_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        base_sd = base_llm.state_dict()
        for name, param in model.language_model.named_parameters():
            if name in base_sd:
                if param.shape == base_sd[name].shape:
                    param.data.copy_(base_sd[name].to(param.dtype))
                elif param.shape[0] > base_sd[name].shape[0]:
                    param.data[:base_sd[name].shape[0]].copy_(base_sd[name].to(param.dtype))
        del base_llm, base_sd
        gc.collect()
        print("LLM backbone replaced with Llama-2-7B base.")

    model = model.to(device).eval()
    if condition == "untrained":
        print("Reinitializing LLM backbone on GPU...")
        with torch.no_grad():
            for name, param in model.language_model.named_parameters():
                if param.ndim >= 2:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
                else:
                    torch.nn.init.zeros_(param.data)
        print("LLM backbone randomized.")
    print(f"Model loaded on {device} (condition={condition})")
    return model, processor


@torch.no_grad()
def extract_all_layers_attention(model, processor, image, instruction, device):
    """Forward pass returning attention weights for ALL 32 layers."""
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

    # Get actual seq_len from attention weights (includes injected image tokens)
    first_attn = attn_storage[0]  # (n_heads, actual_seq_len, actual_seq_len)
    actual_seq_len = first_attn.shape[-1]

    # Token layout: [BOS] [256 image tokens] [text tokens...]
    bos_range = (0, 1)
    image_range = (1, 1 + NUM_IMAGE_TOKENS)
    text_range = (1 + NUM_IMAGE_TOKENS, actual_seq_len)
    last_preaction_idx = actual_seq_len - 1

    # Extract only last_preaction row from each layer
    lp_attn_dict = {}
    for layer_idx, attn in attn_storage.items():
        lp_attn_dict[layer_idx] = attn[:, last_preaction_idx, :]  # (n_heads, seq_len)

    token_info = {
        "seq_len": actual_seq_len,
        "bos_range": bos_range,
        "image_range": image_range,
        "text_range": text_range,
        "last_preaction_idx": last_preaction_idx,
    }

    del outputs, attn_storage
    return lp_attn_dict, token_info


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 files in {args.data_dir}")

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
    print(f"Condition: {args.condition}")

    model, processor = load_model(args.model_path, args.condition, device)

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

    condition_results = {}
    for layer_idx in range(NUM_LAYERS):
        if not layer_accum[layer_idx]:
            continue
        stacked = np.stack(layer_accum[layer_idx], axis=0)
        mean_per_head = stacked.mean(axis=0)

        n_instruction_dominant = 0
        per_head_text_attn = []
        for h in range(NUM_HEADS):
            text_attn = float(mean_per_head[h, 0])
            per_head_text_attn.append(round(text_attn, 4))
            if text_attn > 0.5:
                n_instruction_dominant += 1

        condition_results[f"L{layer_idx}"] = {
            "n_instruction_dominant": n_instruction_dominant,
            "mean_text_attn": round(float(mean_per_head[:, 0].mean()), 4),
            "mean_image_attn": round(float(mean_per_head[:, 1].mean()), 4),
            "mean_bos_attn": round(float(mean_per_head[:, 2].mean()), 4),
            "per_head_text_attn": per_head_text_attn,
        }

    output_path = os.path.join(args.output_dir, f"full_32layer_{args.condition}.json")
    with open(output_path, "w") as f:
        json.dump(condition_results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {args.condition} (all 32 layers)")
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
