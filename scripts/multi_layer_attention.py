"""multi_layer_attention.py — Extract attention distributions across multiple layers and conditions.

For each (condition, layer, head), computes the attention proportion from
last_preaction token to instruction tokens vs image tokens.

Token layout (Prismatic/OpenVLA): [BOS] [256 image tokens] [text tokens...]
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--conditions", default="trained,untrained,llama2base")
    p.add_argument("--layers", default="8,16,31")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="./results/attention/multi_layer_attention_summary.json")
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
        print("Reinitializing LLM backbone...")
        from transformers import AutoModelForCausalLM
        llm_cfg = model.language_model.config
        random_llm = AutoModelForCausalLM.from_config(llm_cfg, torch_dtype=torch.bfloat16)
        model.language_model.load_state_dict(random_llm.state_dict())
        del random_llm
        gc.collect()

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
    print(f"Model loaded on {device} (condition={condition})")
    return model, processor


@torch.no_grad()
def extract_multi_layer_attention(model, processor, image, instruction, device, target_layers):
    """Forward pass returning attention weights for multiple layers."""
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    attn_storage = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_storage[layer_idx] = output[1].detach().cpu().float()
        return hook_fn

    model.language_model.config.output_attentions = True

    hooks = []
    for layer_idx in target_layers:
        h = model.language_model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=False, output_attentions=True)

    for h in hooks:
        h.remove()
    model.language_model.config.output_attentions = False

    # Fallback: if hooks didn't capture, try out.attentions
    if hasattr(out, "attentions") and out.attentions is not None:
        for layer_idx in target_layers:
            if layer_idx not in attn_storage:
                attn_storage[layer_idx] = out.attentions[layer_idx].detach().cpu().float()

    # Get seq_len from attention matrix (NOT from input_ids which excludes image tokens)
    first_layer = target_layers[0]
    attn_seq_len = attn_storage[first_layer].shape[-1]

    # Token layout: [BOS(1)] [image(256)] [text tokens...]
    token_info = {
        "bos_range": (0, 1),
        "image_range": (1, 1 + NUM_IMAGE_TOKENS),
        "text_range": (1 + NUM_IMAGE_TOKENS, attn_seq_len),
        "seq_len": attn_seq_len,
        "last_preaction_idx": attn_seq_len - 1,
    }

    result = {}
    for layer_idx in target_layers:
        if layer_idx in attn_storage:
            attn = attn_storage[layer_idx].squeeze(0).numpy()
            result[layer_idx] = attn
        else:
            print(f"  WARNING: Could not capture attention for layer {layer_idx}")

    del out, inputs
    torch.cuda.empty_cache()
    return result, token_info


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    conditions = [c.strip() for c in args.conditions.split(",")]
    target_layers = [int(x.strip()) for x in args.layers.split(",")]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} tasks")

    task_indices = [0, 2, 4, 6, 8]
    demo_idx = 0
    frame_idx = 25

    task_data = []
    for task_idx in task_indices:
        if task_idx >= len(hdf5_files):
            continue
        hdf5_path = hdf5_files[task_idx]
        instruction = task_desc_from_filename(hdf5_path)
        f = h5py.File(hdf5_path, "r")
        demo_key = f"demo_{demo_idx}"
        images_ds = f["data"][demo_key]["obs"]["agentview_rgb"]
        T = images_ds.shape[0]
        actual_frame = min(frame_idx, T - 1)
        img = Image.fromarray(images_ds[actual_frame])
        f.close()
        task_data.append({"task_idx": task_idx, "instruction": instruction, "image": img})

    print(f"Loaded {len(task_data)} tasks, layers={target_layers}")

    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition}")
        print(f"{'='*60}")

        model, processor = load_model(args.model_path, condition, device)

        layer_accum = {l: [] for l in target_layers}

        for td in task_data:
            print(f"\n  [Task {td['task_idx']}] {td['instruction']}")
            attn_dict, token_info = extract_multi_layer_attention(
                model, processor, td["image"], td["instruction"], device, target_layers
            )

            bos_start, bos_end = token_info["bos_range"]
            img_start, img_end = token_info["image_range"]
            txt_start, txt_end = token_info["text_range"]
            lp_idx = token_info["last_preaction_idx"]

            if td == task_data[0]:
                print(f"    seq_len={token_info['seq_len']}, text_range=({txt_start},{txt_end}), lp_idx={lp_idx}")

            for layer_idx in target_layers:
                if layer_idx not in attn_dict:
                    continue
                attn = attn_dict[layer_idx]
                lp_attn = attn[:, lp_idx, :]

                attn_to_bos = lp_attn[:, bos_start:bos_end].sum(axis=1)
                attn_to_image = lp_attn[:, img_start:img_end].sum(axis=1)
                attn_to_text = lp_attn[:, txt_start:txt_end].sum(axis=1)

                layer_accum[layer_idx].append(
                    np.stack([attn_to_text, attn_to_image, attn_to_bos], axis=1)
                )

                print(f"    L{layer_idx}: mean_text={attn_to_text.mean():.4f}, "
                      f"mean_image={attn_to_image.mean():.4f}, mean_bos={attn_to_bos.mean():.4f}")

        condition_results = {}
        for layer_idx in target_layers:
            if not layer_accum[layer_idx]:
                continue
            stacked = np.stack(layer_accum[layer_idx], axis=0)
            mean_per_head = stacked.mean(axis=0)
            n_heads = mean_per_head.shape[0]

            heads_list = []
            n_instruction_dominant = 0
            for h in range(n_heads):
                text_attn = float(mean_per_head[h, 0])
                image_attn = float(mean_per_head[h, 1])
                bos_attn = float(mean_per_head[h, 2])
                heads_list.append({
                    "head_id": h,
                    "text_attn": round(text_attn, 4),
                    "image_attn": round(image_attn, 4),
                    "bos_attn": round(bos_attn, 4),
                })
                if text_attn > 0.5:
                    n_instruction_dominant += 1

            condition_results[f"L{layer_idx}"] = {
                "heads": heads_list,
                "n_instruction_dominant": n_instruction_dominant,
                "mean_text_attn": round(float(mean_per_head[:, 0].mean()), 4),
                "mean_image_attn": round(float(mean_per_head[:, 1].mean()), 4),
                "mean_bos_attn": round(float(mean_per_head[:, 2].mean()), 4),
            }

        all_results[condition] = condition_results

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  Model freed for condition={condition}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {args.output}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for condition in conditions:
        print(f"\n  {condition}:")
        for layer_idx in target_layers:
            key = f"L{layer_idx}"
            d = all_results[condition][key]
            print(f"    {key}: n_instruction_dominant={d['n_instruction_dominant']}/32, "
                  f"mean_text={d['mean_text_attn']:.4f}, "
                  f"mean_image={d['mean_image_attn']:.4f}, "
                  f"mean_bos={d['mean_bos_attn']:.4f}")


if __name__ == "__main__":
    main()
