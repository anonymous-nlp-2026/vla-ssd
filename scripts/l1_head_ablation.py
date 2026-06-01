"""L1 Head Ablation: Zero out instruction attention for specific heads in Layer 1,
then measure cross-instruction classification accuracy at multiple readout layers."""

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
import torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS  # BOS(1) + image(256) = 257

# From prior L1 attention analysis: intersection of instruction-attending heads across all tasks
INSTRUCTION_HEADS_24 = [0, 1, 2, 4, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 29, 30, 31]
# Complement: heads NOT consistently attending to instruction
CONTROL_HEADS_8 = [3, 5, 6, 9, 13, 21, 24, 28]

READOUT_LAYERS = [1, 8, 16, 31]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--output", default="./results/l1_head_ablation/results.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--demos_per_task", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit total samples for dry-run (e.g. 2)")
    p.add_argument("--condition", type=str, default=None,
                   help="Run only this condition (e.g. ablate_all32)")
    return p.parse_args()


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def install_ablation_hook(model, heads_to_ablate):
    import transformers.models.llama.modeling_llama as llama_mod
    from transformers.models.llama.modeling_llama import repeat_kv

    layer1_attn = model.language_model.model.layers[1].self_attn
    layer1_attn._ablate_heads = heads_to_ablate

    original_fn = llama_mod.eager_attention_forward

    def ablated_eager_attention_forward(module, query, key, value, attention_mask,
                                        scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        if hasattr(module, '_ablate_heads') and module._ablate_heads is not None:
            seq_len = attn_weights.shape[-1]
            if INST_START < seq_len:
                attn_weights[:, module._ablate_heads, :, INST_START:] = 0.0
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn_weights = attn_weights / row_sums

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    llama_mod.eager_attention_forward = ablated_eager_attention_forward
    return original_fn


def remove_ablation_hook(model, original_fn):
    import transformers.models.llama.modeling_llama as llama_mod
    llama_mod.eager_attention_forward = original_fn
    layer1_attn = model.language_model.model.layers[1].self_attn
    layer1_attn._ablate_heads = None


def load_model(model_path, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model, processor


@torch.no_grad()
def extract_features(model, processor, images, instructions, device):
    features = {l: [] for l in READOUT_LAYERS}
    hooks = []
    hook_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            hook_outputs[layer_idx] = hidden[:, -1, :].detach().cpu().float()
        return hook_fn

    for l in READOUT_LAYERS:
        h = model.language_model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    try:
        for img_idx, (image, task_idx, demo_idx) in enumerate(images):
            for inst_idx, instruction in enumerate(instructions):
                prompt = f"In: What action should the robot take to {instruction}?\nOut:"
                inputs = processor(text=[prompt], images=[image], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    model(**inputs)

                for l in READOUT_LAYERS:
                    features[l].append(hook_outputs[l].squeeze(0).numpy())

                del inputs
                hook_outputs.clear()

            if (img_idx + 1) % 10 == 0:
                print(f"  Processed {img_idx + 1}/{len(images)} images")
    finally:
        for h in hooks:
            h.remove()

    for l in READOUT_LAYERS:
        features[l] = np.array(features[l])

    return features


def classify(features, task_indices, n_tasks, n_instructions):
    """10-way instruction classification. Train: tasks 0..(n-3), Test: tasks (n-2)..(n-1)."""
    n_images = len(task_indices)

    task_indices_full = np.repeat(task_indices, n_instructions)
    instruction_labels = np.tile(np.arange(n_instructions), n_images)

    train_mask = task_indices_full < (n_tasks - 2)
    test_mask = task_indices_full >= (n_tasks - 2)

    results = {}
    for l in READOUT_LAYERS:
        X = features[l]
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = instruction_labels[train_mask]
        y_test = instruction_labels[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results[f"L{l}"] = -1.0
            continue

        scaler = StandardScaler()
        pca = PCA(n_components=256)
        X_train_s = pca.fit_transform(scaler.fit_transform(X_train))
        X_test_s = pca.transform(scaler.transform(X_test))

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_train_s, y_train)
        acc = clf.score(X_test_s, y_test)
        results[f"L{l}"] = round(float(acc), 4)

    return results


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dry_run = args.num_samples is not None

    print("=== L1 Head Ablation Experiment ===")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output}")
    if dry_run:
        print(f"*** DRY RUN: {args.num_samples} samples ***")

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    n_tasks = len(hdf5_files)
    instructions = [task_desc_from_filename(f) for f in hdf5_files]
    print(f"Tasks: {n_tasks}")
    for i, inst in enumerate(instructions):
        print(f"  [{i}] {inst}")

    images = []
    task_indices = []
    demos_per_task = args.demos_per_task

    for task_idx, hdf5_path in enumerate(hdf5_files):
        f = h5py.File(hdf5_path, "r")
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        demo_keys = demo_keys[:demos_per_task]

        for local_demo_idx, dk in enumerate(demo_keys):
            img_data = data[dk]["obs"]["agentview_rgb"][0]
            img = Image.fromarray(img_data)
            images.append((img, task_idx, local_demo_idx))
            task_indices.append(task_idx)

        f.close()

    task_indices = np.array(task_indices)
    print(f"Total images: {len(images)}")

    if dry_run:
        images = images[:args.num_samples]
        task_indices = task_indices[:args.num_samples]
        n_instructions_use = min(2, n_tasks)
        instructions_use = instructions[:n_instructions_use]
    else:
        instructions_use = instructions
        n_instructions_use = n_tasks

    model, processor = load_model(args.model_path, device)

    import transformers.models.llama.modeling_llama as llama_mod
    original_eager_fn = llama_mod.eager_attention_forward

    conditions = {
        "baseline": None,
        "ablate_all24": INSTRUCTION_HEADS_24,
        "ablate_top5": [0, 10, 14, 22, 25],
        "ablate_control8": CONTROL_HEADS_8,
        "ablate_all32": list(range(32)),
    }

    if args.condition:
        if args.condition not in conditions:
            raise ValueError(f"Unknown condition: {args.condition}. Available: {list(conditions.keys())}")
        conditions = {args.condition: conditions[args.condition]}

    all_results = {}

    for cond_name, heads in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (heads={heads})")
        print(f"{'='*60}")
        t0 = time.time()

        if heads is not None:
            install_ablation_hook(model, heads)
        else:
            model.language_model.model.layers[1].self_attn._ablate_heads = None
            llama_mod.eager_attention_forward = original_eager_fn

        features = extract_features(model, processor, images, instructions_use, device)
        acc_results = classify(features, task_indices, n_tasks, n_instructions_use)

        elapsed = time.time() - t0
        print(f"  Results: {acc_results}")
        print(f"  Time: {elapsed:.1f}s")

        all_results[cond_name] = acc_results

        remove_ablation_hook(model, original_eager_fn)

        del features
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "conditions": all_results,
        "metadata": {
            "num_images": len(images),
            "num_tasks": n_tasks,
            "num_instructions": n_instructions_use,
            "demos_per_task": args.demos_per_task,
            "readout_layers": READOUT_LAYERS,
            "instruction_heads_24": INSTRUCTION_HEADS_24,
            "control_heads_8": CONTROL_HEADS_8,
            "ablate_top5_heads": [0, 10, 14, 22, 25],
            "checkpoint": args.model_path,
            "train_tasks": "0-7",
            "test_tasks": "8-9",
            "label": "instruction_id (0-9)",
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
