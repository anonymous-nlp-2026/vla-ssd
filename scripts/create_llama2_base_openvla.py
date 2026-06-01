"""create_llama2_base_openvla.py — Replace OpenVLA LLM backbone with Llama-2-7B base weights.

Creates a control checkpoint: trained vision encoder + projector, but base (pretrained, not
VLA-finetuned) LLM backbone. Tests whether instruction routing emerges from language
pretraining vs VLA fine-tuning.

Weight mapping:
  Llama-2 key "X" → OpenVLA key "language_model.X"
  
Vocab mismatch handling (OpenVLA=32064 vs Llama-2=32000):
  embed_tokens/lm_head: first 32000 rows from Llama-2 base, rows 32000-32063 kept from trained OpenVLA.

Usage:
  python create_llama2_base_openvla.py \
    --openvla_path ./checkpoints/openvla-7b \
    --llama2_path <HF_CACHE>/models--NousResearch--Llama-2-7b-hf/snapshots/<hash> \
    --output_path ./checkpoints/openvla-7b-llama2base
"""

import argparse
import gc
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser(description="Replace OpenVLA LLM with Llama-2-7B base")
    p.add_argument("--openvla_path", required=True)
    p.add_argument("--llama2_path", required=True)
    p.add_argument("--output_path", required=True)
    return p.parse_args()


def load_llama2_state_dict(llama2_path):
    """Load all Llama-2 safetensors shards into a single state dict."""
    index_path = os.path.join(llama2_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            f for f in os.listdir(llama2_path) if f.endswith(".safetensors")
        )

    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(llama2_path, shard)
        print(f"  Loading Llama-2 shard: {shard}")
        sd = load_file(shard_path)
        state_dict.update(sd)
    return state_dict


def load_openvla_state_dict(openvla_path):
    """Load all OpenVLA safetensors shards."""
    index_path = os.path.join(openvla_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(openvla_path, shard)
        print(f"  Loading OpenVLA shard: {shard}")
        sd = load_file(shard_path)
        state_dict.update(sd)
    return state_dict


def inject_llama2_weights(openvla_sd, llama2_sd):
    """Replace language_model.* keys in openvla_sd with Llama-2 base weights."""
    replaced = []
    partial_replaced = []
    kept = []
    missing_in_llama2 = []

    for key in list(openvla_sd.keys()):
        if not key.startswith("language_model."):
            kept.append(key)
            continue

        llama2_key = key[len("language_model."):]

        if llama2_key not in llama2_sd:
            missing_in_llama2.append(key)
            kept.append(key)
            continue

        openvla_tensor = openvla_sd[key]
        llama2_tensor = llama2_sd[llama2_key]

        if openvla_tensor.shape == llama2_tensor.shape:
            openvla_sd[key] = llama2_tensor.to(openvla_tensor.dtype)
            replaced.append(key)
        elif openvla_tensor.shape[0] > llama2_tensor.shape[0]:
            # Vocab extension: copy first N rows from Llama-2, keep extra rows from OpenVLA
            n_base = llama2_tensor.shape[0]
            new_tensor = openvla_tensor.clone()
            new_tensor[:n_base] = llama2_tensor.to(openvla_tensor.dtype)
            openvla_sd[key] = new_tensor
            partial_replaced.append(
                f"{key}: [{n_base}/{openvla_tensor.shape[0]}] rows from Llama-2"
            )
        else:
            missing_in_llama2.append(f"{key} (shape mismatch: {openvla_tensor.shape} vs {llama2_tensor.shape})")
            kept.append(key)

    return replaced, partial_replaced, kept, missing_in_llama2


def save_checkpoint(state_dict, openvla_path, output_path):
    """Save state dict as sharded safetensors + copy config/processor files."""
    os.makedirs(output_path, exist_ok=True)

    # Split into 3 shards (same as original for consistency)
    keys = sorted(state_dict.keys())
    n = len(keys)
    shard_size = (n + 2) // 3
    shards = [keys[i:i+shard_size] for i in range(0, n, shard_size)]

    weight_map = {}
    for i, shard_keys in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        shard_dict = {k: state_dict[k] for k in shard_keys}
        save_file(shard_dict, os.path.join(output_path, shard_name))
        for k in shard_keys:
            weight_map[k] = shard_name
        print(f"  Saved {shard_name} ({len(shard_keys)} tensors)")

    # Write index
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Copy config and processor files
    copy_files = [
        "config.json", "configuration_prismatic.py", "modeling_prismatic.py",
        "processing_prismatic.py", "preprocessor_config.json", "processor_config.json",
        "generation_config.json", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    ]
    for fname in copy_files:
        src = os.path.join(openvla_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))


def main():
    args = parse_args()

    print(f"OpenVLA path: {args.openvla_path}")
    print(f"Llama-2 path: {args.llama2_path}")
    print(f"Output path:  {args.output_path}")

    # Verify paths exist
    assert os.path.isdir(args.openvla_path), f"OpenVLA path not found: {args.openvla_path}"
    assert os.path.isdir(args.llama2_path), f"Llama-2 path not found: {args.llama2_path}"

    print("\n[1/4] Loading Llama-2-7B base weights...")
    llama2_sd = load_llama2_state_dict(args.llama2_path)
    print(f"  Loaded {len(llama2_sd)} tensors")

    print("\n[2/4] Loading OpenVLA weights...")
    openvla_sd = load_openvla_state_dict(args.openvla_path)
    print(f"  Loaded {len(openvla_sd)} tensors")

    print("\n[3/4] Injecting Llama-2 base weights into LLM backbone...")
    replaced, partial, kept, missing = inject_llama2_weights(openvla_sd, llama2_sd)
    del llama2_sd
    gc.collect()

    print(f"\n  Summary:")
    print(f"    Fully replaced:    {len(replaced)} tensors")
    print(f"    Partially replaced: {len(partial)} tensors")
    for p in partial:
        print(f"      {p}")
    print(f"    Kept (non-LLM):    {len([k for k in kept if not k.startswith('language_model.')])}")
    print(f"    Missing in Llama-2: {len(missing)}")
    for m in missing:
        print(f"      {m}")

    print("\n[4/4] Saving checkpoint...")
    save_checkpoint(openvla_sd, args.openvla_path, args.output_path)

    total_gb = sum(t.numel() * t.element_size() for t in openvla_sd.values()) / 1e9
    print(f"\n  Done. Total size: {total_gb:.1f} GB")
    print(f"  Saved to: {args.output_path}")

    # Verification: print sample norms
    print("\n  Verification (sample weight norms):")
    for key in sorted(openvla_sd.keys()):
        if "layers.0.self_attn.q_proj" in key:
            print(f"    LLM L0 q_proj norm: {openvla_sd[key].float().norm().item():.2f}")
            break
    for key in sorted(openvla_sd.keys()):
        if "vision_backbone" in key and "weight" in key and openvla_sd[key].dim() >= 2:
            print(f"    Vision sample norm: {openvla_sd[key].float().norm().item():.2f} ({key})")
            break
    for key in sorted(openvla_sd.keys()):
        if "projector" in key and "weight" in key:
            print(f"    Projector sample norm: {openvla_sd[key].float().norm().item():.2f} ({key})")
            break


if __name__ == "__main__":
    main()
