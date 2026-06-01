"""SmolVLA architecture exploration and dry-run on LIBERO-Goal."""

import gc
import os
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

SMOLVLA_CKPT = "./checkpoints/smolvla_libero"
VLM_DIR = "./checkpoints/smolvlm2-500m"
DATA_DIR = "./data/libero/libero_goal"
DEVICE = "cuda:1"


def inspect_checkpoint():
    print("=" * 60)
    print("STEP 1: Inspect SmolVLA checkpoint")
    print("=" * 60)

    from safetensors import safe_open
    ckpt_path = os.path.join(SMOLVLA_CKPT, "model.safetensors")
    with safe_open(ckpt_path, framework="pt") as f:
        keys = sorted(f.keys())

    print(f"Total tensors: {len(keys)}")

    groups = {}
    for k in keys:
        parts = k.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else k
        groups[prefix] = groups.get(prefix, 0) + 1

    print("\nWeight groups:")
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c}")

    text_layers, vision_layers, expert_layers = set(), set(), set()
    for k in keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                lid = int(parts[i+1])
                if "text_model" in k: text_layers.add(lid)
                elif "vision" in k: vision_layers.add(lid)
                elif "expert" in k or "lm_expert" in k: expert_layers.add(lid)

    print(f"\nVLM text layers: {sorted(text_layers)} ({len(text_layers)})")
    print(f"Vision encoder layers: {sorted(vision_layers)} ({len(vision_layers)})")
    print(f"Expert layers: {sorted(expert_layers)} ({len(expert_layers)})")

    print("\n--- Key dimensions ---")
    with safe_open(ckpt_path, framework="pt") as f:
        for k in sorted(keys):
            if any(x in k for x in ["q_proj.weight", "embed_tokens", "lm_head", "action_in", "action_out", "action_time", "state_proj", "connector", "patch_embedding"]):
                t = f.get_tensor(k)
                print(f"  {k}: {list(t.shape)} {t.dtype}")


def load_vlm_model():
    print("\n" + "=" * 60)
    print("STEP 2: Load SmolVLM2-500M backbone")
    print("=" * 60)

    from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor, AutoConfig

    print(f"Loading from {VLM_DIR}...")
    config = AutoConfig.from_pretrained(VLM_DIR, local_files_only=True, trust_remote_code=True)
    tc = config.text_config
    vc = config.vision_config

    print(f"\n=== SmolLM2 text backbone ===")
    print(f"  hidden_size: {tc.hidden_size}")
    print(f"  num_hidden_layers: {tc.num_hidden_layers}")
    print(f"  num_attention_heads: {tc.num_attention_heads}")
    print(f"  num_key_value_heads: {tc.num_key_value_heads}")
    head_dim = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)
    print(f"  head_dim: {head_dim}")
    print(f"  intermediate_size: {tc.intermediate_size}")
    print(f"  vocab_size: {tc.vocab_size}")

    print(f"\n=== SigLIP vision encoder ===")
    print(f"  hidden_size: {vc.hidden_size}")
    print(f"  image_size: {vc.image_size}")
    print(f"  patch_size: {vc.patch_size}")
    print(f"  num_attention_heads: {vc.num_attention_heads}")
    n_patches = (vc.image_size // vc.patch_size) ** 2
    psf = getattr(tc, 'pixel_shuffle_factor', 4)
    n_visual_tokens = n_patches // (psf ** 2)
    print(f"  raw_patches: {n_patches}")
    print(f"  pixel_shuffle_factor: {psf}")
    print(f"  visual_tokens_per_image: {n_visual_tokens}")

    print(f"\n=== SmolVLA-specific ===")
    print(f"  VLM layers: all {tc.num_hidden_layers} (LIBERO config num_vlm_layers=0)")
    print(f"  Action expert hidden: {int(tc.hidden_size * 0.5)} (0.5x for LIBERO)")
    print(f"  VLM frozen: True (train_expert_only=True)")
    print(f"  Actions: continuous via flow matching, chunk=50")

    print(f"\nLoading model...")
    t0 = time.time()
    model = SmolVLMForConditionalGeneration.from_pretrained(
        VLM_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total: {n_params:.1f}M params")

    print("\n--- Structure ---")
    for name, child in model.named_children():
        n_p = sum(p.numel() for p in child.parameters()) / 1e6
        if n_p > 0:
            print(f"  {name}: {n_p:.1f}M params")

    processor = SmolVLMProcessor.from_pretrained(VLM_DIR, local_files_only=True, trust_remote_code=True)
    return model, processor, config


def extract_hidden_states(model, processor, config):
    print("\n" + "=" * 60)
    print("STEP 3: Dry-run hidden state extraction")
    print("=" * 60)

    hdf5_files = sorted(Path(DATA_DIR).glob("*.hdf5"))
    hdf5_path = hdf5_files[0]
    task_name = hdf5_path.stem.replace("_demo", "").replace("_", " ")
    print(f"Task: {task_name}")

    with h5py.File(hdf5_path, "r") as f:
        demo_key = sorted(f["data"].keys())[0]
        images_ds = f["data"][demo_key]["obs"]["agentview_rgb"]
        T_total = images_ds.shape[0]
        print(f"Demo: {demo_key}, total timesteps: {T_total}")

        step = max(1, T_total // 10)
        timesteps = list(range(0, T_total, step))[:10]
        print(f"Timesteps: {timesteps}")
        all_images = [Image.fromarray(images_ds[t]) for t in timesteps]

    instruction = task_name
    tc = config.text_config
    num_layers = tc.num_hidden_layers

    l0_states, llast_states = [], []
    im_l0_states, im_llast_states = [], []
    vis_start, vis_end = None, None

    for i, img in enumerate(all_images):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"What action should the robot take to {instruction}?"}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[img], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, output_hidden_states=True)

        hs = outputs.hidden_states

        if i == 0:
            seq_len = hs[0].shape[1]
            input_ids = inputs["input_ids"][0].cpu().tolist()
            print(f"\n--- Token layout (timestep 0) ---")
            print(f"Input IDs length: {len(input_ids)}")
            print(f"Hidden state seq length: {seq_len}")
            print(f"Num hidden state tensors: {len(hs)} (embedding + {len(hs)-1} layers)")
            print(f"Shape per tensor: {list(hs[0].shape)}")

            # Find image token region
            img_token_id = None
            for attr in ['image_token_id']:
                if hasattr(processor, attr):
                    img_token_id = getattr(processor, attr)
                elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, attr):
                    img_token_id = getattr(processor.tokenizer, attr)
            
            if img_token_id is None:
                # Search for special tokens
                special_ids = set(processor.tokenizer.all_special_ids) if hasattr(processor, 'tokenizer') else set()
                # Find most common token that could be image placeholder
                from collections import Counter
                counts = Counter(input_ids)
                print(f"Token frequency: {counts.most_common(5)}")
                for tid, count in counts.most_common():
                    if count >= 10:
                        img_token_id = tid
                        break
            
            print(f"Image token ID: {img_token_id}")
            
            if img_token_id is not None:
                img_positions = [j for j, tid in enumerate(input_ids) if tid == img_token_id]
                if img_positions:
                    vis_start = img_positions[0]
                    vis_end = img_positions[-1] + 1
                    n_img_tokens = len(img_positions)
                    print(f"Image tokens: positions {vis_start}..{vis_end-1} ({n_img_tokens} tokens)")
                    
                    if seq_len != len(input_ids):
                        expansion = seq_len - len(input_ids)
                        actual_n_img = n_img_tokens + expansion
                        vis_end = vis_start + actual_n_img
                        print(f"After vision encoding: {n_img_tokens}→{actual_n_img} tokens (expansion={expansion})")

                    print(f"Pre-image (system): positions 0..{vis_start-1}")
                    post_img_start = vis_end if seq_len == len(input_ids) else vis_start + actual_n_img
                    print(f"Post-image (instruction): positions {post_img_start}..{seq_len-1}")
            
            if vis_start is None:
                vis_start, vis_end = 0, min(64, seq_len)
                print(f"WARNING: using fallback image region [{vis_start}:{vis_end}]")

            # Decode prompt
            if hasattr(processor, 'tokenizer'):
                decoded = processor.tokenizer.decode(input_ids[:20])
                print(f"First 20 tokens decoded: {repr(decoded)}")

        l0 = hs[0][0, -1, :].cpu().float().numpy()
        llast = hs[-1][0, -1, :].cpu().float().numpy()
        l0_states.append(l0)
        llast_states.append(llast)

        im_l0 = hs[0][0, vis_start:vis_end, :].mean(dim=0).cpu().float().numpy()
        im_llast = hs[-1][0, vis_start:vis_end, :].mean(dim=0).cpu().float().numpy()
        im_l0_states.append(im_l0)
        im_llast_states.append(im_llast)

        del outputs, hs, inputs

    l0_arr = np.stack(l0_states)
    llast_arr = np.stack(llast_states)
    im_l0_arr = np.stack(im_l0_states)
    im_llast_arr = np.stack(im_llast_states)

    print(f"\n{'=' * 40}")
    print(f"RESULTS")
    print(f"{'=' * 40}")
    print(f"\n[last_preaction position]")
    print(f"  L0  shape: {l0_arr.shape}, hidden_dim={l0_arr.shape[1]}")
    print(f"  L{num_layers} shape: {llast_arr.shape}")
    print(f"  L0  stats: mean={l0_arr.mean():.4f} std={l0_arr.std():.4f}")
    print(f"  L{num_layers} stats: mean={llast_arr.mean():.4f} std={llast_arr.std():.4f}")

    print(f"\n[image_mean position]")
    print(f"  L0  stats: mean={im_l0_arr.mean():.4f} std={im_l0_arr.std():.4f}")
    print(f"  L{num_layers} stats: mean={im_llast_arr.mean():.4f} std={im_llast_arr.std():.4f}")

    def cossim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    print(f"\n[Temporal variance]")
    print(f"  last_preaction L0: {np.var(l0_arr, axis=0).mean():.6f}")
    print(f"  last_preaction L{num_layers}: {np.var(llast_arr, axis=0).mean():.6f}")
    print(f"  image_mean L0: {np.var(im_l0_arr, axis=0).mean():.6f}")
    print(f"  image_mean L{num_layers}: {np.var(im_llast_arr, axis=0).mean():.6f}")

    print(f"\n[Cosine sim t=0 vs t=last]")
    print(f"  last_preaction L0: {cossim(l0_arr[0], l0_arr[-1]):.4f}")
    print(f"  last_preaction L{num_layers}: {cossim(llast_arr[0], llast_arr[-1]):.4f}")
    print(f"  image_mean L0: {cossim(im_l0_arr[0], im_l0_arr[-1]):.4f}")
    print(f"  image_mean L{num_layers}: {cossim(im_llast_arr[0], im_llast_arr[-1]):.4f}")


def main():
    t0 = time.time()
    inspect_checkpoint()
    model, processor, config = load_vlm_model()
    extract_hidden_states(model, processor, config)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
