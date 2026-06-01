"""cross_instruction_extract_layers.py — Arbitrary layer range cross-instruction feature extraction.

Based on cross_instruction_extract_dense.py but supports --layers argument (e.g., "9-31").
"""

import argparse
import gc
import glob
import os
import re
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

NUM_IMAGE_TOKENS = 256


def parse_args():
    p = argparse.ArgumentParser(
        description="Arbitrary layer range cross-instruction feature extraction"
    )
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--output_dir", default="./results/cross_instruction_dense/")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--frames_per_demo", type=int, default=1)
    p.add_argument("--max_demos", type=int, default=50)
    p.add_argument("--condition", choices=["trained", "untrained", "llama2base"], default="trained")
    p.add_argument("--layers", type=str, required=True,
                   help="Layer range, e.g. '9-31' or '9-20'")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_layer_range(layers_str):
    parts = layers_str.split("-")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x) for x in layers_str.split(",")]


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
        print("Reinitializing LLM backbone (keeping vision encoder + projector)...")
        from transformers import AutoModelForCausalLM
        llm_cfg = model.language_model.config
        random_llm = AutoModelForCausalLM.from_config(llm_cfg, torch_dtype=torch.bfloat16)
        model.language_model.load_state_dict(random_llm.state_dict())
        del random_llm
        gc.collect()
        print("LLM backbone reinitialized.")

    elif condition == "llama2base":
        print("Replacing LLM backbone with Llama-2-7B base weights...")
        from transformers import AutoModelForCausalLM
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.1f}B params on {device}")
    return model, processor


def sample_frame_indices(demo_length, frames_per_demo):
    if frames_per_demo == 1:
        return [demo_length // 2]
    indices = np.linspace(0, demo_length - 1, frames_per_demo, dtype=int).tolist()
    return indices


@torch.no_grad()
def extract_features_single(model, processor, image, instruction, device, layer_indices):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    image_mean = torch.stack([hs[li][:, 1:1+NUM_IMAGE_TOKENS, :].mean(dim=1).squeeze(0) for li in layer_indices])
    last_preaction = torch.stack([hs[li][:, -1, :].squeeze(0) for li in layer_indices])

    image_mean = image_mean.cpu().half().numpy()
    last_preaction = last_preaction.cpu().half().numpy()

    del out, hs, inputs
    return image_mean, last_preaction


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dry_run = getattr(args, 'dry_run', False)
    layer_indices = parse_layer_range(args.layers)

    output_filename = f"{args.condition}_cross_inst_L{layer_indices[0]}-L{layer_indices[-1]}.h5"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"=== Cross-Instruction Feature Extraction (L{layer_indices[0]}-L{layer_indices[-1]}) ===")
    print(f"condition={args.condition}, gpu={args.gpu}, frames_per_demo={args.frames_per_demo}")
    print(f"layers={layer_indices}")
    print(f"output={output_path}")
    if dry_run:
        print("*** DRY-RUN MODE ***")

    model, processor = load_model(args.model_path, args.condition, device)

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} task files")

    all_instructions = [task_desc_from_filename(f) for f in hdf5_files]
    print(f"Instructions ({len(all_instructions)}):")
    for i, inst in enumerate(all_instructions):
        print(f"  [{i}] {inst}")

    if dry_run:
        hdf5_files = hdf5_files[:1]
        all_instructions_to_use = all_instructions[:2]
        max_demos = 1
    else:
        all_instructions_to_use = all_instructions
        max_demos = args.max_demos

    os.makedirs(args.output_dir, exist_ok=True)
    out_f = h5py.File(output_path, "w")

    out_f.attrs['layers'] = layer_indices

    meta = out_f.create_group("metadata")
    meta.create_dataset("instructions", data=[s.encode() for s in all_instructions])
    meta.create_dataset("layers", data=layer_indices)
    meta.create_dataset("task_files", data=[Path(f).name.encode() for f in hdf5_files])

    total_forwards = 0
    t_start = time.time()

    for task_idx, hdf5_path in enumerate(hdf5_files):
        task_name = Path(hdf5_path).stem.replace("_demo", "")
        print(f"\n[Task {task_idx}] {task_name}")

        f = h5py.File(hdf5_path, "r")
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        demo_keys = demo_keys[:max_demos]

        task_grp = out_f.create_group(f"task_{task_idx}")
        frame_counter = 0

        for demo_idx, dk in enumerate(tqdm(demo_keys, desc=f"task_{task_idx}")):
            images_ds = data[dk]["obs"]["agentview_rgb"]
            T = images_ds.shape[0]
            frame_indices = sample_frame_indices(T, args.frames_per_demo)

            for t in frame_indices:
                img = Image.fromarray(images_ds[t])
                frame_grp = task_grp.create_group(f"frame_{frame_counter}")
                frame_grp.attrs["demo_key"] = dk
                frame_grp.attrs["timestep"] = t
                frame_grp.attrs["demo_length"] = T

                for inst_idx, instruction in enumerate(all_instructions_to_use):
                    im_mean, lp = extract_features_single(
                        model, processor, img, instruction, device, layer_indices
                    )
                    inst_grp = frame_grp.create_group(f"instruction_{inst_idx}")
                    inst_grp.create_dataset("image_mean", data=im_mean, dtype="float16")
                    inst_grp.create_dataset("last_preaction", data=lp, dtype="float16")
                    total_forwards += 1

                frame_counter += 1

        f.close()
        out_f.flush()
        elapsed = time.time() - t_start
        rate = total_forwards / elapsed if elapsed > 0 else 0
        print(f"  {frame_counter} frames x {len(all_instructions_to_use)} instructions = {frame_counter * len(all_instructions_to_use)} forwards")
        print(f"  Total forwards so far: {total_forwards}, rate: {rate:.1f} fwd/s, elapsed: {elapsed:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    out_f.close()
    total_time = time.time() - t_start
    sz = os.path.getsize(output_path) / 1024**2
    print(f"\nDone. {total_forwards} forward passes in {total_time/60:.1f} min")
    print(f"Output: {output_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
