"""extract_features.py — Extract OpenVLA hidden states for temporal distance probing.

Input:  LIBERO HDF5 demos + OpenVLA checkpoint (or untrained variant)
Output: Per-task HDF5 files with per-demo hidden states (T, num_layers, 4096) in fp16
Dependencies: transformers, torch, h5py, tqdm, Pillow

Token layout (Prismatic/OpenVLA): [BOS] [256 image tokens] [text tokens...]
  - last_preaction: hidden state at the last sequence position (pre-action)
  - image_mean: mean of 256 image token hidden states (positions 1:257)
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
NUM_HIDDEN_LAYERS = 33  # embedding + 32 transformer layers


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract OpenVLA hidden states for temporal distance probing"
    )
    p.add_argument("--model_path", required=True,
                   help="Path to OpenVLA checkpoint directory")
    p.add_argument("--data_dir", default="./data/libero/libero_10/",
                   help="Directory containing LIBERO HDF5 files")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for extracted features")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--mode", choices=["trained", "untrained"], default="trained",
                   help="'untrained' reinitializes LLM backbone randomly")
    p.add_argument("--task_filter", default=None,
                   help="Substring filter on HDF5 filename")
    p.add_argument("--task_instruction", default=None,
                   help="Override task instruction for all tasks (default: derived from filename)")
    p.add_argument("--max_demos", type=int, default=50,
                   help="Max demos per task (default: 50)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for forward passes (default: 4)")
    p.add_argument("--layers", default=None,
                   help="Layers to extract: '0-32' or '0,5,10,15,20,25,32'. Default: all 33")
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Max tasks to process (for testing)")
    return p.parse_args()


def parse_layers(spec, max_layer=32):
    if spec is None:
        return list(range(max_layer + 1))
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    m = re.match(r"^[A-Z_]+SCENE\d+_(.+)$", stem)
    if m:
        return m.group(1).replace("_", " ")
    return stem.replace("_", " ")


def load_model(model_path, mode, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    if mode == "untrained":
        print("Reinitializing LLM backbone (keeping vision encoder + projector)...")
        from transformers import AutoModelForCausalLM

        llm_cfg = model.language_model.config
        random_llm = AutoModelForCausalLM.from_config(llm_cfg, torch_dtype=torch.bfloat16)
        model.language_model.load_state_dict(random_llm.state_dict())
        del random_llm
        gc.collect()
        print("LLM backbone reinitialized.")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.1f}B params on {device}")
    return model, processor


@torch.no_grad()
def extract_one_task(model, processor, hdf5_path, task_desc, device,
                     batch_size, max_demos, layer_indices, out_path):
    prompt = f"In: What action should the robot take to {task_desc}?\nOut:"

    f = h5py.File(hdf5_path, "r")
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    if max_demos:
        demo_keys = demo_keys[:max_demos]

    total_steps = sum(data[dk]["actions"].shape[0] for dk in demo_keys)
    pbar = tqdm(total=total_steps, desc=Path(hdf5_path).stem[:50], unit="step")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_f = h5py.File(out_path, "w")
    out_f.attrs["layers"] = np.array(layer_indices, dtype=np.int32)
    out_f.attrs["num_image_tokens"] = NUM_IMAGE_TOKENS

    for demo_key in demo_keys:
        images_ds = data[demo_key]["obs"]["agentview_rgb"]
        T = images_ds.shape[0]

        lp_list = []
        im_list = []

        for t0 in range(0, T, batch_size):
            t1 = min(t0 + batch_size, T)
            batch_imgs = [Image.fromarray(images_ds[t]) for t in range(t0, t1)]
            B = len(batch_imgs)

            inputs = processor(
                text=[prompt] * B, images=batch_imgs, padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs, output_hidden_states=True)
            hs_tuple = out.hidden_states

            if hs_tuple is None:
                raise RuntimeError(
                    "hidden_states is None — model may not support output_hidden_states"
                )

            # Batch GPU→CPU: stack on GPU first, single .cpu() transfer
            lp_gpu = torch.stack([hs_tuple[li][:, -1, :] for li in layer_indices])  # (L, B, D)
            im_gpu = torch.stack([hs_tuple[li][:, 1:1+NUM_IMAGE_TOKENS, :].mean(dim=1) for li in layer_indices])  # (L, B, D)
            lp_batch = lp_gpu.permute(1, 0, 2).cpu().half()  # (B, L, D)
            im_batch = im_gpu.permute(1, 0, 2).cpu().half()  # (B, L, D)
            del lp_gpu, im_gpu

            for b in range(B):
                lp_list.append(lp_batch[b])
                im_list.append(im_batch[b])

            del out, hs_tuple, inputs, lp_batch, im_batch
            pbar.update(t1 - t0)

        # Write this demo immediately, then free memory
        lp_arr = torch.stack(lp_list).numpy()  # (T, num_layers, 4096)
        im_arr = torch.stack(im_list).numpy()
        g = out_f.create_group(demo_key)
        g.create_dataset("last_preaction", data=lp_arr, dtype="float16",
                         compression="gzip", compression_opts=4)
        g.create_dataset("image_mean", data=im_arr, dtype="float16",
                         compression="gzip", compression_opts=4)
        out_f.flush()
        del lp_list, im_list, lp_arr, im_arr

    pbar.close()
    f.close()
    out_f.close()

    sz = os.path.getsize(out_path) / 1024**3
    print(f"Saved {out_path} ({sz:.2f} GB)")


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    layer_indices = parse_layers(args.layers)

    print(f"=== Feature Extraction: mode={args.mode}, gpu={args.gpu}, bs={args.batch_size} ===")
    print(f"Layers: {layer_indices[0]}-{layer_indices[-1]} ({len(layer_indices)} layers)")
    model, processor = load_model(args.model_path, args.mode, device)

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    if args.task_filter:
        hdf5_files = [f for f in hdf5_files if args.task_filter in f]
    if args.max_tasks:
        hdf5_files = hdf5_files[:args.max_tasks]
    print(f"Found {len(hdf5_files)} task file(s)")

    t_all = time.time()
    for i, hdf5_path in enumerate(hdf5_files):
        if args.task_instruction is not None:
            task_desc = args.task_instruction
        else:
            task_desc = task_desc_from_filename(hdf5_path)
        task_id = Path(hdf5_path).stem.replace("_demo", "")
        out_path = os.path.join(args.output_dir, f"{task_id}.h5")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(hdf5_files)}] Skip {task_id} (exists)")
            continue

        print(f"\n[{i+1}/{len(hdf5_files)}] {task_desc}")
        t0 = time.time()
        extract_one_task(
            model, processor, hdf5_path, task_desc, device,
            args.batch_size, args.max_demos, layer_indices, out_path
        )
        print(f"  Took {time.time()-t0:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTotal: {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
