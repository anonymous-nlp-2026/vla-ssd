"""CLIP ViT-L/14 feature extraction on LIBERO-Goal demos.

Output format matches DINO extraction: traj_{i}/{cls_token, patch_mean}.
"""
import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-large-patch14"

TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract CLIP ViT-L/14 features for LIBERO-Goal demos."
    )
    p.add_argument(
        "--model_path", default=MODEL_ID,
        help="HuggingFace model ID or local path to CLIP ViT-L/14 weights.",
    )
    p.add_argument(
        "--data_dir",
        default="./data/libero/libero_goal/",
    )
    p.add_argument(
        "--output_dir",
        default="./features/clip_libero_goal/",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_demos", type=int, default=50)
    p.add_argument("--image_key", default="agentview_rgb")
    return p.parse_args()


def extract_one_task(model, processor, hdf5_path, args, device):
    task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "").replace(".hdf5", "")
    output_path = os.path.join(args.output_dir, f"{task_name}.h5")

    if os.path.exists(output_path):
        print(f"  SKIP (exists): {output_path}")
        return 0, 0.0

    with h5py.File(hdf5_path, "r") as src:
        demo_keys = sorted(
            [k for k in src["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )[: args.max_demos]
        print(f"  {len(demo_keys)} demos")

        total_steps = 0
        t0 = time.time()

        with h5py.File(output_path, "w") as dst:
            for di, dk in enumerate(demo_keys):
                images = src[f"data/{dk}/obs/{args.image_key}"][:]  # (T, H, W, 3)
                T = images.shape[0]
                total_steps += T

                cls_tokens = []
                patch_means = []

                for start in range(0, T, args.batch_size):
                    end = min(start + args.batch_size, T)
                    batch_imgs = [img for img in images[start:end]]

                    inputs = processor(
                        images=batch_imgs, return_tensors="pt"
                    )
                    pixel_values = inputs["pixel_values"].to(device)

                    with torch.no_grad():
                        outputs = model.vision_model(
                            pixel_values, output_hidden_states=False
                        )
                    hidden = outputs.last_hidden_state  # (B, 257, 1024)

                    cls_tokens.append(
                        hidden[:, 0, :].cpu().to(torch.float16).numpy()
                    )
                    patch_means.append(
                        hidden[:, 1:, :].mean(dim=1).cpu().to(torch.float16).numpy()
                    )

                cls_all = np.concatenate(cls_tokens, axis=0)      # (T, 1024)
                patch_all = np.concatenate(patch_means, axis=0)    # (T, 1024)

                grp = dst.create_group(f"traj_{di}")
                grp.create_dataset(
                    "cls_token", data=cls_all,
                    compression="gzip", compression_opts=4,
                )
                grp.create_dataset(
                    "patch_mean", data=patch_all,
                    compression="gzip", compression_opts=4,
                )

                if (di + 1) % 10 == 0 or di == len(demo_keys) - 1:
                    print(f"    demo {di+1}/{len(demo_keys)} done")

        elapsed = time.time() - t0
        fps = total_steps / elapsed if elapsed > 0 else 0
        print(f"  {total_steps} frames, {elapsed:.1f}s, {fps:.0f} fps -> {output_path}")
        return total_steps, elapsed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"Loading CLIP ViT-L/14 from {args.model_path} ...")
    model = CLIPModel.from_pretrained(
        args.model_path, local_files_only=(args.model_path != MODEL_ID),
    )
    model.vision_model = model.vision_model.to(device).eval()
    del model.text_model, model.text_projection, model.visual_projection
    processor = CLIPProcessor.from_pretrained(
        args.model_path, local_files_only=(args.model_path != MODEL_ID),
    )
    print(f"Vision model on {device}")

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    hdf5_files = [f for f in hdf5_files
                  if os.path.basename(f).replace("_demo.hdf5", "") in TASKS]
    print(f"Found {len(hdf5_files)} task files\n")

    grand_steps = 0
    grand_time = 0.0

    for fi, fpath in enumerate(hdf5_files):
        print(f"[{fi+1}/{len(hdf5_files)}] {os.path.basename(fpath)}")
        steps, elapsed = extract_one_task(model, processor, fpath, args, device)
        grand_steps += steps
        grand_time += elapsed

    if grand_time > 0:
        print(f"\n=== Done ===")
        print(f"Total: {grand_steps} frames, {grand_time:.1f}s, {grand_steps/grand_time:.0f} fps")
    else:
        print("\nAll tasks skipped (already extracted).")


if __name__ == "__main__":
    main()
