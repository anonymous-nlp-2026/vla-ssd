"""extract_features_rand_inst.py — Feature extraction with controlled instruction modes.

For plan_008 (Random Instruction Control): extract VLA hidden states under different
instruction conditions to test whether instruction content affects representation structure.

Modes:
  random:     Random nonsense string replaces task description (run per-seed, 3 seeds recommended)
  shuffled:   Word-level shuffle of each task's real instruction
  wrong_task: Cyclic shift — each task gets the next task's instruction

Input:  LIBERO-Goal HDF5 demos  +  trained OpenVLA-7B checkpoint
Output: Per-task HDF5 files in ./features/trained_libero_goal_{mode}[_s{seed}]/
        Each file: demo_X/{last_preaction, image_mean} with shape (T, num_layers, 4096) in fp16

Usage:
  # Dry run (1 demo, 1 task — verify pipeline):
  python scripts/extract_features_rand_inst.py --instruction_mode random --seed 0 --dry_run

  # Full extraction (each takes ~48 min on RTX PRO 6000):
  python scripts/extract_features_rand_inst.py --instruction_mode random --seed 0 --gpu 0
  python scripts/extract_features_rand_inst.py --instruction_mode random --seed 1 --gpu 0
  python scripts/extract_features_rand_inst.py --instruction_mode random --seed 2 --gpu 0
  python scripts/extract_features_rand_inst.py --instruction_mode shuffled --gpu 0
  python scripts/extract_features_rand_inst.py --instruction_mode wrong_task --gpu 0

Dependencies: transformers, torch, h5py, tqdm, Pillow, numpy
"""

import argparse
import gc
import glob
import os
import random
import sys
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import load_model, extract_one_task, parse_layers

FEAT_ROOT = Path("./features")
DATA_DIR = Path("./data/libero/libero_goal")
MODEL_PATH = "./checkpoints/openvla-7b"

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

TASK_DESCS = {t: t.replace("_", " ") for t in TASKS}


def generate_random_string(rng):
    """Pronounceable nonsense: 3-6 words, consonant-vowel alternation."""
    C = list("bcdfghjklmnpqrstvwxyz")
    V = list("aeiou")
    words = []
    for _ in range(rng.randint(3, 6)):
        w = ""
        for j in range(rng.randint(2, 5)):
            w += rng.choice(C) if j % 2 == 0 else rng.choice(V)
        words.append(w)
    return " ".join(words)


def get_instructions(mode, seed=42):
    """Generate per-task instructions for the given mode.

    Returns:
        dict: {task_name: instruction_string}
        str:  human-readable description of the instruction set
    """
    rng = random.Random(seed)

    if mode == "random":
        s = generate_random_string(rng)
        return {t: s for t in TASKS}, f"random string: \"{s}\""

    elif mode == "shuffled":
        out = {}
        for t in TASKS:
            ws = TASK_DESCS[t].split()
            shuffled = ws.copy()
            for _ in range(20):
                rng.shuffle(shuffled)
                if shuffled != ws:
                    break
            out[t] = " ".join(shuffled)
        return out, "per-task word shuffle"

    elif mode == "wrong_task":
        out = {TASKS[i]: TASK_DESCS[TASKS[(i + 1) % len(TASKS)]]
               for i in range(len(TASKS))}
        return out, "cyclic shift (task i gets task i+1 instruction)"

    raise ValueError(f"Unknown mode: {mode}")


def get_output_dir(mode, seed=None):
    if mode == "random":
        return FEAT_ROOT / f"trained_libero_goal_random_s{seed}"
    return FEAT_ROOT / f"trained_libero_goal_{mode}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract OpenVLA features with controlled instruction modes (plan_008)")
    p.add_argument("--instruction_mode", required=True,
                   choices=["random", "shuffled", "wrong_task"],
                   help="Instruction control mode")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (affects random string generation and shuffle order)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_demos", type=int, default=50)
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Limit number of tasks (for testing)")
    p.add_argument("--layers", default=None,
                   help="Layers to extract: '0-32' or '0,5,10'. Default: all 33")
    p.add_argument("--dry_run", action="store_true",
                   help="Process 1 demo of 1 task to verify the pipeline works")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    layer_indices = parse_layers(args.layers)

    instructions, desc = get_instructions(args.instruction_mode, args.seed)
    output_dir = get_output_dir(args.instruction_mode, args.seed)

    print(f"=== plan_008: Instruction Control Feature Extraction ===")
    print(f"Mode: {args.instruction_mode}  Seed: {args.seed}")
    print(f"Strategy: {desc}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layer_indices[0]}-{layer_indices[-1]} ({len(layer_indices)} layers)")

    print(f"\nInstruction mapping:")
    for t, inst in instructions.items():
        real = TASK_DESCS[t]
        print(f"  {t[:45]:45s}")
        print(f"    real:    \"{real}\"")
        print(f"    control: \"{inst}\"")

    max_demos = 1 if args.dry_run else args.max_demos
    max_tasks = 1 if args.dry_run else args.max_tasks
    if args.dry_run:
        print(f"\n*** DRY RUN: 1 task, 1 demo ***")

    print(f"\nLoading model from {MODEL_PATH} ...")
    model, processor = load_model(MODEL_PATH, "trained", device)

    hdf5_files = sorted(glob.glob(str(DATA_DIR / "*_demo.hdf5")))
    tasks_to_process = [(p, Path(p).stem.replace("_demo", ""))
                        for p in hdf5_files
                        if Path(p).stem.replace("_demo", "") in instructions]
    if max_tasks:
        tasks_to_process = tasks_to_process[:max_tasks]

    print(f"\nProcessing {len(tasks_to_process)} task(s), "
          f"{max_demos} demo(s) each, batch_size={args.batch_size}")

    t_all = time.time()
    for i, (hdf5_path, task_id) in enumerate(tasks_to_process):
        out_path = str(output_dir / f"{task_id}.h5")

        if os.path.exists(out_path) and not args.dry_run:
            print(f"[{i+1}/{len(tasks_to_process)}] Skip {task_id} (exists)")
            continue

        task_desc = instructions[task_id]
        prompt_str = f"In: What action should the robot take to {task_desc}?\nOut:"
        print(f"\n[{i+1}/{len(tasks_to_process)}] {task_id}")
        print(f"  Prompt: \"{prompt_str[:80]}\"")

        t0 = time.time()
        extract_one_task(
            model, processor, hdf5_path, task_desc, device,
            args.batch_size, max_demos, layer_indices, out_path,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    total_min = (time.time() - t_all) / 60
    print(f"\n{'='*60}")
    print(f"Total: {total_min:.1f} min")
    print(f"Features saved to: {output_dir}")

    if args.dry_run:
        out_files = list(output_dir.glob("*.h5"))
        if out_files:
            import h5py
            with h5py.File(str(out_files[0]), "r") as f:
                print(f"\nDry-run output verification:")
                print(f"  File: {out_files[0].name}")
                print(f"  Keys: {list(f.keys())}")
                for k in list(f.keys())[:1]:
                    print(f"  {k}/last_preaction shape: {f[k]['last_preaction'].shape}")
                    print(f"  {k}/image_mean shape: {f[k]['image_mean'].shape}")
            print("  Pipeline OK.")


if __name__ == "__main__":
    main()
