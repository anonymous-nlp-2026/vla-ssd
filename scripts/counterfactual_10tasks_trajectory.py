"""Trajectory-level Cohen's d from 10-task counterfactual experiment.

Re-runs inference to get per-demo MSEs (not saved in original run).
Same model, data, seed, conditions as counterfactual_instruction_10tasks.py.
"""

import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, "./data/LIBERO")
os.environ["MUJOCO_GL"] = "egl"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from PIL import Image
from scipy import stats

SEED = 42
MODEL_PATH = "./checkpoints/openvla-7b"
UNNORM_KEY = "bridge_orig"
DATA_DIR = "./data/libero/libero_goal"
RESULTS_DIR = "./results/counterfactual_10tasks"
N_DEMOS = 10
N_TIMESTEPS = 50
CONDITIONS = ["wrong", "empty", "shuffled", "paraphrased"]

TASK_INSTRUCTIONS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "correct": "open the middle drawer of the cabinet",
        "paraphrased": "pull open the center drawer on the cabinet",
        "wrong": "push the plate to the front of the stove",
    },
    "open_the_top_drawer_and_put_the_bowl_inside": {
        "correct": "open the top drawer and put the bowl inside",
        "paraphrased": "open the upper drawer and place the bowl in it",
        "wrong": "turn on the stove",
    },
    "push_the_plate_to_the_front_of_the_stove": {
        "correct": "push the plate to the front of the stove",
        "paraphrased": "slide the plate toward the front of the stove",
        "wrong": "put the wine bottle on the rack",
    },
    "put_the_bowl_on_the_plate": {
        "correct": "put the bowl on the plate",
        "paraphrased": "place the bowl onto the plate",
        "wrong": "turn on the stove",
    },
    "put_the_bowl_on_the_stove": {
        "correct": "put the bowl on the stove",
        "paraphrased": "set the bowl down on the stove",
        "wrong": "open the middle drawer of the cabinet",
    },
    "put_the_bowl_on_top_of_the_cabinet": {
        "correct": "put the bowl on top of the cabinet",
        "paraphrased": "place the bowl on the cabinet top",
        "wrong": "put the wine bottle on the rack",
    },
    "put_the_cream_cheese_in_the_bowl": {
        "correct": "put the cream cheese in the bowl",
        "paraphrased": "place the cream cheese into the bowl",
        "wrong": "put the wine bottle on top of the cabinet",
    },
    "put_the_wine_bottle_on_the_rack": {
        "correct": "put the wine bottle on the rack",
        "paraphrased": "set the wine bottle onto the rack",
        "wrong": "push the plate to the front of the stove",
    },
    "put_the_wine_bottle_on_top_of_the_cabinet": {
        "correct": "put the wine bottle on top of the cabinet",
        "paraphrased": "place the wine bottle on the cabinet top",
        "wrong": "put the cream cheese in the bowl",
    },
    "turn_on_the_stove": {
        "correct": "turn on the stove",
        "paraphrased": "switch on the stove burner",
        "wrong": "put the bowl on the plate",
    },
}


def get_shuffled_instruction(instruction, rng):
    words = instruction.split()
    rng.shuffle(words)
    return " ".join(words)


def compute_trajectory_d(demo_mses, B=10000):
    arr = np.array(demo_mses)
    n = len(arr)
    d = float(np.mean(arr) / np.std(arr, ddof=1))
    t_res = stats.ttest_1samp(arr, 0)
    rng_b = np.random.RandomState(42)
    boot_ds = []
    for _ in range(B):
        sample = rng_b.choice(arr, size=n, replace=True)
        bd = np.mean(sample) / (np.std(sample, ddof=1) + 1e-12)
        boot_ds.append(bd)
    ci_lo = float(np.percentile(boot_ds, 2.5))
    ci_hi = float(np.percentile(boot_ds, 97.5))
    return d, ci_lo, ci_hi, float(t_res.pvalue)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    device = "cuda:1"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"Loading model on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
    ).eval()
    print("Model loaded.", flush=True)

    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))[:10]
    task_names = []
    demo_mses = {c: {} for c in CONDITIONS}

    total_passes = 0
    t_start = time.time()
    total_expected = len(hdf5_files) * N_DEMOS * N_TIMESTEPS * (1 + len(CONDITIONS))

    for task_idx, hdf5_path in enumerate(hdf5_files):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        task_names.append(task_name)
        task_cfg = TASK_INSTRUCTIONS[task_name]
        correct_instruction = task_cfg["correct"]

        print(f"\nTask {task_idx}/10: {task_name}", flush=True)

        cf_instructions = {}
        for cond in CONDITIONS:
            if cond == "wrong":
                cf_instructions[cond] = task_cfg["wrong"]
            elif cond == "empty":
                cf_instructions[cond] = ""
            elif cond == "shuffled":
                cf_instructions[cond] = get_shuffled_instruction(correct_instruction, rng)
            elif cond == "paraphrased":
                cf_instructions[cond] = task_cfg["paraphrased"]

        correct_prompt = f"In: What action should the robot take to {correct_instruction}?\nOut:"
        cf_prompts = {}
        for cond in CONDITIONS:
            inst = cf_instructions[cond]
            if inst == "":
                cf_prompts[cond] = "In: What action should the robot take to ?\nOut:"
            else:
                cf_prompts[cond] = f"In: What action should the robot take to {inst}?\nOut:"

        for cond in CONDITIONS:
            demo_mses[cond][task_name] = []

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(N_DEMOS):
                demo_key = f"data/demo_{demo_idx}"
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(N_TIMESTEPS, len(images_data))

                demo_step_mses = {c: [] for c in CONDITIONS}

                for step_idx in range(n_steps):
                    img = Image.fromarray(images_data[step_idx])
                    inputs_c = processor(correct_prompt, img).to(device, dtype=torch.bfloat16)
                    c_action = model.predict_action(
                        inputs_c["input_ids"], unnorm_key=UNNORM_KEY,
                        pixel_values=inputs_c["pixel_values"]
                    )
                    total_passes += 1

                    for cond in CONDITIONS:
                        inputs_cf = processor(cf_prompts[cond], img).to(device, dtype=torch.bfloat16)
                        cf_action = model.predict_action(
                            inputs_cf["input_ids"], unnorm_key=UNNORM_KEY,
                            pixel_values=inputs_cf["pixel_values"]
                        )
                        mse = float(np.mean((c_action - cf_action) ** 2))
                        demo_step_mses[cond].append(mse)
                        total_passes += 1

                    if total_passes % 250 == 0:
                        elapsed = time.time() - t_start
                        rate = total_passes / elapsed
                        remaining = total_expected - total_passes
                        eta = remaining / rate if rate > 0 else 0
                        print(f"  {total_passes}/{total_expected} passes, {rate:.1f}/s, ETA {eta:.0f}s ({eta/60:.1f}min)", flush=True)

                for cond in CONDITIONS:
                    demo_mses[cond][task_name].append(float(np.mean(demo_step_mses[cond])))

                print(f"  demo {demo_idx}/10 done", flush=True)

        for cond in CONDITIONS:
            vals = demo_mses[cond][task_name]
            print(f"  {cond}: demo_mse={np.mean(vals):.6f} +/- {np.std(vals):.6f}", flush=True)

    # Load token-level d from original results
    token_level_d = {}
    combined_path = os.path.join(RESULTS_DIR, "counterfactual_10tasks_combined.json")
    if os.path.exists(combined_path):
        with open(combined_path) as fp:
            combined = json.load(fp)
        for cond in CONDITIONS:
            token_level_d[cond] = combined["conditions"][cond]["global_metrics"]["paired_mse"]["cohens_d"]

    results = {
        "description": "Trajectory-level Cohen's d from 10-task counterfactual analysis",
        "method": "demo-level aggregation, paired Cohen's d, bootstrap CI B=10000",
        "detail": "Each demo's MSE = mean of 50 timestep-level MSEs. "
                  "Cohen's d = mean(demo_mses) / std(demo_mses, ddof=1). "
                  "Bootstrap CI: 10000 resamples of demo-level values.",
        "aggregation_method": "mean_mse_per_demo",
        "n_tasks": len(task_names),
        "n_demos_per_task": N_DEMOS,
        "n_timesteps_per_demo": N_TIMESTEPS,
        "global": {},
        "per_task": {},
        "cross_task_variance": {},
    }

    for cond in CONDITIONS:
        all_demos = []
        for tn in task_names:
            all_demos.extend(demo_mses[cond][tn])

        d_g, ci_lo, ci_hi, p_val = compute_trajectory_d(all_demos)
        tok_d = token_level_d.get(cond, None)

        results["global"][cond] = {
            "cohens_d": round(d_g, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "p_value": p_val,
            "n_demos": len(all_demos),
            "mean_mse": round(float(np.mean(all_demos)), 6),
            "std_mse": round(float(np.std(all_demos, ddof=1)), 6),
            "token_level_d": round(tok_d, 4) if tok_d else None,
            "inflation_ratio": round(tok_d / d_g, 4) if tok_d and d_g > 0 else None,
            "demo_mses": [round(x, 6) for x in all_demos],
        }

    results["per_task"] = {}
    for tn in task_names:
        results["per_task"][tn] = {}
        for cond in CONDITIONS:
            vals = demo_mses[cond][tn]
            if len(vals) > 1:
                d_t, ci_lo, ci_hi, p_val = compute_trajectory_d(vals)
            else:
                d_t, ci_lo, ci_hi, p_val = float("nan"), float("nan"), float("nan"), float("nan")
            results["per_task"][tn][cond] = {
                "cohens_d": round(d_t, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "p_value": p_val,
                "n_demos": len(vals),
                "mean_mse": round(float(np.mean(vals)), 6),
                "demo_mses": [round(x, 6) for x in vals],
            }

    results["cross_task_variance"] = {}
    for cond in CONDITIONS:
        task_ds = []
        for tn in task_names:
            task_ds.append(results["per_task"][tn][cond]["cohens_d"])
        results["cross_task_variance"][cond] = {
            "mean": round(float(np.mean(task_ds)), 4),
            "std": round(float(np.std(task_ds, ddof=1)), 4),
            "cv": round(float(np.std(task_ds, ddof=1) / (np.mean(task_ds) + 1e-12)), 4),
            "min": round(float(np.min(task_ds)), 4),
            "max": round(float(np.max(task_ds)), 4),
            "per_task_ds": [round(x, 4) for x in task_ds],
        }

    out_path = os.path.join(RESULTS_DIR, "trajectory_level_cohens_d.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"Results saved to {out_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("TRAJECTORY-LEVEL COHEN'S D — 10 TASKS", flush=True)
    print(f"{'='*70}", flush=True)
    for cond in CONDITIONS:
        g = results["global"][cond]
        cv = results["cross_task_variance"][cond]
        tok = g["token_level_d"] or 0
        infl = g["inflation_ratio"] or 0
        print(f"  {cond:12s}: traj_d={g['cohens_d']:.3f} [{g['ci_95'][0]:.3f}, {g['ci_95'][1]:.3f}]  "
              f"token_d={tok:.3f}  ratio={infl:.3f}  "
              f"cross-task: {cv['mean']:.3f}+/-{cv['std']:.3f} (CV={cv['cv']:.3f})", flush=True)


if __name__ == "__main__":
    main()
