"""Standalone DINOv2 RSA — Method B (within-task, cosine+euclidean, N=20).
Establishes provenance for DINO cls_token and patch_mean RSA values.
"""

import itertools
import json
import os
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_DIR = Path("./features/dinov2_libero_goal")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/rsa/rsa_dino_standalone.json")

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

N_SAMPLE = 20


def subsample_to_length(arr, target_len):
    T = arr.shape[0]
    if T <= target_len:
        return arr
    idx = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[idx]


def mean_cosine_distance(feat_a, feat_b):
    a = subsample_to_length(feat_a, N_SAMPLE)
    b = subsample_to_length(feat_b, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([cosine_dist(a[i], b[i]) for i in range(n)]))


def mean_action_distance(act_a, act_b):
    a = subsample_to_length(act_a, N_SAMPLE)
    b = subsample_to_length(act_b, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in range(n)]))


def load_dino_features(task, feature_key):
    path = FEAT_DIR / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("traj_"):
                did = int(key.split("_")[1])
                result[did] = f[key][feature_key][:].astype(np.float32)
    return result


def load_actions(task):
    path = DATA_DIR / f"{task}_demo.hdf5"
    result = {}
    with h5py.File(path, "r") as f:
        for key in f["data"].keys():
            if key.startswith("demo_"):
                did = int(key.split("_")[1])
                result[did] = f[f"data/{key}/actions"][:].astype(np.float32)
    return result


def compute_rsa(feats, actions):
    common = sorted(set(feats.keys()) & set(actions.keys()))
    if len(common) < 2:
        return None, len(common)
    pairs = list(itertools.combinations(common, 2))
    rep_dists = [mean_cosine_distance(feats[a], feats[b]) for a, b in pairs]
    act_dists = [mean_action_distance(actions[a], actions[b]) for a, b in pairs]
    rsa, _ = spearmanr(rep_dists, act_dists)
    return float(rsa), len(common)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [
        np.mean(np.random.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def run_for_feature(feature_key):
    per_task = {}
    rsa_values = []
    n_demos_per_task = []

    for task in TASKS:
        feats = load_dino_features(task, feature_key)
        actions = load_actions(task)
        rsa, n_demos = compute_rsa(feats, actions)
        per_task[task] = round(rsa, 4) if rsa is not None else None
        n_demos_per_task.append(n_demos)
        if rsa is not None:
            rsa_values.append(rsa)
        tag = f"{rsa:.4f}" if rsa is not None else "SKIP"
        print(f"  [{feature_key}] {task}: {tag} (n={n_demos})")

    mean_rsa, ci_lo, ci_hi = bootstrap_ci(rsa_values)
    return {
        "mean_rsa": round(mean_rsa, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "per_task": per_task,
    }, n_demos_per_task


def main():
    t0 = time.time()
    np.random.seed(42)

    print("=== DINOv2 Standalone RSA (Method B) ===\n")

    cls_result, n_demos = run_for_feature("cls_token")
    print()
    patch_result, _ = run_for_feature("patch_mean")

    elapsed = time.time() - t0

    results = {
        "method": "B (within-task + cosine rep dist + euclidean action dist)",
        "rep_distance": "cosine",
        "action_distance": "euclidean",
        "n_sample": N_SAMPLE,
        "seed": 42,
        "n_tasks": len(TASKS),
        "n_demos_per_task": n_demos,
        "cls_token": cls_result,
        "patch_mean": patch_result,
        "script": "./scripts/rsa_dino_standalone.py",
        "runtime_seconds": round(elapsed, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Results ---")
    print(f"cls_token  RSA: {cls_result['mean_rsa']:.4f} {cls_result['ci_95']}")
    print(f"patch_mean RSA: {patch_result['mean_rsa']:.4f} {patch_result['ci_95']}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
