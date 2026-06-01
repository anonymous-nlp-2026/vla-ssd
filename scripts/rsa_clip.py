"""RSA for CLIP ViT-L/14 features vs action dissimilarity.

Matches rsa_siglip_only.py methodology but reads CLIP features
(traj_X/cls_token format, same as DINO).
"""

import itertools
import json
import os
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_DIR = Path("./features/clip_libero_goal")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/rsa/rsa_clip.json")

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


def load_features(task, feature_key="cls_token"):
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
        return None
    pairs = list(itertools.combinations(common, 2))
    rep_dists = [mean_cosine_distance(feats[a], feats[b]) for a, b in pairs]
    act_dists = [mean_action_distance(actions[a], actions[b]) for a, b in pairs]
    rsa, _ = spearmanr(rep_dists, act_dists)
    return float(rsa)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def run_rsa(feature_key="cls_token"):
    per_task = {}
    rsa_values = []

    for task in TASKS:
        feats = load_features(task, feature_key)
        actions = load_actions(task)
        rsa = compute_rsa(feats, actions)
        per_task[task] = rsa
        if rsa is not None:
            rsa_values.append(rsa)
        tag = f"{rsa:.4f}" if rsa is not None else "SKIP"
        print(f"  {task}: {tag}")

    if not rsa_values:
        return None, None, None, per_task

    mean_rsa, ci_lo, ci_hi = bootstrap_ci(rsa_values)
    return mean_rsa, ci_lo, ci_hi, per_task


def main():
    np.random.seed(42)

    results = {}
    for fkey in ["cls_token", "patch_mean"]:
        print(f"\n--- CLIP ViT-L/14  feature_key={fkey} ---")
        mean_rsa, ci_lo, ci_hi, per_task = run_rsa(fkey)
        if mean_rsa is None:
            print("  No valid RSA values.")
            continue

        results[fkey] = {
            "rsa": round(mean_rsa, 4),
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "per_task": {k: round(v, 4) if v is not None else None
                        for k, v in per_task.items()},
        }
        print(f"\n  CLIP ({fkey}) RSA: {mean_rsa:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    best_key = max(results, key=lambda k: results[k]["rsa"]) if results else None
    best_rsa = results[best_key]["rsa"] if best_key else None

    output = {
        "model": "CLIP ViT-L/14 (openai)",
        "per_feature_key": results,
        "best_key": best_key,
        "best_rsa": best_rsa,
        "comparison": {
            "siglip_projector": 0.615,
            "siglip_only": 0.4876,
            "trained_vla": 0.4876,
            "untrained_vla": 0.4846,
            "dino": 0.388,
            "clip": best_rsa,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nComparison:")
    for k, v in output["comparison"].items():
        print(f"  {k:20s}: {v}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
