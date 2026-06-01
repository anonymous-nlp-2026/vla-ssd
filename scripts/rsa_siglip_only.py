"""RSA for SigLIP+projector features (no LLM) vs action dissimilarity."""

import itertools
import json
import os
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_DIR = Path("./features/siglip_only_libero_goal")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/rsa/rsa_siglip_only.json")

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


def load_features(task):
    path = FEAT_DIR / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("demo_"):
                did = int(key.split("_")[1])
                result[did] = f[key]["last_preaction"][:].astype(np.float32)
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


def main():
    np.random.seed(42)
    per_task = {}
    rsa_values = []

    for task in TASKS:
        feats = load_features(task)
        actions = load_actions(task)
        rsa = compute_rsa(feats, actions)
        per_task[task] = rsa
        if rsa is not None:
            rsa_values.append(rsa)
        tag = f"{rsa:.4f}" if rsa is not None else "SKIP"
        print(f"  {task}: {tag}")

    mean_rsa, ci_lo, ci_hi = bootstrap_ci(rsa_values)

    results = {
        "siglip_only_peak_rsa": round(mean_rsa, 4),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "per_task": {k: round(v, 4) if v is not None else None for k, v in per_task.items()},
        "comparison": {
            "trained_with_inst": 0.4876,
            "untrained_with_inst": 0.4846,
            "untrained_no_inst": 0.4802,
            "dino": 0.388,
            "siglip_only": round(mean_rsa, 4),
        },
    }

    if mean_rsa >= 0.45:
        results["gate_judgment"] = (
            f"PASS — SigLIP-only RSA = {mean_rsa:.4f} ≈ trained/untrained "
            f"({0.4876:.4f}/{0.4846:.4f}). Vision encoder+projector alone "
            f"accounts for action-relevant structure; LLM is near-passthrough."
        )
    else:
        results["gate_judgment"] = (
            f"FAIL — SigLIP-only RSA = {mean_rsa:.4f} < 0.45 threshold. "
            f"LLM contributes to action-relevant representations beyond "
            f"vision encoder+projector."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSigLIP-only RSA: {mean_rsa:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\nComparison:")
    for k, v in results["comparison"].items():
        print(f"  {k:25s}: {v}")
    print(f"\n{results['gate_judgment']}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
