"""RSA analysis for LIBERO-Spatial: 33 layers x 2 aggregations x 2 conditions (trained/untrained).

Method B (matches rsa_unified_vla.py):
  - rep_dist: cosine
  - act_dist: euclidean (L2)
  - N_SAMPLE: 20 (np.linspace subsample)
  - pairing: within-task itertools.combinations
  - aggregation: per-task Spearman -> mean + bootstrap CI (1000 resamples, seed=42)

Input:
  - Features: ./features/{trained,untrained}_libero_spatial/*.h5
    Format: demo_{i}/{image_mean,last_preaction} -> (T, 33, 4096)
  - Actions: ./data/libero/libero_spatial/{task}_demo.hdf5
    Format: data/demo_{i}/actions

Output:
  - ./results/rsa/rsa_libero_spatial.json

Dependencies: numpy, scipy, h5py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

FEAT_BASE = Path("./features")
DATA_DIR = Path("./data/libero/libero_spatial")
OUT_PATH = Path("./results/rsa/rsa_libero_spatial.json")

FEAT_DIRS = {
    "trained": FEAT_BASE / "trained_libero_spatial",
    "untrained": FEAT_BASE / "untrained_libero_spatial",
}

AGG_KEYS = ["image_mean", "last_preaction"]
N_LAYERS = 33
N_SAMPLE = 20


def discover_tasks(feat_dir):
    """Auto-discover task names from .h5 feature files."""
    return sorted([p.stem for p in feat_dir.glob("*.h5")])


def subsample_to_length(arr, target_len):
    T = arr.shape[0]
    if T <= target_len:
        return arr
    idx = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[idx]


def load_actions(task):
    path = DATA_DIR / f"{task}_demo.hdf5"
    if not path.exists():
        return {}
    result = {}
    with h5py.File(path, "r") as f:
        for key in f["data"].keys():
            if key.startswith("demo_"):
                did = int(key.split("_")[1])
                result[did] = f[f"data/{key}/actions"][:].astype(np.float32)
    return result


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [
        np.mean(np.random.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def compute_rsa_all_layers(feat_dir, task, aggs, actions_cache):
    """Compute RSA for all layers of a single task."""
    if task not in actions_cache:
        actions_cache[task] = load_actions(task)
    actions = actions_cache[task]

    results = {}

    feat_path = feat_dir / f"{task}.h5"
    if not feat_path.exists():
        for agg in aggs:
            results[agg] = {layer: None for layer in range(N_LAYERS)}
        return results

    with h5py.File(feat_path, "r") as f:
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        common_dids = sorted(
            set(int(k.split("_")[1]) for k in demo_keys) & set(actions.keys())
        )

        if len(common_dids) < 2:
            for agg in aggs:
                results[agg] = {layer: None for layer in range(N_LAYERS)}
            return results

        n = len(common_dids)
        n_pairs = n * (n - 1) // 2

        sub_acts = np.stack(
            [subsample_to_length(actions[d], N_SAMPLE) for d in common_dids]
        )
        act_dists = np.zeros(n_pairs)
        for t in range(sub_acts.shape[1]):
            act_dists += pdist(sub_acts[:, t, :], metric="euclidean")
        act_dists /= sub_acts.shape[1]

        for agg in aggs:
            sub_feats = []
            for did in common_dids:
                raw = f[f"demo_{did}/{agg}"][:]
                sub = subsample_to_length(raw, N_SAMPLE).astype(np.float32)
                sub_feats.append(sub)

            feat_stack = np.stack(sub_feats)
            del sub_feats

            results[agg] = {}
            for layer in range(N_LAYERS):
                feat_layer = feat_stack[:, :, layer, :]

                rep_dists = np.zeros(n_pairs)
                for t in range(feat_layer.shape[1]):
                    rep_dists += pdist(feat_layer[:, t, :], metric="cosine")
                rep_dists /= feat_layer.shape[1]

                valid = ~(np.isnan(rep_dists) | np.isnan(act_dists))
                if valid.sum() < 3:
                    results[agg][layer] = None
                else:
                    rsa, _ = spearmanr(rep_dists[valid], act_dists[valid])
                    results[agg][layer] = float(rsa)

            del feat_stack

    return results


def main():
    np.random.seed(42)
    t0 = time.time()

    for cond, feat_dir in FEAT_DIRS.items():
        if not feat_dir.exists():
            print(f"ERROR: Feature directory not found: {feat_dir}")
            print("Run feature extraction first.")
            sys.exit(1)

    if not DATA_DIR.exists():
        print(f"ERROR: Action data directory not found: {DATA_DIR}")
        sys.exit(1)

    actions_cache = {}
    all_results = {}

    for cond, feat_dir in FEAT_DIRS.items():
        tasks = discover_tasks(feat_dir)
        if not tasks:
            print(f"SKIP {cond}: no .h5 files in {feat_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Condition: {cond} ({len(tasks)} tasks)")
        print(f"{'='*60}")

        cond_results = {}

        for agg in AGG_KEYS:
            print(f"\n--- {cond} / {agg} ---")
            layer_data = {layer: {"per_task": {}, "rsa_values": []} for layer in range(N_LAYERS)}

            for task in tasks:
                task_t = time.time()
                rsa_by_layer = compute_rsa_all_layers(feat_dir, task, [agg], actions_cache)

                for layer in range(N_LAYERS):
                    val = rsa_by_layer[agg][layer]
                    layer_data[layer]["per_task"][task] = (
                        round(val, 4) if val is not None else None
                    )
                    if val is not None:
                        layer_data[layer]["rsa_values"].append(val)

                print(f"  {task}: {time.time()-task_t:.1f}s")

            layer_results = {}
            for layer in range(N_LAYERS):
                vals = layer_data[layer]["rsa_values"]
                if vals:
                    mean_rsa, ci_lo, ci_hi = bootstrap_ci(vals)
                    layer_results[f"layer_{layer}"] = {
                        "mean_rsa": round(mean_rsa, 4),
                        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
                        "per_task": layer_data[layer]["per_task"],
                    }
                else:
                    layer_results[f"layer_{layer}"] = {
                        "mean_rsa": None,
                        "per_task": layer_data[layer]["per_task"],
                    }

            cond_results[agg] = layer_results

            valid_layers = {
                k: v["mean_rsa"]
                for k, v in layer_results.items()
                if v["mean_rsa"] is not None
                and not (isinstance(v["mean_rsa"], float) and v["mean_rsa"] != v["mean_rsa"])
            }
            if valid_layers:
                best_k = max(valid_layers, key=valid_layers.get)
                best_l = int(best_k.split("_")[1])
                best_r = valid_layers[best_k]
                print(f"  BEST: layer {best_l}, RSA = {best_r:.4f}")

        all_results[cond] = cond_results

    summary = {}
    for cond, cond_res in all_results.items():
        for agg, layer_res in cond_res.items():
            valid = {
                k: v["mean_rsa"]
                for k, v in layer_res.items()
                if v["mean_rsa"] is not None
                and not (isinstance(v["mean_rsa"], float) and v["mean_rsa"] != v["mean_rsa"])
            }
            if valid:
                best_k = max(valid, key=valid.get)
                best_l = int(best_k.split("_")[1])
                best_r = valid[best_k]
                summary[f"{cond}_{agg}_best"] = {"layer": best_l, "rsa": best_r}

    first_cond = list(all_results.keys())[0] if all_results else None
    task_list = discover_tasks(FEAT_DIRS[first_cond]) if first_cond else []

    output = {
        "method": "B (within-task + cosine rep dist + euclidean action dist + full trajectory)",
        "rep_distance": "cosine",
        "action_distance": "euclidean (L2)",
        "n_sample": N_SAMPLE,
        "n_layers": N_LAYERS,
        "n_tasks": len(task_list),
        "tasks": task_list,
        "seed": 42,
        "conditions": all_results,
        "summary": summary,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved to {OUT_PATH}")
    print(f"Total time: {time.time()-t0:.1f}s\n")

    print("Summary (best layer per condition):")
    for k, v in summary.items():
        print(f"  {k}: layer {v['layer']}, RSA = {v['rsa']:.4f}")


if __name__ == "__main__":
    main()
