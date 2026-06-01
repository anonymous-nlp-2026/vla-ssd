"""RSA per-DoF decomposition: break down action distance by dimension groups.

Method B framework:
  - rep_dist: cosine
  - act_dist: per-DoF euclidean (xyz, rpy, gripper) + full 7D euclidean
  - N_SAMPLE: 20 (np.linspace subsample)
  - pairing: within-task itertools.combinations
  - aggregation: per-task Spearman -> mean across tasks
  - 50 demos/task, full trajectory
"""

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

FEAT_BASE = Path("./features")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/ablations/rsa_per_dof.json")

FEAT_DIRS = {
    "trained": FEAT_BASE / "trained_libero_goal",
    "untrained": FEAT_BASE / "untrained_libero_goal",
}

AGG_KEYS = ["image_mean", "last_preaction"]
N_LAYERS = 33
N_SAMPLE = 20

DOF_GROUPS = {
    "xyz": [0, 1, 2],
    "rpy": [3, 4, 5],
    "gripper": [6],
    "full": [0, 1, 2, 3, 4, 5, 6],
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def discover_tasks(feat_dir):
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


def compute_rsa_per_dof(feat_dir, task, aggs, actions_cache):
    if task not in actions_cache:
        actions_cache[task] = load_actions(task)
    actions = actions_cache[task]

    results = {}

    feat_path = feat_dir / f"{task}.h5"
    if not feat_path.exists():
        for agg in aggs:
            results[agg] = {dof: [None]*N_LAYERS for dof in DOF_GROUPS}
        return results

    with h5py.File(feat_path, "r") as f:
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        common_dids = sorted(
            set(int(k.split("_")[1]) for k in demo_keys) & set(actions.keys())
        )

        if len(common_dids) < 2:
            for agg in aggs:
                results[agg] = {dof: [None]*N_LAYERS for dof in DOF_GROUPS}
            return results

        # Load features: shape (T, n_layers, dim) -> mean over T -> (n_layers, dim)
        demo_features = {}
        for did in common_dids:
            grp = f[f"demo_{did}"]
            demo_features[did] = {}
            for agg in aggs:
                if agg in grp:
                    raw = grp[agg][:]  # (T, n_layers, dim)
                    demo_features[did][agg] = np.mean(raw, axis=0).astype(np.float32)  # (n_layers, dim)

        # Subsample actions
        demo_actions_sub = {}
        for did in common_dids:
            demo_actions_sub[did] = subsample_to_length(actions[did], N_SAMPLE)

    # Compute pairwise action distances per DoF group
    pairs = list(combinations(common_dids, 2))
    n_pairs = len(pairs)
    act_dists_per_dof = {dof: np.empty(n_pairs) for dof in DOF_GROUPS}

    for pi, (did_i, did_j) in enumerate(pairs):
        act_i = demo_actions_sub[did_i]
        act_j = demo_actions_sub[did_j]
        min_len = min(len(act_i), len(act_j))
        act_i_t = act_i[:min_len]
        act_j_t = act_j[:min_len]
        for dof_name, dims in DOF_GROUPS.items():
            if len(dims) == 1:
                d = np.mean(np.abs(act_i_t[:, dims[0]] - act_j_t[:, dims[0]]))
            else:
                d = np.mean(np.sqrt(np.sum((act_i_t[:, dims] - act_j_t[:, dims])**2, axis=1)))
            act_dists_per_dof[dof_name][pi] = d

    # Compute RSA per layer per DoF
    for agg in aggs:
        results[agg] = {}
        for dof_name in DOF_GROUPS:
            layer_rsa = []
            act_d = act_dists_per_dof[dof_name]
            act_std_ok = np.std(act_d) > 1e-12

            for layer in range(N_LAYERS):
                reps = []
                valid = True
                for did in common_dids:
                    if agg not in demo_features[did]:
                        valid = False
                        break
                    feat = demo_features[did][agg]  # (n_layers, dim)
                    if layer >= feat.shape[0]:
                        valid = False
                        break
                    reps.append(feat[layer])  # (dim,)

                if not valid or len(reps) < 2:
                    layer_rsa.append(None)
                    continue

                reps = np.array(reps, dtype=np.float64)  # (n_demos, dim)
                # Use scipy pdist for speed
                rep_dists = pdist(reps, metric='cosine')
                # Handle NaN from zero vectors
                rep_dists = np.nan_to_num(rep_dists, nan=1.0)

                if len(rep_dists) != len(act_d):
                    layer_rsa.append(None)
                    continue

                if np.std(rep_dists) < 1e-12 or not act_std_ok:
                    layer_rsa.append(None)
                    continue

                rho, _ = spearmanr(rep_dists, act_d)
                layer_rsa.append(float(rho))

            results[agg][dof_name] = layer_rsa

    return results


def main():
    t0 = time.time()
    print("RSA per-DoF decomposition", flush=True)
    print(f"DoF groups: {list(DOF_GROUPS.keys())}", flush=True)

    actions_cache = {}
    all_results = {}

    for cond_name, feat_dir in FEAT_DIRS.items():
        tasks = discover_tasks(feat_dir)
        print(f"\n{cond_name}: {len(tasks)} tasks", flush=True)
        all_results[cond_name] = {}

        for ti, task in enumerate(tasks):
            t1 = time.time()
            task_res = compute_rsa_per_dof(feat_dir, task, AGG_KEYS, actions_cache)
            all_results[cond_name][task] = task_res
            print(f"  [{ti+1}/{len(tasks)}] {task} ({time.time()-t1:.1f}s)", flush=True)

    # Aggregate: mean across tasks per layer
    conditions = {}
    for cond_name in FEAT_DIRS:
        tasks = list(all_results[cond_name].keys())
        for agg in AGG_KEYS:
            key = f"{cond_name}_{agg}"
            conditions[key] = {}
            for dof_name in DOF_GROUPS:
                layer_vals = []
                for layer in range(N_LAYERS):
                    task_rhos = []
                    for task in tasks:
                        v = all_results[cond_name][task].get(agg, {}).get(dof_name, [None]*N_LAYERS)[layer]
                        if v is not None:
                            task_rhos.append(v)
                    if task_rhos:
                        layer_vals.append(float(np.mean(task_rhos)))
                    else:
                        layer_vals.append(None)
                conditions[key][dof_name] = layer_vals

    # Peak summary
    peak_summary = {}
    for key in conditions:
        peak_summary[key] = {}
        for dof_name in DOF_GROUPS:
            vals = conditions[key][dof_name]
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_i, best_v = max(valid, key=lambda x: x[1])
                peak_summary[key][dof_name] = {"layer": best_i, "rsa": round(best_v, 4)}
            else:
                peak_summary[key][dof_name] = {"layer": None, "rsa": None}

    # Dominance analysis: at peak layer of full, compare xyz/rpy/gripper
    dominance = {}
    for key in conditions:
        full_peak = peak_summary[key]["full"]
        if full_peak["layer"] is not None:
            peak_layer = full_peak["layer"]
            contrib = {}
            for dof_name in ["xyz", "rpy", "gripper"]:
                v = conditions[key][dof_name][peak_layer]
                contrib[dof_name] = round(v, 4) if v is not None else None
            valid_contrib = {k: v for k, v in contrib.items() if v is not None}
            if valid_contrib:
                dominant = max(valid_contrib, key=valid_contrib.get)
            else:
                dominant = None
            dominance[key] = {
                "at_full_peak_layer": peak_layer,
                "most_contributing_dof": dominant,
                "relative_contributions": contrib,
            }
        else:
            dominance[key] = {"most_contributing_dof": None, "relative_contributions": {}}

    output = {
        "method": "Method B per-DoF RSA decomposition",
        "rep_distance": "cosine",
        "action_distance": "per-DoF euclidean (xyz/rpy) or absolute (gripper)",
        "n_sample": N_SAMPLE,
        "n_layers": N_LAYERS,
        "dof_groups": list(DOF_GROUPS.keys()),
        "dof_dims": {k: v for k, v in DOF_GROUPS.items()},
        "conditions": conditions,
        "peak_summary": peak_summary,
        "dominance_analysis": dominance,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}", flush=True)
    print(f"Saved to {OUT_PATH}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s\n", flush=True)

    print("Peak summary:", flush=True)
    for key in sorted(peak_summary.keys()):
        print(f"  {key}:")
        for dof_name in DOF_GROUPS:
            p = peak_summary[key][dof_name]
            print(f"    {dof_name}: layer {p['layer']}, RSA = {p['rsa']}")

    print("\nDominance analysis:", flush=True)
    for key in sorted(dominance.keys()):
        d = dominance[key]
        print(f"  {key}: dominant={d['most_contributing_dof']}, contributions={d.get('relative_contributions', {})}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
