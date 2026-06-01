"""RSA ablation: cosine action distance (replacing euclidean) on LIBERO-Goal.

Method B framework:
  - rep_dist: cosine
  - act_dist: cosine (ablation; original uses euclidean)
  - N_SAMPLE: 20 (np.linspace subsample)
  - pairing: within-task itertools.combinations
  - aggregation: per-task Spearman -> mean + bootstrap CI (1000 resamples, seed=42)

Input:
  - Features: ./features/{trained,untrained}_libero_goal/*.h5
  - Actions: ./data/libero/libero_goal/{task}_demo.hdf5

Output:
  - ./results/ablations/rsa_cosine_metric.json
"""

import json
import os
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

FEAT_BASE = Path("./features")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/ablations/rsa_cosine_metric.json")
EUCLIDEAN_PATH = Path("./results/rsa/unified_rsa_results.json")

FEAT_DIRS = {
    "trained": FEAT_BASE / "trained_libero_goal",
    "untrained": FEAT_BASE / "untrained_libero_goal",
}

AGG_KEYS = ["image_mean", "last_preaction"]
N_LAYERS = 33
N_SAMPLE = 20

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


def cosine_pdist(X):
    """Cosine distance pairwise, handling zero vectors."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    zero_mask = (norms.ravel() < 1e-12)
    X_safe = X.copy()
    X_safe[zero_mask] = 0.0
    norms_safe = norms.copy()
    norms_safe[zero_mask] = 1.0
    X_normed = X_safe / norms_safe

    n = X.shape[0]
    n_pairs = n * (n - 1) // 2
    dists = np.empty(n_pairs, dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if zero_mask[i] or zero_mask[j]:
                dists[idx] = 1.0
            else:
                cos_sim = np.dot(X_normed[i], X_normed[j])
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                dists[idx] = 1.0 - cos_sim
            idx += 1
    return dists


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    rng = np.random.RandomState(42)
    boot_means = [
        np.mean(rng.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def compute_rsa_all_layers(feat_dir, task, aggs, actions_cache):
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

        # Cosine action distance (the ablation change)
        sub_acts = np.stack(
            [subsample_to_length(actions[d], N_SAMPLE) for d in common_dids]
        )
        act_dists = np.zeros(n_pairs)
        for t in range(sub_acts.shape[1]):
            act_dists += cosine_pdist(sub_acts[:, t, :])
        act_dists /= sub_acts.shape[1]

        for agg in aggs:
            sub_feats = []
            for did in common_dids:
                raw = f[f"demo_{did}/{agg}"][:]
                sub = subsample_to_length(raw, N_SAMPLE).astype(np.float32)
                sub_feats.append(sub)

            feat_stack = np.stack(sub_feats)

            layer_results = {}
            for layer in range(N_LAYERS):
                rep_dists = np.zeros(n_pairs)
                for t in range(feat_stack.shape[1]):
                    rep_dists += pdist(feat_stack[:, t, layer, :], metric="cosine")
                rep_dists /= feat_stack.shape[1]

                valid = np.isfinite(rep_dists) & np.isfinite(act_dists)
                if valid.sum() < 3:
                    layer_results[layer] = None
                    continue

                rho, _ = spearmanr(rep_dists[valid], act_dists[valid])
                layer_results[layer] = float(rho) if np.isfinite(rho) else None

            results[agg] = layer_results

    return results


def load_euclidean_profiles():
    """Load existing euclidean RSA per-layer profiles for comparison."""
    if not EUCLIDEAN_PATH.exists():
        return None

    with open(EUCLIDEAN_PATH) as f:
        data = json.load(f)

    profiles = {}
    cond_map = {
        "trained_image_mean": ("trained_with_inst", "image_mean"),
        "trained_last_preaction": ("trained_with_inst", "last_preaction"),
        "untrained_image_mean": ("untrained_with_inst", "image_mean"),
        "untrained_last_preaction": ("untrained_with_inst", "last_preaction"),
    }

    for key, (cond, agg) in cond_map.items():
        if cond in data.get("conditions", {}):
            agg_data = data["conditions"][cond].get(agg, {})
            vals = []
            for layer in range(N_LAYERS):
                lk = f"layer_{layer}"
                if lk in agg_data and agg_data[lk].get("mean_rsa") is not None:
                    vals.append(agg_data[lk]["mean_rsa"])
                else:
                    vals.append(None)
            profiles[key] = vals

    return profiles


def main():
    np.random.seed(42)
    t0 = time.time()

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
            }
            if valid_layers:
                best_k = max(valid_layers, key=valid_layers.get)
                best_l = int(best_k.split("_")[1])
                best_r = valid_layers[best_k]
                print(f"  BEST: layer {best_l}, RSA = {best_r:.4f}")

        all_results[cond] = cond_results

    # Build per-condition flat arrays for output
    conditions_flat = {}
    peak_layers = {}
    peak_rsa_values = {}

    for cond in ["trained", "untrained"]:
        if cond not in all_results:
            continue
        for agg in AGG_KEYS:
            key = f"{cond}_{agg}"
            layer_res = all_results[cond][agg]
            vals = []
            for layer in range(N_LAYERS):
                lk = f"layer_{layer}"
                v = layer_res[lk]["mean_rsa"] if lk in layer_res else None
                vals.append(v)
            conditions_flat[key] = vals

            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_i, best_v = max(valid, key=lambda x: x[1])
                peak_layers[key] = best_i
                peak_rsa_values[key] = best_v

    # Comparison with euclidean
    euclidean_profiles = load_euclidean_profiles()
    comparison = {}

    if euclidean_profiles:
        for key in conditions_flat:
            cos_vals = conditions_flat[key]
            euc_key = key
            if euc_key in euclidean_profiles:
                euc_vals = euclidean_profiles[euc_key]
                paired = [(c, e) for c, e in zip(cos_vals, euc_vals) if c is not None and e is not None]
                if len(paired) >= 3:
                    cos_arr = np.array([p[0] for p in paired])
                    euc_arr = np.array([p[1] for p in paired])
                    rho, _ = spearmanr(cos_arr, euc_arr)
                    comparison[f"{key}_corr_with_euclidean"] = round(float(rho), 4)

        # Check inverted-U replication for trained_image_mean
        trained_im_cos = conditions_flat.get("trained_image_mean", [])
        valid_cos = [v for v in trained_im_cos if v is not None]
        if len(valid_cos) >= 10:
            peak_idx = np.argmax(valid_cos)
            n_layers_valid = len(valid_cos)
            inverted_u = bool((peak_idx > n_layers_valid * 0.15) and (peak_idx < n_layers_valid * 0.85))
        else:
            inverted_u = None

        euc_peak = None
        cos_peak = peak_layers.get("trained_image_mean")
        if "trained_image_mean" in euclidean_profiles:
            euc_profile = euclidean_profiles["trained_image_mean"]
            euc_valid = [v for v in euc_profile if v is not None]
            if euc_valid:
                euc_peak = np.argmax(euc_valid)

        overall_corr = comparison.get("trained_image_mean_corr_with_euclidean")

        comparison_summary = {
            "inverted_u_replicated": inverted_u,
            "cosine_peak_layer": int(cos_peak) if cos_peak is not None else None,
            "euclidean_peak_layer": int(euc_peak) if euc_peak is not None else None,
            "peak_layer_shift": int(cos_peak - euc_peak) if (cos_peak is not None and euc_peak is not None) else None,
            "correlation_with_euclidean_profile": {
                k: v for k, v in comparison.items()
            },
        }
    else:
        comparison_summary = {"error": "euclidean results not found"}

    task_list = discover_tasks(FEAT_DIRS["trained"])

    output = {
        "method": "Method B with cosine action distance",
        "rep_distance": "cosine",
        "action_distance": "cosine",
        "n_sample": N_SAMPLE,
        "n_layers": N_LAYERS,
        "n_tasks": len(task_list),
        "tasks": task_list,
        "seed": 42,
        "conditions": conditions_flat,
        "peak_layers": peak_layers,
        "peak_rsa_values": peak_rsa_values,
        "comparison_with_euclidean": comparison_summary,
        "full_results": all_results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"Saved to {OUT_PATH}")
    print(f"Total time: {time.time()-t0:.1f}s\n")

    print("Peak layers (cosine action dist):")
    for k, v in peak_layers.items():
        print(f"  {k}: layer {v}, RSA = {peak_rsa_values[k]:.4f}")

    print("\nComparison with euclidean:")
    if isinstance(comparison_summary, dict) and "error" not in comparison_summary:
        print(f"  Inverted-U replicated: {comparison_summary['inverted_u_replicated']}")
        print(f"  Peak layer shift: {comparison_summary['peak_layer_shift']}")
        for k, v in comparison_summary.get("correlation_with_euclidean_profile", {}).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
