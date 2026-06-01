"""RSA Permutation Test — null distribution for statistical significance.

Method B: temporal average of per-timestep pairwise distances.
  rep_dist(i,j) = mean_t cosine(rep_i[t], rep_j[t])
  act_dist(i,j) = mean_t euclidean(act_i[t], act_j[t])

Shuffles action labels within each task (permutes which demo's action
trajectory pairs with which demo's representation), recomputes RSA 1000 times.
Tests trained_image_mean on layers [0, 8, 12, 24, 32].
"""

import json
import os
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

FEAT_DIR = Path("./features/trained_libero_goal")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/ablations/rsa_permutation_null.json")

TEST_LAYERS = [0, 8, 12, 24, 32]
N_SAMPLE = 20
N_PERMUTATIONS = 1000
N_DEMOS = 50
BASE_SEED = 42


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def discover_tasks():
    return sorted([p.stem for p in FEAT_DIR.glob("*.h5")])


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


def compute_temporal_avg_dist_matrix(trajectories, metric='cosine'):
    """Compute pairwise distance matrix averaged over timesteps.
    
    trajectories: (n_demos, n_timesteps, dim)
    Returns: (n_demos, n_demos) symmetric distance matrix
    """
    n, T, d = trajectories.shape
    dist_sum = np.zeros((n, n))
    for t in range(T):
        dists_condensed = pdist(trajectories[:, t, :], metric=metric)
        dist_sum += squareform(dists_condensed)
    return dist_sum / T


def load_task_data(task):
    """Load feature trajectories and action trajectories.
    
    Returns:
        feat_trajs: {layer: (n_demos, N_SAMPLE, 4096)}
        act_trajs: (n_demos, N_SAMPLE, act_dim)
    """
    actions = load_actions(task)
    feat_path = FEAT_DIR / f"{task}.h5"
    if not feat_path.exists():
        return None, None

    with h5py.File(feat_path, "r") as f:
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        common_dids = sorted(
            set(int(k.split("_")[1]) for k in demo_keys) & set(actions.keys())
        )
        if len(common_dids) < 2:
            return None, None

        common_dids = common_dids[:N_DEMOS]
        n = len(common_dids)

        feat_trajs = {layer: [] for layer in TEST_LAYERS}
        act_trajs = []

        for did in common_dids:
            # image_mean: (T, 33, 4096) - spatial mean features
            feat_all = f[f"demo_{did}/image_mean"][:]
            feat_all = subsample_to_length(feat_all, N_SAMPLE)  # (N_SAMPLE, 33, 4096)

            for layer in TEST_LAYERS:
                feat_trajs[layer].append(feat_all[:, layer, :])  # (N_SAMPLE, 4096)

            act = actions[did]
            act = subsample_to_length(act, N_SAMPLE)  # (N_SAMPLE, act_dim)
            act_trajs.append(act)

        # Stack into arrays
        for layer in TEST_LAYERS:
            feat_trajs[layer] = np.array(feat_trajs[layer])  # (n, N_SAMPLE, 4096)
        act_trajs = np.array(act_trajs)  # (n, N_SAMPLE, act_dim)

    return feat_trajs, act_trajs


def main():
    t0 = time.time()
    tasks = discover_tasks()
    print(f"Found {len(tasks)} tasks")
    print(f"Testing layers: {TEST_LAYERS}")
    print(f"N_PERMUTATIONS: {N_PERMUTATIONS}")

    # Load all task data
    print("\nLoading features and actions...")
    all_data = {}
    for task in tasks:
        feats, acts = load_task_data(task)
        if feats is not None:
            all_data[task] = (feats, acts)
            print(f"  {task}: {acts.shape[0]} demos, {acts.shape[1]} timesteps")
    print(f"Loaded {len(all_data)} tasks with valid data")

    # Precompute distance matrices (temporal average of per-timestep distances)
    print("\nPrecomputing distance matrices...")
    rep_dist_matrices = {}  # {(task, layer): (n, n) matrix}
    act_dist_matrices = {}  # {task: (n, n) matrix}

    for task, (feats, acts) in all_data.items():
        # Action distances: average euclidean over timesteps
        act_dist_matrices[task] = compute_temporal_avg_dist_matrix(acts, metric='euclidean')

        for layer in TEST_LAYERS:
            # Rep distances: average cosine over timesteps
            rep_dist_matrices[(task, layer)] = compute_temporal_avg_dist_matrix(
                feats[layer], metric='cosine'
            )

    print(f"  Precomputation done ({time.time()-t0:.1f}s)")

    # Compute observed RSA
    print("\nComputing observed RSA...")
    observed_per_layer = {}
    triu_indices_cache = {}

    for layer in TEST_LAYERS:
        task_rhos = []
        for task, (feats, acts) in all_data.items():
            n = acts.shape[0]
            if n not in triu_indices_cache:
                triu_indices_cache[n] = np.triu_indices(n, k=1)
            triu = triu_indices_cache[n]

            rep_dists = rep_dist_matrices[(task, layer)][triu]
            act_dists = act_dist_matrices[task][triu]

            valid = np.isfinite(rep_dists) & np.isfinite(act_dists)
            if valid.sum() < 3:
                continue
            rho, _ = spearmanr(rep_dists[valid], act_dists[valid])
            if np.isfinite(rho):
                task_rhos.append(rho)

        observed_per_layer[layer] = np.mean(task_rhos) if task_rhos else None
        print(f"  Layer {layer}: observed RSA = {observed_per_layer[layer]:.4f} (n_tasks={len(task_rhos)})")

    # Permutation test
    print(f"\nRunning {N_PERMUTATIONS} permutations...")
    null_distributions = {layer: [] for layer in TEST_LAYERS}

    for perm_i in range(N_PERMUTATIONS):
        if (perm_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Permutation {perm_i+1}/{N_PERMUTATIONS} ({elapsed:.1f}s elapsed)")

        rng = np.random.RandomState(BASE_SEED + perm_i)

        for layer in TEST_LAYERS:
            task_rhos = []
            for task, (feats, acts) in all_data.items():
                n = acts.shape[0]
                triu = triu_indices_cache[n]

                # Shuffle action labels: permute rows/cols of action distance matrix
                perm_idx = rng.permutation(n)
                act_dist_perm = act_dist_matrices[task][perm_idx][:, perm_idx]
                act_dists = act_dist_perm[triu]

                rep_dists = rep_dist_matrices[(task, layer)][triu]

                valid = np.isfinite(rep_dists) & np.isfinite(act_dists)
                if valid.sum() < 3:
                    continue
                rho, _ = spearmanr(rep_dists[valid], act_dists[valid])
                if np.isfinite(rho):
                    task_rhos.append(rho)

            if task_rhos:
                null_distributions[layer].append(np.mean(task_rhos))

    # Compute statistics
    print("\nResults:")
    results = {}
    for layer in TEST_LAYERS:
        obs = observed_per_layer[layer]
        null = np.array(null_distributions[layer])
        null_mean = np.mean(null)
        null_std = np.std(null)
        null_max = np.max(null)
        p_value = np.mean(null >= obs) if obs is not None else None

        results[f"layer_{layer}"] = {
            "observed_rsa": round(float(obs), 6) if obs is not None else None,
            "null_mean": round(float(null_mean), 6),
            "null_std": round(float(null_std), 6),
            "null_max": round(float(null_max), 6),
            "null_min": round(float(np.min(null)), 6),
            "p_value": float(p_value) if p_value is not None else None,
            "significant_001": bool(p_value < 0.001) if p_value is not None else None,
        }
        sig = "***" if (p_value is not None and p_value < 0.001) else ""
        print(f"  Layer {layer}: obs={obs:.4f}, null_mean={null_mean:.4f}+/-{null_std:.4f}, "
              f"null_max={null_max:.4f}, p={p_value:.4f} {sig}")

    # Summary
    all_sig = all(r["significant_001"] for r in results.values() if r["significant_001"] is not None)
    if all_sig:
        conclusion = "All observed RSA values are statistically significant (p < 0.001, permutation test with 1000 shuffles)"
    else:
        sig_layers = [l for l in TEST_LAYERS if results[f"layer_{l}"]["significant_001"]]
        nonsig_layers = [l for l in TEST_LAYERS if not results[f"layer_{l}"]["significant_001"]]
        if sig_layers and not nonsig_layers:
            conclusion = "All observed RSA values are statistically significant (p < 0.001, permutation test with 1000 shuffles)"
        elif sig_layers:
            conclusion = f"Layers {sig_layers} significant at p<0.001. Layers {nonsig_layers} not significant at p<0.001."
        else:
            conclusion = f"No layers significant at p<0.001. P-values: " + ", ".join(
                f"L{l}={results[f'layer_{l}']['p_value']:.4f}" for l in TEST_LAYERS
            )

    output = {
        "method": "RSA permutation test (1000 shuffles)",
        "condition": "trained_image_mean",
        "rep_distance": "cosine",
        "action_distance": "euclidean",
        "distance_computation": "temporal average of per-timestep pairwise distances (Method B)",
        "n_permutations": N_PERMUTATIONS,
        "n_sample": N_SAMPLE,
        "base_seed": BASE_SEED,
        "n_tasks": len(all_data),
        "layers_tested": TEST_LAYERS,
        "results": results,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved to {OUT_PATH}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"\nConclusion: {conclusion}")


if __name__ == "__main__":
    main()
