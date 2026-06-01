import json
import time
import numpy as np
from pathlib import Path
import h5py
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_ROOT = Path("./features")
DATA_ROOT = Path("./data/libero/libero_goal")
OUT_DIR   = Path("./results/rsa")

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

TEMPORAL_BINS = {
    "early": (0.0, 0.33),
    "mid":   (0.33, 0.67),
    "late":  (0.67, 1.0),
}

LAYER = 23
N_BOOTSTRAP = 10000
DEMO_IDS = list(range(10))


def get_temporal_indices(T, phase):
    lo, hi = TEMPORAL_BINS[phase]
    t_norm = np.linspace(0, 1, T, endpoint=False)
    mask = (t_norm >= lo) & (t_norm < hi)
    if phase == "late":
        mask = (t_norm >= lo) & (t_norm <= 1.0)
    return np.where(mask)[0]


def subsample_to_length(arr, target_len):
    T = arr.shape[0]
    if T <= target_len:
        return arr
    idx = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[idx]


def compute_mean_cosine_distance(feat_a, feat_b, n_sample=20):
    L = min(len(feat_a), len(feat_b), n_sample)
    a = subsample_to_length(feat_a, L)
    b = subsample_to_length(feat_b, L)
    dists = []
    for i in range(L):
        na = np.linalg.norm(a[i])
        nb = np.linalg.norm(b[i])
        if na < 1e-8 or nb < 1e-8:
            dists.append(1.0)
        else:
            dists.append(cosine_dist(a[i], b[i]))
    return float(np.mean(dists))


def build_dissimilarity_matrix(data, keys, phase):
    N = len(keys)
    dm = np.zeros((N, N), dtype=np.float32)
    phase_cache = {}
    for i, k in enumerate(keys):
        arr = data[k]
        idx = get_temporal_indices(len(arr), phase)
        if len(idx) == 0:
            idx = np.array([0])
        phase_cache[i] = arr[idx]
    for i in range(N):
        for j in range(i + 1, N):
            d = compute_mean_cosine_distance(phase_cache[i], phase_cache[j])
            dm[i, j] = d
            dm[j, i] = d
    return dm


def compute_rsa(rdm, adm):
    idx = np.triu_indices_from(rdm, k=1)
    r_vec = rdm[idx]
    a_vec = adm[idx]
    if np.std(r_vec) < 1e-10 or np.std(a_vec) < 1e-10:
        return 0.0
    rho, _ = spearmanr(r_vec, a_vec)
    return float(rho)


def compute_per_task_rsa(task, model_type, token_key):
    feat_path = FEAT_ROOT / f"{model_type}_libero_goal" / f"{task}.h5"
    act_path = DATA_ROOT / f"{task}_demo.hdf5"

    feats = {}
    with h5py.File(feat_path, "r") as f:
        for did in DEMO_IDS:
            key = f"demo_{did}/{token_key}"
            if key in f:
                feats[did] = f[key][:, LAYER, :].astype(np.float32)

    actions = {}
    with h5py.File(act_path, "r") as f:
        for did in DEMO_IDS:
            key = f"data/demo_{did}/actions"
            if key in f:
                actions[did] = f[key][:].astype(np.float32)

    common = sorted(set(feats.keys()) & set(actions.keys()))
    if len(common) < 2:
        return None

    feat_dict = {did: feats[did] for did in common}
    act_dict = {did: actions[did] for did in common}

    phase_rsa = []
    for phase in TEMPORAL_BINS:
        adm = build_dissimilarity_matrix(act_dict, common, phase)
        rdm = build_dissimilarity_matrix(feat_dict, common, phase)
        rsa = compute_rsa(rdm, adm)
        phase_rsa.append(rsa)

    return float(np.mean(phase_rsa))


def bootstrap_ci(trained_vals, untrained_vals, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = len(trained_vals)
    t_arr = np.array(trained_vals)
    u_arr = np.array(untrained_vals)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas[i] = t_arr[idx].mean() - u_arr[idx].mean()

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0))

    return {
        "trained_mean": float(t_arr.mean()),
        "untrained_mean": float(u_arr.mean()),
        "delta": float(t_arr.mean() - u_arr.mean()),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "p_value": round(p_value, 4),
        "n_bootstrap": n_boot,
        "bootstrap_delta_mean": round(float(deltas.mean()), 6),
        "bootstrap_delta_std": round(float(deltas.std()), 6),
        "per_task_trained": [round(v, 4) for v in trained_vals],
        "per_task_untrained": [round(v, 4) for v in untrained_vals],
        "per_task_delta": [round(t - u, 4) for t, u in zip(trained_vals, untrained_vals)],
    }


def main():
    t0 = time.time()

    results = {}
    for token_key in ["image_mean", "last_preaction"]:
        print(f"\n=== {token_key} (layer {LAYER}) ===")
        trained_rsa = []
        untrained_rsa = []
        for task in TASKS:
            t_rsa = compute_per_task_rsa(task, "trained", token_key)
            u_rsa = compute_per_task_rsa(task, "untrained", token_key)
            if t_rsa is not None and u_rsa is not None:
                trained_rsa.append(t_rsa)
                untrained_rsa.append(u_rsa)
                print(f"  {task[:40]:40s}  T={t_rsa:.4f}  U={u_rsa:.4f}  Δ={t_rsa-u_rsa:+.4f}")
            else:
                print(f"  {task[:40]:40s}  SKIPPED (missing data)")

        print(f"\n  Mean: T={np.mean(trained_rsa):.4f}  U={np.mean(untrained_rsa):.4f}  Δ={np.mean(trained_rsa)-np.mean(untrained_rsa):+.4f}")

        boot = bootstrap_ci(trained_rsa, untrained_rsa)
        print(f"  Bootstrap Δ: {boot['delta']:.4f}  95% CI: [{boot['ci_95_lower']:.4f}, {boot['ci_95_upper']:.4f}]  p={boot['p_value']:.4f}")
        results[token_key] = boot

    # conclusion: only state whether CI excludes 0
    im_ci = results["image_mean"]
    lp_ci = results["last_preaction"]
    conclusions = []
    if im_ci["ci_95_lower"] > 0:
        conclusions.append("image_mean: CI excludes 0 (significant)")
    else:
        conclusions.append("image_mean: CI includes 0 (not significant)")
    if lp_ci["ci_95_lower"] > 0:
        conclusions.append("last_preaction: CI excludes 0 (significant)")
    else:
        conclusions.append("last_preaction: CI includes 0 (not significant)")
    results["conclusion"] = "; ".join(conclusions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bootstrap_training_effect.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nSaved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Conclusion: {results['conclusion']}")


if __name__ == "__main__":
    main()
