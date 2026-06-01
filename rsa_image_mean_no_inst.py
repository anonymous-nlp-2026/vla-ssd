import json
import time
from pathlib import Path

import h5py
import numpy as np
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

N_VLA_LAYERS = 33


def load_vla_features(model_type, task, demo_ids, layer, token_key="image_mean"):
    path = FEAT_ROOT / f"{model_type}_libero_goal_no_inst" / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"demo_{did}/{token_key}"
            if key in f:
                data = f[key][:, layer, :]
                result[did] = data.astype(np.float32)
    return result


def load_actions(task, demo_ids):
    path = DATA_ROOT / f"{task}_demo.hdf5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"data/demo_{did}/actions"
            if key in f:
                result[did] = f[key][:].astype(np.float32)
    return result


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


def main():
    tasks = TASKS
    demo_ids = list(range(10))
    layers = list(range(N_VLA_LAYERS))

    print(f"RSA with image_mean token aggregation (NO-INST condition)")
    print(f"Config: {len(tasks)} tasks x {len(demo_ids)} demos = {len(tasks) * len(demo_ids)} trajectories")

    print("Loading actions...")
    all_actions = {}
    for task in tasks:
        act = load_actions(task, demo_ids)
        for did, a in act.items():
            all_actions[(task, did)] = a
    print(f"  Loaded {len(all_actions)} demos")

    sorted_keys = sorted(all_actions.keys())

    print("Precomputing ADM...")
    act_common = {k: all_actions[k] for k in sorted_keys}
    adm_cache = {
        phase: build_dissimilarity_matrix(act_common, sorted_keys, phase)
        for phase in TEMPORAL_BINS
    }

    results = {"per_layer": {}}

    for layer in layers:
        t0 = time.time()
        layer_key = f"layer_{layer}"

        feat_t = {}
        feat_u = {}
        for task in tasks:
            ft = load_vla_features("trained", task, demo_ids, layer, token_key="image_mean")
            fu = load_vla_features("untrained", task, demo_ids, layer, token_key="image_mean")
            for did in demo_ids:
                k = (task, did)
                if did in ft and k in act_common:
                    feat_t[k] = ft[did]
                if did in fu and k in act_common:
                    feat_u[k] = fu[did]

        keys = sorted(set(feat_t.keys()) & set(feat_u.keys()) & set(act_common.keys()))

        layer_rsa = {"trained": {}, "untrained": {}}
        phase_rsa_accum = {p: {"trained": [], "untrained": []} for p in TEMPORAL_BINS}

        for phase in TEMPORAL_BINS:
            adm = adm_cache[phase]
            rdm_t = build_dissimilarity_matrix(feat_t, keys, phase)
            rdm_u = build_dissimilarity_matrix(feat_u, keys, phase)
            rsa_t = compute_rsa(rdm_t, adm)
            rsa_u = compute_rsa(rdm_u, adm)
            layer_rsa["trained"][phase] = rsa_t
            layer_rsa["untrained"][phase] = rsa_u

        rsa_t_all = np.mean([layer_rsa["trained"][p] for p in TEMPORAL_BINS])
        rsa_u_all = np.mean([layer_rsa["untrained"][p] for p in TEMPORAL_BINS])

        results["per_layer"][layer_key] = {
            "trained_rsa": round(float(rsa_t_all), 4),
            "untrained_rsa": round(float(rsa_u_all), 4),
            "per_phase": {
                phase: {
                    "trained_rsa": round(layer_rsa["trained"][phase], 4),
                    "untrained_rsa": round(layer_rsa["untrained"][phase], 4),
                }
                for phase in TEMPORAL_BINS
            },
        }

        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{layers[-1]}: trained={rsa_t_all:.4f}  untrained={rsa_u_all:.4f}  [{elapsed:.1f}s]")

    trained_vals = [results["per_layer"][f"layer_{l}"]["trained_rsa"] for l in layers]
    untrained_vals = [results["per_layer"][f"layer_{l}"]["untrained_rsa"] for l in layers]

    trained_peak = max(trained_vals)
    trained_peak_layer = layers[np.argmax(trained_vals)]
    untrained_peak = max(untrained_vals)
    untrained_peak_layer = layers[np.argmax(untrained_vals)]

    output = {
        "token_key": "image_mean",
        "condition": "no_inst",
        "trained_no_inst": {
            "peak_rsa": trained_peak,
            "peak_layer": trained_peak_layer,
            "per_layer_rsa": trained_vals,
        },
        "untrained_no_inst": {
            "peak_rsa": untrained_peak,
            "peak_layer": untrained_peak_layer,
            "per_layer_rsa": untrained_vals,
        },
        "per_layer_detail": results["per_layer"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rsa_image_mean_no_inst.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*60}")
    print(f"SUMMARY: image_mean RSA (no-inst)")
    print(f"{'='*60}")
    print(f"Trained   peak RSA: {trained_peak:.4f} (layer {trained_peak_layer})")
    print(f"Untrained peak RSA: {untrained_peak:.4f} (layer {untrained_peak_layer})")
    print(f"\n{'Layer':>8} | {'Trained':>10} | {'Untrained':>10} | {'T-U Delta':>10}")
    print("-" * 48)
    for l in layers:
        lk = f"layer_{l}"
        r = results["per_layer"][lk]
        print(f"{l:>8d} | {r['trained_rsa']:>10.4f} | {r['untrained_rsa']:>10.4f} | {r['trained_rsa'] - r['untrained_rsa']:>+10.4f}")


if __name__ == "__main__":
    main()
