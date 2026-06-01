"""RSA (Representational Similarity Analysis) for LIBERO-Goal VLA features.

Input:
  - Trained/Untrained VLA features: ./features/{trained,untrained}_libero_goal/*.h5
    Keys: demo_X/last_preaction, shape=(T, 33, 4096)
  - DINO features: ./features/dinov2_libero_goal/*.h5
    Keys: traj_X/cls_token, shape=(T, 1024)
  - Actions: ./data/libero/libero_goal/*_demo.hdf5
    Keys: data/demo_X/actions, shape=(T, 7)

Output:
  - ./results/rsa/rsa_results.json
  - stdout summary table

Dependencies: numpy, scipy, h5py, json, argparse
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
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
N_SAMPLE_FRAMES = 20  # subsample each phase to this many frames


def load_vla_features(model_type, task, demo_ids, layer, suffix=""):
    path = FEAT_ROOT / f"{model_type}_libero_goal{suffix}" / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"demo_{did}/last_preaction"
            if key in f:
                result[did] = f[key][:, layer, :].astype(np.float32)
    return result


def load_dino_features(task, demo_ids):
    path = FEAT_ROOT / "dinov2_libero_goal" / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"traj_{did}/cls_token"
            if key in f:
                result[did] = f[key][:].astype(np.float32)
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


def get_phase_slice(arr, phase):
    """Extract temporal phase from trajectory and subsample to N_SAMPLE_FRAMES."""
    T = len(arr)
    lo, hi = TEMPORAL_BINS[phase]
    t_norm = np.linspace(0, 1, T, endpoint=False)
    if phase == "late":
        mask = (t_norm >= lo) & (t_norm <= 1.0)
    else:
        mask = (t_norm >= lo) & (t_norm < hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.array([0])
    seg = arr[idx]
    if len(seg) > N_SAMPLE_FRAMES:
        sample_idx = np.linspace(0, len(seg) - 1, N_SAMPLE_FRAMES, dtype=int)
        seg = seg[sample_idx]
    return seg


def build_phase_tensor(data, keys, phase):
    """Stack all demos' phase features into (N, n_frames, D) tensor, padded to max length."""
    slices = [get_phase_slice(data[k], phase) for k in keys]
    max_len = max(len(s) for s in slices)
    D = slices[0].shape[1]
    N = len(slices)
    tensor = np.zeros((N, max_len, D), dtype=np.float32)
    lengths = np.zeros(N, dtype=int)
    for i, s in enumerate(slices):
        tensor[i, :len(s)] = s
        lengths[i] = len(s)
    return tensor, lengths


def pairwise_cosine_dm_vectorized(tensor, lengths):
    """Compute pairwise mean cosine distance matrix. tensor: (N, T, D), lengths: (N,)."""
    N, T, D = tensor.shape
    # L2-normalize each frame
    norms = np.linalg.norm(tensor, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = tensor / norms  # (N, T, D)

    # zero out padded frames
    for i in range(N):
        if lengths[i] < T:
            normed[i, lengths[i]:] = 0.0

    # pairwise cosine similarity per frame: sim[i,j,t] = dot(normed[i,t], normed[j,t])
    # Use einsum for (N, T, D) x (N, T, D) -> (N, N, T)
    sim = np.einsum('itd,jtd->ijt', normed, normed)  # (N, N, T)

    # build valid mask: frame t is valid for pair (i,j) only if t < min(lengths[i], lengths[j])
    min_lens = np.minimum(lengths[:, None], lengths[None, :])  # (N, N)
    frame_idx = np.arange(T)[None, None, :]  # (1, 1, T)
    valid = frame_idx < min_lens[:, :, None]  # (N, N, T)

    # cosine distance = 1 - sim, averaged over valid frames
    dist = 1.0 - sim
    dist = dist * valid
    count = valid.sum(axis=2).astype(np.float32)
    count = np.maximum(count, 1.0)
    dm = dist.sum(axis=2) / count

    return dm.astype(np.float32)


def compute_rsa(rdm, adm):
    """Spearman correlation between upper triangles."""
    idx = np.triu_indices_from(rdm, k=1)
    r_vec = rdm[idx]
    a_vec = adm[idx]
    if np.std(r_vec) < 1e-10 or np.std(a_vec) < 1e-10:
        return 0.0
    rho, _ = spearmanr(r_vec, a_vec)
    return float(rho)


def linear_cka(X, Y):
    """Linear CKA between (N, D1) and (N, D2)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def main():
    parser = argparse.ArgumentParser(description="RSA analysis for LIBERO-Goal VLA features")
    parser.add_argument("--n_demos", type=int, default=10, help="demos per task (default: 10)")
    parser.add_argument("--n_tasks", type=int, default=10, help="number of tasks (default: 10)")
    parser.add_argument("--layers", type=str, default="all",
                        help="'all' or comma-separated (e.g. '0,8,16,24,32')")
    parser.add_argument("--feat_suffix", type=str, default="", help="suffix for feature dirs, e.g. _no_inst")
    parser.add_argument("--skip_cka", action="store_true", help="skip CKA computation")
    args = parser.parse_args()

    tasks = TASKS[:args.n_tasks]
    demo_ids = list(range(args.n_demos))

    if args.layers == "all":
        layers = list(range(N_VLA_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"Config: {len(tasks)} tasks x {args.n_demos} demos = {len(tasks) * args.n_demos} trajectories")
    print(f"Layers: {len(layers)} | CKA: {'skip' if args.skip_cka else 'yes'}")
    print()

    # -- load actions --
    print("Loading actions...")
    all_actions = {}
    for task in tasks:
        act = load_actions(task, demo_ids)
        for did, a in act.items():
            all_actions[(task, did)] = a
    print(f"  Loaded {len(all_actions)} demos")

    # -- load DINO features --
    print("Loading DINO features...")
    dino_feats = {}
    for task in tasks:
        df = load_dino_features(task, demo_ids)
        for did in df:
            dino_feats[(task, did)] = df[did]
    print(f"  Loaded {len(dino_feats)} DINO demos")

    common_keys_base = sorted(set(dino_feats) & set(all_actions))
    N_demos = len(common_keys_base)
    print(f"  Common demos: {N_demos}")

    # -- precompute ADM and DINO RDM per phase --
    print("Precomputing ADM and DINO RDM (vectorized)...")
    adm_cache = {}
    dino_rsa_cache = {}
    for phase in TEMPORAL_BINS:
        act_tensor, act_lens = build_phase_tensor(all_actions, common_keys_base, phase)
        adm_cache[phase] = pairwise_cosine_dm_vectorized(act_tensor, act_lens)

        dino_tensor, dino_lens = build_phase_tensor(dino_feats, common_keys_base, phase)
        dino_rdm = pairwise_cosine_dm_vectorized(dino_tensor, dino_lens)
        dino_rsa_cache[phase] = compute_rsa(dino_rdm, adm_cache[phase])
    print("  Done")

    # -- per-layer RSA --
    results = {
        "per_layer": {},
        "per_timestep": {phase: {} for phase in TEMPORAL_BINS},
        "cka": {},
        "gates": {},
    }

    phase_rsa_accum = {phase: {"trained": [], "untrained": [], "dino": []}
                       for phase in TEMPORAL_BINS}

    for li, layer in enumerate(layers):
        t0 = time.time()
        layer_key = f"layer_{layer}"

        trained_feats = {}
        untrained_feats = {}
        for task in tasks:
            tf = load_vla_features("trained", task, demo_ids, layer, suffix=args.feat_suffix)
            uf = load_vla_features("untrained", task, demo_ids, layer, suffix=args.feat_suffix)
            for did in tf:
                trained_feats[(task, did)] = tf[did]
            for did in uf:
                untrained_feats[(task, did)] = uf[did]

        common_keys = sorted(set(common_keys_base) & set(trained_feats) & set(untrained_feats))

        layer_rsa = {"trained": {}, "untrained": {}, "dino": {}}
        for phase in TEMPORAL_BINS:
            adm = adm_cache[phase]

            t_tensor, t_lens = build_phase_tensor(trained_feats, common_keys, phase)
            u_tensor, u_lens = build_phase_tensor(untrained_feats, common_keys, phase)

            rdm_t = pairwise_cosine_dm_vectorized(t_tensor, t_lens)
            rdm_u = pairwise_cosine_dm_vectorized(u_tensor, u_lens)

            rsa_t = compute_rsa(rdm_t, adm)
            rsa_u = compute_rsa(rdm_u, adm)
            rsa_d = dino_rsa_cache[phase]

            layer_rsa["trained"][phase] = rsa_t
            layer_rsa["untrained"][phase] = rsa_u
            layer_rsa["dino"][phase] = rsa_d

            phase_rsa_accum[phase]["trained"].append(rsa_t)
            phase_rsa_accum[phase]["untrained"].append(rsa_u)
            phase_rsa_accum[phase]["dino"].append(rsa_d)

        rsa_t_all = np.mean([layer_rsa["trained"][p] for p in TEMPORAL_BINS])
        rsa_u_all = np.mean([layer_rsa["untrained"][p] for p in TEMPORAL_BINS])
        rsa_d_all = np.mean([layer_rsa["dino"][p] for p in TEMPORAL_BINS])

        results["per_layer"][layer_key] = {
            "trained_rsa": round(float(rsa_t_all), 4),
            "untrained_rsa": round(float(rsa_u_all), 4),
            "dino_rsa": round(float(rsa_d_all), 4),
            "per_phase": {
                phase: {
                    "trained_rsa": round(layer_rsa["trained"][phase], 4),
                    "untrained_rsa": round(layer_rsa["untrained"][phase], 4),
                    "dino_rsa": round(layer_rsa["dino"][phase], 4),
                }
                for phase in TEMPORAL_BINS
            },
        }

        if not args.skip_cka:
            t_mat = np.stack([trained_feats[k].mean(axis=0) for k in common_keys])
            u_mat = np.stack([untrained_feats[k].mean(axis=0) for k in common_keys])
            cka_val = linear_cka(t_mat, u_mat)
            results["cka"][layer_key] = round(cka_val, 4)

        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{layers[-1]}: trained={rsa_t_all:.4f}  untrained={rsa_u_all:.4f}  "
              f"dino={rsa_d_all:.4f}  [{elapsed:.1f}s]")

    # -- per-timestep summary --
    for phase in TEMPORAL_BINS:
        results["per_timestep"][phase] = {
            "trained_rsa": round(float(np.mean(phase_rsa_accum[phase]["trained"])), 4),
            "untrained_rsa": round(float(np.mean(phase_rsa_accum[phase]["untrained"])), 4),
            "dino_rsa": round(float(np.mean(phase_rsa_accum[phase]["dino"])), 4),
        }

    # -- gates --
    trained_peaks = [results["per_layer"][f"layer_{l}"]["trained_rsa"] for l in layers]
    untrained_peaks = [results["per_layer"][f"layer_{l}"]["untrained_rsa"] for l in layers]

    g1_trained_peak = max(trained_peaks)
    g1_untrained_peak = max(untrained_peaks)
    g1_delta = g1_trained_peak - g1_untrained_peak
    g1_pass = g1_delta > 0.05

    g2_count = sum(1 for l in layers
                   if results["per_layer"][f"layer_{l}"]["trained_rsa"]
                   > results["per_layer"][f"layer_{l}"]["untrained_rsa"])
    g2_pass = g2_count > len(layers) / 2

    results["gates"] = {
        "G1_trained_peak": round(g1_trained_peak, 4),
        "G1_untrained_peak": round(g1_untrained_peak, 4),
        "G1_delta": round(g1_delta, 4),
        "G1_pass": bool(g1_pass),
        "G2_layers_trained_gt_untrained": g2_count,
        "G2_total_layers": len(layers),
        "G2_pass": bool(g2_pass),
    }

    # -- save --
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix_tag = args.feat_suffix.strip("_") if args.feat_suffix else "normal"
    out_path = OUT_DIR / f"rsa_{suffix_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # -- summary table --
    print("\n" + "=" * 80)
    print("RSA SUMMARY (Spearman corr between representation & action dissimilarity)")
    print("=" * 80)

    print(f"\n{'Layer':>8} | {'Trained':>10} | {'Untrained':>10} | {'DINO':>10} | {'T-U Delta':>10}"
          + (f" | {'CKA(T,U)':>10}" if not args.skip_cka else ""))
    print("-" * (55 + (13 if not args.skip_cka else 0)))
    for l in layers:
        lk = f"layer_{l}"
        r = results["per_layer"][lk]
        row = (f"{l:>8d} | {r['trained_rsa']:>10.4f} | {r['untrained_rsa']:>10.4f} | "
               f"{r['dino_rsa']:>10.4f} | {r['trained_rsa'] - r['untrained_rsa']:>+10.4f}")
        if not args.skip_cka and lk in results["cka"]:
            row += f" | {results['cka'][lk]:>10.4f}"
        print(row)

    print(f"\n{'Phase':>8} | {'Trained':>10} | {'Untrained':>10} | {'DINO':>10}")
    print("-" * 48)
    for phase in TEMPORAL_BINS:
        r = results["per_timestep"][phase]
        print(f"{phase:>8} | {r['trained_rsa']:>10.4f} | {r['untrained_rsa']:>10.4f} | "
              f"{r['dino_rsa']:>10.4f}")

    print(f"\nGates:")
    g = results["gates"]
    print(f"  G1: peak RSA trained={g['G1_trained_peak']:.4f} vs untrained={g['G1_untrained_peak']:.4f}"
          f"  delta={g['G1_delta']:+.4f}  {'PASS' if g['G1_pass'] else 'FAIL'}")
    print(f"  G2: trained > untrained in {g['G2_layers_trained_gt_untrained']}/{g['G2_total_layers']} layers"
          f"  {'PASS' if g['G2_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
