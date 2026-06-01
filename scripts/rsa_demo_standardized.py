"""RSA with standardized 10 demos (demo 0-9) across all models.

Method B: within-task pairs, cosine rep distance, euclidean action distance,
subsample to 20 timesteps, per-task Spearman -> mean across tasks.
"""
import itertools, json, os
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_ROOT = Path("./features")
DATA_ROOT = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/rsa/rsa_demo_standardized.json")

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
DEMO_IDS = list(range(10))  # demo_0 through demo_9
N_BOOT = 1000
SEED = 42


def subsample(arr, n):
    T = arr.shape[0]
    if T <= n:
        return arr
    return arr[np.linspace(0, T - 1, n, dtype=int)]


def mean_cosine_distance(fa, fb):
    a, b = subsample(fa, N_SAMPLE), subsample(fb, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([cosine_dist(a[i], b[i]) for i in range(n)]))


def mean_euclidean_distance(aa, ab):
    a, b = subsample(aa, N_SAMPLE), subsample(ab, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in range(n)]))


def compute_rsa(feats, actions):
    common = sorted(set(feats.keys()) & set(actions.keys()))
    if len(common) < 2:
        return None
    pairs = list(itertools.combinations(common, 2))
    rep_dists = [mean_cosine_distance(feats[a], feats[b]) for a, b in pairs]
    act_dists = [mean_euclidean_distance(actions[a], actions[b]) for a, b in pairs]
    rsa, _ = spearmanr(rep_dists, act_dists)
    return float(rsa)


def bootstrap_ci(values, n_boot=N_BOOT, ci=0.95):
    values = np.array(values)
    rng = np.random.RandomState(SEED)
    boot_means = [np.mean(rng.choice(values, len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def bootstrap_delta_ci(vals_a, vals_b, n_boot=N_BOOT, ci=0.95):
    """Bootstrap CI for (mean_a - mean_b) using paired resampling."""
    a, b = np.array(vals_a), np.array(vals_b)
    assert len(a) == len(b)
    rng = np.random.RandomState(SEED + 1)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(len(a), len(a), replace=True)
        deltas.append(np.mean(a[idx]) - np.mean(b[idx]))
    lo = np.percentile(deltas, (1 - ci) / 2 * 100)
    hi = np.percentile(deltas, (1 + ci) / 2 * 100)
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


# ── Loaders ──

def load_actions(task):
    path = DATA_ROOT / f"{task}_demo.hdf5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            k = f"data/demo_{did}"
            if f"demo_{did}" in f["data"]:
                r[did] = f[f"data/demo_{did}/actions"][:].astype(np.float32)
    return r


def load_feats_traj(feat_dir, task, fkey):
    """DINO/CLIP: traj_X naming."""
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            tk = f"traj_{did}"
            if tk in f and fkey in f[tk]:
                r[did] = f[tk][fkey][:].astype(np.float32)
    return r


def load_feats_demo(feat_dir, task, fkey):
    """SigLIP/fused: demo_X naming."""
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            dk = f"demo_{did}"
            if dk in f and fkey in f[dk]:
                r[did] = f[dk][fkey][:].astype(np.float32)
    return r


def load_vla_feats(feat_dir, task, layer, fkey="last_preaction"):
    """VLA: demo_X naming, select layer from (T, 33, 4096)."""
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            dk = f"demo_{did}"
            if dk in f and fkey in f[dk]:
                r[did] = f[dk][fkey][:, layer, :].astype(np.float32)
    return r


# ── Main ──

def run_model(name, loader_fn, loader_kwargs):
    per_task = {}
    rsa_values = []
    demo_ids_used = {}

    for task in TASKS:
        feats = loader_fn(task=task, **loader_kwargs)
        actions = load_actions(task)
        common = sorted(set(feats.keys()) & set(actions.keys()))
        demo_ids_used[task] = [f"demo_{d}" for d in common]
        rsa = compute_rsa(feats, actions)
        per_task[task] = rsa
        if rsa is not None:
            rsa_values.append(rsa)
        tag = f"{rsa:.4f}" if rsa is not None else "SKIP"
        print(f"  {task}: {tag} ({len(common)} demos)")

    if not rsa_values:
        return None
    mean_rsa, ci_lo, ci_hi = bootstrap_ci(rsa_values)
    print(f"  => {name}: {mean_rsa:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    return {
        "mean_rsa": round(mean_rsa, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "per_task": {k: round(v, 4) if v is not None else None for k, v in per_task.items()},
        "rsa_values": [round(v, 4) for v in rsa_values],
        "demo_ids": demo_ids_used,
    }


def main():
    np.random.seed(SEED)

    models_config = [
        ("dino_cls", load_feats_traj,
         {"feat_dir": FEAT_ROOT / "dinov2_libero_goal", "fkey": "cls_token"}),
        ("dino_patch_mean", load_feats_traj,
         {"feat_dir": FEAT_ROOT / "dinov2_libero_goal", "fkey": "patch_mean"}),
        ("clip_cls", load_feats_traj,
         {"feat_dir": FEAT_ROOT / "clip_libero_goal", "fkey": "cls_token"}),
        ("clip_patch_mean", load_feats_traj,
         {"feat_dir": FEAT_ROOT / "clip_libero_goal", "fkey": "patch_mean"}),
        ("siglip_projector", load_feats_demo,
         {"feat_dir": FEAT_ROOT / "siglip_only_libero_goal", "fkey": "last_preaction"}),
        # VLA models for pairwise comparison
        ("vla_trained_L9_lastpre", lambda task, **kw: load_vla_feats(
            FEAT_ROOT / "trained_libero_goal", task, layer=9, fkey="last_preaction"), {}),
        ("vla_untrained_L9_lastpre", lambda task, **kw: load_vla_feats(
            FEAT_ROOT / "untrained_libero_goal", task, layer=9, fkey="last_preaction"), {}),
        ("vla_trained_L9_imgmean", lambda task, **kw: load_vla_feats(
            FEAT_ROOT / "trained_libero_goal", task, layer=9, fkey="image_mean"), {}),
        ("vla_untrained_L9_imgmean", lambda task, **kw: load_vla_feats(
            FEAT_ROOT / "untrained_libero_goal", task, layer=9, fkey="image_mean"), {}),
    ]

    results = {}
    for name, loader, kwargs in models_config:
        print(f"\n--- {name} ---")
        r = run_model(name, loader, kwargs)
        if r is not None:
            results[name] = r

    # Pairwise delta bootstrap CIs
    print("\n=== Pairwise Δ Bootstrap CIs ===")
    delta_pairs = [
        ("vla_trained_L9_lastpre", "vla_untrained_L9_lastpre", "trained_vs_untrained_lastpre"),
        ("vla_trained_L9_imgmean", "vla_untrained_L9_imgmean", "trained_vs_untrained_imgmean"),
        ("siglip_projector", "vla_trained_L9_lastpre", "siglip_vs_vla_trained"),
        ("dino_cls", "siglip_projector", "dino_cls_vs_siglip"),
    ]

    deltas = {}
    for a_key, b_key, label in delta_pairs:
        if a_key in results and b_key in results:
            a_vals = results[a_key]["rsa_values"]
            b_vals = results[b_key]["rsa_values"]
            if len(a_vals) == len(b_vals):
                d_mean, d_lo, d_hi = bootstrap_delta_ci(a_vals, b_vals)
                sig = "**SIG**" if (d_lo > 0 or d_hi < 0) else "n.s."
                print(f"  {label}: Δ={d_mean:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] {sig}")
                deltas[label] = {
                    "delta": round(d_mean, 4),
                    "ci_low": round(d_lo, 4),
                    "ci_high": round(d_hi, 4),
                    "significant": d_lo > 0 or d_hi < 0,
                }

    # Build output
    output = {
        "method": "within-task + euclidean action + cosine rep + N_SAMPLE=20",
        "demos_per_task": 10,
        "demo_ids_standard": [f"demo_{i}" for i in DEMO_IDS],
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "rsa_values"}
                   for k, v in results.items()},
        "pairwise_deltas": deltas,
        "token_info": {
            "vla_image_mean": "Mean of 256 image patch tokens (positions 1:257), excludes BOS and text tokens",
            "vla_last_preaction": "Hidden state at last sequence position (pre-action token)",
            "siglip_projector": "SigLIP features projected to 4096-dim LLM space (siglip_only_libero_goal/last_preaction)",
            "dino": "DINOv2 ViT-L cls_token (1024-dim) or mean of patch tokens (1024-dim)",
            "clip": "CLIP ViT-L/14 cls_token (1024-dim) or mean of patch tokens (1024-dim)",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    # Summary table
    print(f"\n{'Model':<30} {'RSA':>8} {'CI_low':>8} {'CI_high':>8}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:<30} {r['mean_rsa']:>8.4f} {r['ci_low']:>8.4f} {r['ci_high']:>8.4f}")


if __name__ == "__main__":
    main()
