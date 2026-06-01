"""RSA Method Audit: Compare cross-task+phase vs per-task+full-trajectory methods.

Three factors differ between rsa_analysis.py and rsa_siglip_only.py:
  1. RDM scope: cross-task (100x100 for 10 tasks x 10 demos) vs per-task (~10x10, avg'd)
  2. Temporal: phase split (early/mid/late avg) vs full trajectory
  3. Action distance: cosine vs euclidean

Uses N_DEMOS=10 per task (matching original rsa_analysis.py default).
"""
import itertools, json, os, sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

FEAT_ROOT = Path("./features")
DATA_ROOT = Path("./data/libero/libero_goal")
OUT_PATH = Path("./results/rsa/rsa_method_audit.json")

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

TEMPORAL_BINS = {"early": (0.0, 0.33), "mid": (0.33, 0.67), "late": (0.67, 1.0)}
N_SAMPLE = 20
N_DEMOS = 10
DEMO_IDS = list(range(N_DEMOS))

def subsample(arr, n):
    T = arr.shape[0]
    if T <= n: return arr
    return arr[np.linspace(0, T-1, n, dtype=int)]

def get_phase_idx(T, phase):
    lo, hi = TEMPORAL_BINS[phase]
    t = np.linspace(0, 1, T, endpoint=False)
    mask = (t >= lo) & (t < hi) if phase != "late" else (t >= lo) & (t <= 1.0)
    idx = np.where(mask)[0]
    return idx if len(idx) > 0 else np.array([0])

def cos_dist_safe(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 1.0
    return float(cosine_dist(a, b))

def mean_cos_dist(fa, fb):
    a, b = subsample(fa, N_SAMPLE), subsample(fb, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([cos_dist_safe(a[i], b[i]) for i in range(n)]))

def mean_euc_dist(aa, ab):
    a, b = subsample(aa, N_SAMPLE), subsample(ab, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in range(n)]))


def load_feats_traj(feat_dir, task, fkey="cls_token"):
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            k = f"traj_{did}/{fkey}"
            if k.split("/")[0] in f and fkey in f[f"traj_{did}"]:
                r[did] = f[k][:].astype(np.float32)
    return r

def load_feats_demo(feat_dir, task, fkey="last_preaction"):
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            k = f"demo_{did}"
            if k in f and fkey in f[k]:
                r[did] = f[k][fkey][:].astype(np.float32)
    return r

def load_vla_feats(feat_dir, task, layer):
    path = feat_dir / f"{task}.h5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            k = f"demo_{did}"
            if k in f and "last_preaction" in f[k]:
                r[did] = f[k]["last_preaction"][:, layer, :].astype(np.float32)
    return r

def load_actions(task):
    path = DATA_ROOT / f"{task}_demo.hdf5"
    r = {}
    with h5py.File(path, "r") as f:
        for did in DEMO_IDS:
            k = f"data/demo_{did}"
            if f"demo_{did}" in f["data"]:
                r[did] = f[f"data/demo_{did}/actions"][:].astype(np.float32)
    return r

# ── Method B: Per-task, full trajectory, euclidean action ──

def rsa_pertask(feats_pt, acts_pt, act_dist="euclidean"):
    afn = mean_euc_dist if act_dist == "euclidean" else mean_cos_dist
    vals, pt = [], {}
    for task in TASKS:
        feats, acts = feats_pt.get(task, {}), acts_pt.get(task, {})
        common = sorted(set(feats) & set(acts))
        if len(common) < 2: pt[task] = None; continue
        pairs = list(itertools.combinations(common, 2))
        rd = [mean_cos_dist(feats[a], feats[b]) for a, b in pairs]
        ad = [afn(acts[a], acts[b]) for a, b in pairs]
        if np.std(rd) < 1e-10 or np.std(ad) < 1e-10:
            pt[task] = 0.0; vals.append(0.0); continue
        rho, _ = spearmanr(rd, ad)
        pt[task] = round(float(rho), 4); vals.append(float(rho))
    return (round(float(np.mean(vals)), 4), pt) if vals else (None, pt)

# ── Method A: Cross-task, phase split, cosine action ──

def rsa_crosstask_phase(feats_flat, acts_flat, act_dist="cosine"):
    afn = mean_cos_dist if act_dist == "cosine" else mean_euc_dist
    keys = sorted(set(feats_flat) & set(acts_flat))
    N = len(keys)
    if N < 2: return None, {}
    prs = {}
    for phase in TEMPORAL_BINS:
        fc, ac = {}, {}
        for i, k in enumerate(keys):
            fi = get_phase_idx(len(feats_flat[k]), phase)
            ai = get_phase_idx(len(acts_flat[k]), phase)
            fc[i] = feats_flat[k][fi]
            ac[i] = acts_flat[k][ai]
        rdm = np.zeros((N, N), dtype=np.float32)
        adm = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i+1, N):
                rd = mean_cos_dist(fc[i], fc[j])
                ad = afn(ac[i], ac[j])
                rdm[i,j] = rdm[j,i] = rd
                adm[i,j] = adm[j,i] = ad
        tri = np.triu_indices(N, k=1)
        rv, av = rdm[tri], adm[tri]
        if np.std(rv) < 1e-10 or np.std(av) < 1e-10:
            prs[phase] = 0.0
        else:
            rho, _ = spearmanr(rv, av)
            prs[phase] = round(float(rho), 4)
    return round(float(np.mean(list(prs.values()))), 4), prs

# ── Method C: Cross-task, full trajectory ──

def rsa_crosstask_full(feats_flat, acts_flat, act_dist="euclidean"):
    afn = mean_euc_dist if act_dist == "euclidean" else mean_cos_dist
    keys = sorted(set(feats_flat) & set(acts_flat))
    N = len(keys)
    if N < 2: return None
    rdm = np.zeros((N, N), dtype=np.float32)
    adm = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            ki, kj = keys[i], keys[j]
            rd = mean_cos_dist(feats_flat[ki], feats_flat[kj])
            ad = afn(acts_flat[ki], acts_flat[kj])
            rdm[i,j] = rdm[j,i] = rd
            adm[i,j] = adm[j,i] = ad
    tri = np.triu_indices(N, k=1)
    rv, av = rdm[tri], adm[tri]
    if np.std(rv) < 1e-10 or np.std(av) < 1e-10: return 0.0
    rho, _ = spearmanr(rv, av)
    return round(float(rho), 4)


def main():
    np.random.seed(42)
    print(f"Config: {len(TASKS)} tasks x {N_DEMOS} demos = {len(TASKS)*N_DEMOS} trajectories", flush=True)

    models = {
        "dino_cls": ("traj", FEAT_ROOT / "dinov2_libero_goal", "cls_token"),
        "dino_patch_mean": ("traj", FEAT_ROOT / "dinov2_libero_goal", "patch_mean"),
        "clip_cls": ("traj", FEAT_ROOT / "clip_libero_goal", "cls_token"),
        "clip_patch_mean": ("traj", FEAT_ROOT / "clip_libero_goal", "patch_mean"),
        "siglip_projector": ("demo", FEAT_ROOT / "siglip_only_libero_goal", "last_preaction"),
    }

    print("Loading actions...", flush=True)
    acts_pt, acts_flat = {}, {}
    for task in TASKS:
        a = load_actions(task)
        acts_pt[task] = a
        for did, v in a.items(): acts_flat[(task, did)] = v
    print(f"  {len(acts_flat)} demos", flush=True)

    results = {"models": {}}

    for mname, (ltype, fdir, fkey) in models.items():
        print(f"\n--- {mname} ---", flush=True)
        fpt, fflat = {}, {}
        for task in TASKS:
            feats = load_feats_traj(fdir, task, fkey) if ltype == "traj" else load_feats_demo(fdir, task, fkey)
            fpt[task] = feats
            for did, v in feats.items(): fflat[(task, did)] = v
        print(f"  {len(fflat)} feat demos", flush=True)

        # Method A: cross-task + phase + cosine action (rsa_analysis.py)
        print("  Computing Method A (cross-task+phase+cos_act)...", flush=True)
        ra, pa = rsa_crosstask_phase(fflat, acts_flat, "cosine")
        print(f"    A = {ra}  phases={pa}", flush=True)

        # Method B: per-task + full + euclidean action (rsa_siglip_only.py)
        print("  Computing Method B (per-task+full+euc_act)...", flush=True)
        rb, ptb = rsa_pertask(fpt, acts_pt, "euclidean")
        print(f"    B = {rb}", flush=True)

        # Factor isolation
        print("  Factor isolation...", flush=True)
        ra_euc, _ = rsa_crosstask_phase(fflat, acts_flat, "euclidean")
        rb_cos, _ = rsa_pertask(fpt, acts_pt, "cosine")
        rc_euc = rsa_crosstask_full(fflat, acts_flat, "euclidean")
        rc_cos = rsa_crosstask_full(fflat, acts_flat, "cosine")
        print(f"    A(euc_act)={ra_euc} B(cos_act)={rb_cos} C(euc)={rc_euc} C(cos)={rc_cos}", flush=True)

        results["models"][mname] = {
            "method_A": ra, "method_A_phases": pa,
            "method_B": rb, "method_B_per_task": ptb,
            "A_with_euc_act": ra_euc, "B_with_cos_act": rb_cos,
            "C_crosstask_full_euc": rc_euc, "C_crosstask_full_cos": rc_cos,
        }

    # VLA models (layer 9 = typical peak)
    for vname, vdir in [("vla_trained_L9", "trained_libero_goal"), ("vla_untrained_L9", "untrained_libero_goal")]:
        print(f"\n--- {vname} ---", flush=True)
        fpt, fflat = {}, {}
        for task in TASKS:
            feats = load_vla_feats(FEAT_ROOT / vdir, task, layer=9)
            fpt[task] = feats
            for did, v in feats.items(): fflat[(task, did)] = v

        ra, pa = rsa_crosstask_phase(fflat, acts_flat, "cosine")
        rb, ptb = rsa_pertask(fpt, acts_pt, "euclidean")
        ra_euc, _ = rsa_crosstask_phase(fflat, acts_flat, "euclidean")
        rc_euc = rsa_crosstask_full(fflat, acts_flat, "euclidean")
        print(f"  A={ra} B={rb} A(euc)={ra_euc} C(euc)={rc_euc}", flush=True)

        results["models"][vname] = {
            "method_A": ra, "method_A_phases": pa,
            "method_B": rb, "method_B_per_task": ptb,
            "A_with_euc_act": ra_euc, "C_crosstask_full_euc": rc_euc,
        }

    # Summary
    print(f"\n{'='*100}", flush=True)
    print(f"{'Model':<22} {'A(orig)':>10} {'B(orig)':>10} {'A-B':>8} {'A(euc)':>10} {'B(cos)':>10} {'C(euc)':>10} {'C(cos)':>10}", flush=True)
    print("-"*100, flush=True)
    for n, d in results["models"].items():
        a, b = d.get("method_A"), d.get("method_B")
        ae = d.get("A_with_euc_act", "—")
        bc = d.get("B_with_cos_act", "—")
        ce = d.get("C_crosstask_full_euc", "—")
        cc = d.get("C_crosstask_full_cos", "—")
        gap = f"{a-b:+.4f}" if (a is not None and b is not None) else "N/A"
        def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"{n:<22} {fmt(a):>10} {fmt(b):>10} {gap:>8} {fmt(ae):>10} {fmt(bc):>10} {fmt(ce):>10} {fmt(cc):>10}", flush=True)

    results["method_descriptions"] = {
        "method_A": "rsa_analysis.py: cross-task 100x100 RDM + phase split (early/mid/late avg) + cosine action dist",
        "method_B": "rsa_siglip_only.py: per-task RDM + full trajectory + euclidean action dist",
        "A_with_euc_act": "Method A but euclidean action dist (isolates action_dist effect)",
        "B_with_cos_act": "Method B but cosine action dist (isolates action_dist effect)",
        "C_crosstask_full_euc": "cross-task + full trajectory + euclidean (isolates phase effect from A)",
        "C_crosstask_full_cos": "cross-task + full trajectory + cosine (isolates phase effect from A)",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
