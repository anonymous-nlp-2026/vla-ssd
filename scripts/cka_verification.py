#!/usr/bin/env python3
"""CKA verification: independent cross-validation of RSA findings."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import json
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

FEAT_BASE = Path("./features")
DATA_DIR = Path("./data/libero/libero_goal")
OUT_DIR = Path("./results/cka")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
N_LAYERS = 33


def subsample(arr, target=N_SAMPLE):
    T = arr.shape[0]
    if T <= target:
        return arr
    return arr[np.linspace(0, T - 1, target, dtype=int)]


def linear_cka(X, Y):
    """Linear CKA: ||Y'X||^2_F / (||X'X||_F * ||Y'Y||_F), centered."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    YtX = Y.T @ X
    num = np.sum(YtX ** 2)
    denom = np.sqrt(np.sum((X.T @ X) ** 2) * np.sum((Y.T @ Y) ** 2))
    return float(num / denom) if denom > 1e-10 else 0.0


def kernel_cka(K, L):
    """CKA between two symmetric matrices (kernel or distance)."""
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic_kl = np.sum(KH * LH)
    hsic_kk = np.sum(KH * KH)
    hsic_ll = np.sum(LH * LH)
    denom = np.sqrt(hsic_kk * hsic_ll)
    return float(hsic_kl / denom) if denom > 1e-10 else 0.0


def load_all_actions():
    actions = {}
    for task in TASKS:
        path = DATA_DIR / f"{task}_demo.hdf5"
        ta = {}
        with h5py.File(path, 'r') as f:
            for key in f['data'].keys():
                if key.startswith('demo_'):
                    did = int(key.split('_')[1])
                    ta[did] = subsample(f[f'data/{key}/actions'][:].astype(np.float32))
        actions[task] = ta
    return actions


def load_vla_all_layers(variant, token_key, actions):
    """Load all layers, subsampled. Returns dict[task][did] = (<=20, 33, 4096) fp16."""
    feat_dir = FEAT_BASE / variant
    result = {}
    for task in TASKS:
        path = feat_dir / f"{task}.h5"
        tf = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                if not key.startswith('demo_'):
                    continue
                did = int(key.split('_')[1])
                if did not in actions.get(task, {}):
                    continue
                raw = f[key][token_key][:]  # (T, 33, 4096) fp16
                tf[did] = subsample(raw)    # keep fp16
        result[task] = tf
        print(f"  {variant}/{token_key} {task}: {len(tf)} demos", flush=True)
    return result


def load_vla_single_layer(variant, token_key, layer, actions):
    feat_dir = FEAT_BASE / variant
    result = {}
    for task in TASKS:
        path = feat_dir / f"{task}.h5"
        tf = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                if not key.startswith('demo_'):
                    continue
                did = int(key.split('_')[1])
                if did not in actions.get(task, {}):
                    continue
                raw = f[key][token_key][:]  # (T, 33, 4096)
                data = subsample(raw)[:, layer, :]
                tf[did] = data.astype(np.float32)
        result[task] = tf
    return result


def load_siglip(actions):
    feat_dir = FEAT_BASE / "siglip_only_libero_goal"
    result = {}
    for task in TASKS:
        path = feat_dir / f"{task}.h5"
        tf = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                if not key.startswith('demo_'):
                    continue
                did = int(key.split('_')[1])
                if did not in actions.get(task, {}):
                    continue
                tf[did] = subsample(f[key]['last_preaction'][:].astype(np.float32))
        result[task] = tf
    return result


def load_dino(actions):
    feat_dir = FEAT_BASE / "dinov2_libero_goal"
    result = {}
    for task in TASKS:
        path = feat_dir / f"{task}.h5"
        tf = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                if not key.startswith('traj_'):
                    continue
                did = int(key.split('_')[1])
                if did not in actions.get(task, {}):
                    continue
                tf[did] = subsample(f[key]['cls_token'][:].astype(np.float32))
        result[task] = tf
    return result


def flatten(feats, actions):
    """Stack all frames into (N, D_feat) and (N, 7)."""
    Xs, Ys = [], []
    for task in TASKS:
        tf, ta = feats.get(task, {}), actions.get(task, {})
        for did in sorted(set(tf) & set(ta)):
            f, a = tf[did], ta[did]
            n = min(len(f), len(a))
            Xs.append(f[:n])
            Ys.append(a[:n])
    return np.concatenate(Xs), np.concatenate(Ys)


def rdm_cka_pertask(feats, actions):
    """Per-task kernel CKA between demo-mean feature RDM and action RDM."""
    ckas = []
    for task in TASKS:
        tf, ta = feats.get(task, {}), actions.get(task, {})
        common = sorted(set(tf) & set(ta))
        if len(common) < 3:
            continue
        feat_means = np.stack([tf[d][:min(len(tf[d]), len(ta[d]))].mean(0) for d in common])
        act_means = np.stack([ta[d][:min(len(tf[d]), len(ta[d]))].mean(0) for d in common])
        feat_rdm = squareform(pdist(feat_means, 'cosine'))
        act_rdm = squareform(pdist(act_means, 'euclidean'))
        ckas.append(kernel_cka(feat_rdm, act_rdm))
    return float(np.mean(ckas)) if ckas else 0.0


def extract_layer(all_feats, layer):
    """Slice one layer from all-layer features."""
    out = {}
    for task, demos in all_feats.items():
        out[task] = {did: d[:, layer, :].astype(np.float32) for did, d in demos.items()}
    return out


def both_cka(feats, actions, label=""):
    t0 = time.time()
    X, Y = flatten(feats, actions)
    cka_act = linear_cka(X, Y)
    t1 = time.time()
    cka_rdm = rdm_cka_pertask(feats, actions)
    t2 = time.time()
    print(f"  {label}: act={cka_act:.4f} rdm={cka_rdm:.4f} ({t1-t0:.1f}s/{t2-t1:.1f}s)", flush=True)
    return cka_act, cka_rdm


def main():
    np.random.seed(42)
    t_start = time.time()

    print("Loading actions...", flush=True)
    actions = load_all_actions()
    total = sum(len(v) for v in actions.values())
    print(f"  {total} demos across {len(TASKS)} tasks", flush=True)

    # 1. SigLIP projector
    print("\n[1/5] SigLIP projector", flush=True)
    feats = load_siglip(actions)
    proj_act, proj_rdm = both_cka(feats, actions, "projector")
    del feats

    # 2. DINO
    print("\n[2/5] DINO", flush=True)
    feats = load_dino(actions)
    dino_act, dino_rdm = both_cka(feats, actions, "dino")
    del feats

    # 3. VLA trained image_mean (all layers)
    print("\n[3/5] VLA trained image_mean — per-layer sweep", flush=True)
    trained_im = load_vla_all_layers("trained_libero_goal", "image_mean", actions)

    pl_act, pl_rdm = [], []
    for layer in range(N_LAYERS):
        lf = extract_layer(trained_im, layer)
        X, Y = flatten(lf, actions)
        ca = linear_cka(X, Y)
        cr = rdm_cka_pertask(lf, actions)
        pl_act.append(round(ca, 4))
        pl_rdm.append(round(cr, 4))
        if layer % 8 == 0 or layer == 23 or layer == 32:
            print(f"    L{layer:2d}: act={ca:.4f} rdm={cr:.4f}", flush=True)
        del lf

    tr_im_act, tr_im_rdm = pl_act[23], pl_rdm[23]
    del trained_im

    # 4. VLA untrained image_mean L23
    print("\n[4/5] VLA untrained image_mean L23", flush=True)
    feats = load_vla_single_layer("untrained_libero_goal", "image_mean", 23, actions)
    un_im_act, un_im_rdm = both_cka(feats, actions, "untrained_im_L23")
    del feats

    # 5. VLA trained last_preaction (all layers)
    print("\n[5/5] VLA trained last_preaction — per-layer sweep", flush=True)
    trained_lp = load_vla_all_layers("trained_libero_goal", "last_preaction", actions)

    pl_lp_act = []
    best_layer, best_act = -1, -1
    for layer in range(N_LAYERS):
        lf = extract_layer(trained_lp, layer)
        X, Y = flatten(lf, actions)
        ca = linear_cka(X, Y)
        pl_lp_act.append(round(ca, 4))
        if ca > best_act:
            best_act, best_layer = ca, layer
        del lf

    best_lf = extract_layer(trained_lp, best_layer)
    best_rdm = rdm_cka_pertask(best_lf, actions)
    del best_lf, trained_lp
    print(f"  best: L{best_layer} act={best_act:.4f} rdm={best_rdm:.4f}", flush=True)

    # === Assemble results ===
    vals = {
        "projector": proj_act,
        "trained_image_mean_L23": tr_im_act,
        "untrained_image_mean_L23": un_im_act,
        "dino": dino_act,
    }
    ranking = sorted(vals.items(), key=lambda x: -x[1])
    rank_str = " > ".join(f"{k}({v:.4f})" for k, v in ranking)
    rsa_order = ["projector", "trained_image_mean_L23", "untrained_image_mean_L23", "dino"]
    cka_order = [k for k, _ in ranking]

    results = {
        "cka_vs_actions": {
            "projector": round(proj_act, 4),
            "trained_image_mean_L23": round(tr_im_act, 4),
            "untrained_image_mean_L23": round(un_im_act, 4),
            "dino": round(dino_act, 4),
            "trained_last_preaction_best": round(best_act, 4),
            "trained_last_preaction_best_layer": best_layer,
        },
        "cka_vs_action_rdm": {
            "projector": round(proj_rdm, 4),
            "trained_image_mean_L23": round(tr_im_rdm, 4),
            "untrained_image_mean_L23": round(un_im_rdm, 4),
            "dino": round(dino_rdm, 4),
            "trained_last_preaction_best": round(best_rdm, 4),
        },
        "perlayer_cka_trained_image_mean": pl_act,
        "perlayer_cka_rdm_trained_image_mean": pl_rdm,
        "perlayer_cka_trained_last_preaction": pl_lp_act,
        "comparison_with_rsa": {
            "rsa_ranking": "projector(0.6153) > trained(0.5014) > untrained(0.4847) > dino(0.3880)",
            "cka_vs_actions_ranking": rank_str,
            "consistent": rsa_order == cka_order,
        },
    }

    out_path = OUT_DIR / "cka_verification.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n=== Done ({elapsed:.0f}s) ===", flush=True)
    print(f"CKA ranking: {rank_str}", flush=True)
    print(f"RSA ranking: projector > trained > untrained > dino", flush=True)
    print(f"Consistent: {rsa_order == cka_order}", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
