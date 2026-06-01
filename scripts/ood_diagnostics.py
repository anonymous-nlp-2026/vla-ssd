"""OOD diagnostics: activation magnitude, cosine similarity, effective rank.
Loads all data once as float16 (~24GB), processes each layer from cache."""
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

LLAMA2BASE_DIR = "./features/llama2base_libero_goal/"
TRAINED_DIR = "./features/trained_libero_goal/"
OUT_JSON = "./results/ood_diagnostics.json"
N_LAYERS = 33
READOUT = "image_mean"
MAX_SAMPLES_PCA = 5000

def load_paired_data():
    """Load all data once, keep as float16 to save memory."""
    l_files = sorted(glob.glob(os.path.join(LLAMA2BASE_DIR, "*.h5")))
    all_l2b = []
    all_tr = []
    for lf in l_files:
        tf = os.path.join(TRAINED_DIR, Path(lf).name)
        if not os.path.exists(tf):
            continue
        fl = h5py.File(lf, "r")
        ft = h5py.File(tf, "r")
        common = sorted(set(fl.keys()) & set(ft.keys()))
        for dk in common:
            dl = fl[dk][READOUT][:]  # (T, 33, 4096) float16
            dt = ft[dk][READOUT][:]
            T = min(dl.shape[0], dt.shape[0])
            all_l2b.append(dl[:T])
            all_tr.append(dt[:T])
        fl.close()
        ft.close()
    L = np.concatenate(all_l2b, axis=0)  # (N, 33, 4096) float16
    T = np.concatenate(all_tr, axis=0)
    return L, T

def main():
    t0 = time.time()
    print("Loading all paired data (float16)...", flush=True)
    L_all, T_all = load_paired_data()
    N = L_all.shape[0]
    print(f"Loaded: {N} samples, shape={L_all.shape}, "
          f"mem={L_all.nbytes/1e9:.1f}+{T_all.nbytes/1e9:.1f} GB, "
          f"{time.time()-t0:.0f}s", flush=True)

    rng = np.random.RandomState(42)
    n_pca = min(N, MAX_SAMPLES_PCA)
    pca_idx = rng.choice(N, n_pca, replace=False) if N > n_pca else np.arange(N)

    results = {}
    for layer in range(N_LAYERS):
        t1 = time.time()
        Lf = L_all[:, layer, :].astype(np.float32)
        Tf = T_all[:, layer, :].astype(np.float32)

        norm_l = float(np.mean(np.linalg.norm(Lf, axis=1)))
        norm_t = float(np.mean(np.linalg.norm(Tf, axis=1)))

        dot = np.sum(Lf * Tf, axis=1)
        nl = np.linalg.norm(Lf, axis=1)
        nt = np.linalg.norm(Tf, axis=1)
        cos = dot / (nl * nt + 1e-8)
        cos_mean = float(np.mean(cos))
        cos_std = float(np.std(cos))

        Ls = Lf[pca_idx]
        Ls = Ls - Ls.mean(axis=0)
        s_l = np.linalg.svd(Ls, compute_uv=False)
        v_l = s_l ** 2
        var_l = float(v_l[:10].sum() / v_l.sum()) if v_l.sum() > 0 else 0.0

        Ts = Tf[pca_idx]
        Ts = Ts - Ts.mean(axis=0)
        s_t = np.linalg.svd(Ts, compute_uv=False)
        v_t = s_t ** 2
        var_t = float(v_t[:10].sum() / v_t.sum()) if v_t.sum() > 0 else 0.0

        results[f"layer_{layer}"] = {
            "activation_norm_llama2base": norm_l,
            "activation_norm_trained": norm_t,
            "cosine_sim_mean": cos_mean,
            "cosine_sim_std": cos_std,
            "variance_explained_top10_llama2base": var_l,
            "variance_explained_top10_trained": var_t,
        }
        print(f"L{layer:2d}: norm_l2b={norm_l:.1f} norm_tr={norm_t:.1f} "
              f"cos={cos_mean:.4f}+-{cos_std:.4f} "
              f"var_l2b={var_l:.4f} var_tr={var_t:.4f} "
              f"({time.time()-t1:.1f}s)", flush=True)
        del Lf, Tf, Ls, Ts

    del L_all, T_all

    output = {"per_layer": results}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_JSON}", flush=True)
    print(f"Total time: {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
