"""TracVLA-Phi3V R2 probe seed expansion: seeds 5-14.

Extends phi3v_r2_action_probe.py (seeds 0-4) to 15 total seeds.
Phase 2 only — features already extracted in phi3v_trained_libero_goal/.
"""
import glob
import json
import os
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = "./data/libero/libero_goal/"
FEATURE_DIR = "./features/phi3v_trained_libero_goal/"
OUT_DIR = "./results/phi3v_r2_seed_expansion/"
OUT_JSON = os.path.join(OUT_DIR, "phi3v_r2_seed_expansion.json")

N_LAYERS = 33
HIDDEN_DIM_MODEL = 3072
PROBE_HIDDEN = 256
ACTION_DIM = 7
SEEDS = list(range(5, 15))
PROBE_BATCH = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DEVICE = "cuda:0"


class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim=HIDDEN_DIM_MODEL, out_dim=ACTION_DIM, hidden_dim=PROBE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_features_and_actions(layer):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            continue
        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            demo_keys = sorted(
                [k for k in ff.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            for k in demo_keys:
                idx = int(k.split("_")[-1])
                feat = np.asarray(ff[k]["last_preaction"][:, layer, :], dtype=np.float32)
                act = np.asarray(fa[f"data/demo_{idx}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val(task_data):
    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list, val_tasks = [], [], []
    for task in sorted(task_data.keys()):
        demos = task_data[task]
        n_train = int(len(demos) * 0.8)
        for i, (feat, act) in enumerate(demos):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            x = torch.from_numpy(feat[:T - 1])
            y = torch.from_numpy(act[1:T])
            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * (T - 1))
    return (torch.cat(X_tr_list), torch.cat(y_tr_list),
            torch.cat(X_val_list), torch.cat(y_val_list), val_tasks)


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-6)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    X_tr_d = X_tr.to(device)
    y_tr_d = y_tr.to(device)
    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    model = ActionMLPProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = len(X_tr_d)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, PROBE_BATCH):
            idx = perm[start:start + PROBE_BATCH]
            pred = model(X_tr_d[idx])
            loss = criterion(pred, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_d)
            val_loss = criterion(val_pred, y_val_d).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_val_d).cpu()
    pred_orig = (pred_norm * y_std + y_mean).numpy()
    y_true = y_val_raw.numpy()

    unique_tasks = sorted(set(val_tasks))
    per_task_r2 = {}
    for t in unique_tasks:
        mask = [i for i, vt in enumerate(val_tasks) if vt == t]
        yt = y_true[mask]
        yp = pred_orig[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean(axis=0)) ** 2)
        per_task_r2[t] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0

    return float(np.mean(list(per_task_r2.values())))


def main():
    device = torch.device(DEVICE)
    t_start = time.time()

    print("=" * 60)
    print("TracVLA-Phi3V R2 Probe — Seed Expansion (5-14)")
    print(f"Seeds: {SEEDS}")
    print(f"Layers: {N_LAYERS} (L0-L32)")
    print(f"Device: {device}")
    print(f"Feature: last_preaction")
    print(f"Split: sequential 80/20 (no shuffle)")
    print(f"R2 space: original (denormalized)")
    print("=" * 60)

    results = {"trained": {}}

    for layer in range(N_LAYERS):
        t0 = time.time()
        task_data = load_features_and_actions(layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)

        seed_r2s = []
        for seed in SEEDS:
            r2 = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, device, seed)
            seed_r2s.append(r2)
            torch.cuda.empty_cache()

        results["trained"][f"L{layer}"] = {
            "seeds": seed_r2s,
            "mean": float(np.mean(seed_r2s)),
            "std": float(np.std(seed_r2s)),
        }

        elapsed = time.time() - t0
        print(f"  L{layer}: mean={np.mean(seed_r2s):.4f} +/- {np.std(seed_r2s):.4f} "
              f"({[round(v, 4) for v in seed_r2s]}) {elapsed:.1f}s")

        del task_data, X_tr, y_tr, X_val, y_val

    output = {
        "model": "TracVLA-Phi3V",
        "checkpoint": "./checkpoints/tracevla-phi3v/",
        "conditions": ["trained"],
        "seeds": SEEDS,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM_MODEL,
        "feature": "last_preaction",
        "split": "sequential_80_20_no_shuffle",
        "r2_space": "original_denormalized",
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {OUT_JSON}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
