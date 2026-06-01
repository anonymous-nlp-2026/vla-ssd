"""Probe Seed Robustness: verify peak layer stability across 5 random seeds.

Trains MLP action probes (input→256→7) on trained_libero_goal/image_mean
for all 33 layers × 5 seeds. Reports per-layer mean±std R² and peak layer consistency.

Hyperparams (plan_011):
  epochs=50, patience=5, lr=1e-3, batch=256, hidden=256
  target: a_{t+1}, z-score norm, 80/20 per-task split
"""

import argparse
import json
import os
import time
import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 42]
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
HIDDEN_DIM = 256
N_LAYERS = 33
INPUT_DIM = 4096
ACTION_DIM = 7

FEATURE_DIR = "./features/trained_libero_goal/"
DATA_DIR = "./data/libero/libero_goal/"
OUT_PATH = "./results/ablations/probe_seed_robustness.json"


class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


def load_data_for_layer(layer):
    """Load features and actions for a single layer across all tasks."""
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
                feat = np.asarray(ff[k]["image_mean"][:, layer, :], dtype=np.float32)
                act = np.asarray(fa[f"data/demo_{idx}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val(task_data):
    """80/20 per-task split, build (x_t, a_{t+1}) pairs."""
    X_tr, y_tr = [], []
    X_val, y_val, val_tasks = [], [], []

    for task in sorted(task_data.keys()):
        demos = task_data[task]
        n_train = int(len(demos) * 0.8)

        for i, (feat, act) in enumerate(demos):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            x = torch.from_numpy(feat[: T - 1])
            y = torch.from_numpy(act[1:T])

            if i < n_train:
                X_tr.append(x)
                y_tr.append(y)
            else:
                X_val.append(x)
                y_val.append(y)
                val_tasks.extend([task] * (T - 1))

    return (torch.cat(X_tr), torch.cat(y_tr),
            torch.cat(X_val), torch.cat(y_val), val_tasks)


def train_probe(X_tr, y_tr, X_val, y_val, val_tasks, seed, device):
    """Train MLP probe with given seed, return mean R² across tasks."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # z-score normalization (fit on train)
    mu = X_tr.mean(dim=0)
    std = X_tr.std(dim=0).clamp(min=1e-8)
    X_tr_n = (X_tr - mu) / std
    X_val_n = (X_val - mu) / std

    X_tr_n = X_tr_n.to(device)
    y_tr_d = y_tr.to(device)
    X_val_n = X_val_n.to(device)
    y_val_d = y_val.to(device)

    model = MLPProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_count = 0
    n_tr = X_tr_n.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            pred = model(X_tr_n[idx])
            loss = loss_fn(pred, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_n)
            val_loss = loss_fn(val_pred, y_val_d).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_pred = val_pred.cpu()
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    # compute per-task R²
    y_val_np = y_val.numpy()
    pred_np = best_pred.numpy()
    tasks_arr = np.array(val_tasks)
    unique_tasks = sorted(set(val_tasks))

    r2_list = []
    for task in unique_tasks:
        mask = tasks_arr == task
        y_t = y_val_np[mask]
        p_t = pred_np[mask]
        ss_res = np.sum((y_t - p_t) ** 2)
        ss_tot = np.sum((y_t - y_t.mean(axis=0)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        r2_list.append(float(r2))

    return float(np.mean(r2_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run seed=0, layer=0 to verify logic")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:1] if args.dry_run else SEEDS
    layers = [0] if args.dry_run else list(range(N_LAYERS))

    print(f"Probe Seed Robustness | device={device} | "
          f"seeds={seeds} | layers={len(layers)}")
    print("=" * 60)

    t_start = time.time()
    per_seed_per_layer = {f"seed_{s}": [] for s in seeds}

    for li, layer in enumerate(layers):
        task_data = load_data_for_layer(layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
        print(f"Layer {layer:2d} | train={X_tr.shape[0]}, val={X_val.shape[0]}", end="")

        for s in seeds:
            r2 = train_probe(X_tr, y_tr, X_val, y_val, val_tasks, s, device)
            per_seed_per_layer[f"seed_{s}"].append(r2)
            print(f" | s{s}={r2:.4f}", end="")
        print()

    # summary
    all_r2 = np.array([per_seed_per_layer[f"seed_{s}"] for s in seeds])  # (n_seeds, n_layers)
    per_layer_mean = all_r2.mean(axis=0).tolist()
    per_layer_std = all_r2.std(axis=0).tolist()
    peak_layer_per_seed = [int(np.argmax(all_r2[i])) for i in range(len(seeds))]
    peak_r2_per_seed = [float(all_r2[i].max()) for i in range(len(seeds))]

    result = {
        "method": "MLP action probe seed robustness",
        "condition": "trained_image_mean",
        "seeds": seeds,
        "hyperparams": {
            "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
            "hidden": HIDDEN_DIM, "batch": BATCH_SIZE
        },
        "per_seed_per_layer": per_seed_per_layer,
        "summary": {
            "per_layer_mean": per_layer_mean,
            "per_layer_std": per_layer_std,
            "peak_layer_per_seed": peak_layer_per_seed,
            "peak_r2_per_seed": peak_r2_per_seed,
            "peak_layer_mean": float(np.mean(peak_layer_per_seed)),
            "peak_layer_std": float(np.std(peak_layer_per_seed)),
            "peak_r2_mean": float(np.mean(peak_r2_per_seed)),
            "peak_r2_std": float(np.std(peak_r2_per_seed)),
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nPeak layers: {peak_layer_per_seed} (mean={np.mean(peak_layer_per_seed):.1f} ± {np.std(peak_layer_per_seed):.1f})")
    print(f"Peak R²: {[f'{x:.4f}' for x in peak_r2_per_seed]} (mean={np.mean(peak_r2_per_seed):.4f} ± {np.std(peak_r2_per_seed):.4f})")
    print(f"Done in {(time.time()-t_start)/60:.1f} min. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
