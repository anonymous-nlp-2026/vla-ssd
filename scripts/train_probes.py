"""train_probes.py — Train probes (linear/MLP) on VLA/DINO-V2 hidden states.

Input:  features/{mode}/{task}.h5 — per-task HDF5 with per-trajectory hidden states
Output: results/{mode}/{probe_type}/ — per-layer JSON + summary CSV

Usage:
  python train_probes.py --features_dir features/trained/ --output_dir results/trained/ --probe_type temporal_distance
  python train_probes.py --features_dir features/trained/ --output_dir results/trained/ --probe_type oracle
  python train_probes.py --features_dir features/trained/ --output_dir results/trained/ --probe_type subtask_predicate --label_file labels.json
"""

import argparse
import csv
import json
import os
import glob
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             accuracy_score, f1_score, confusion_matrix)
from sklearn.decomposition import PCA as SklearnPCA

# OpenVLA: last_preaction / image_mean  shape (T, 33, 4096)
# DINO-V2: cls_token / patch_mean       shape (T, 1024)
AGG_MAP = {
    "last_preaction": ("last_preaction", "cls_token"),
    "image_mean": ("image_mean", "patch_mean"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train probes (linear/MLP) on VLA/DINO-V2 features")
    p.add_argument("--features_dir", required=True,
                   help="Directory with per-task HDF5 feature files")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save probe results")
    p.add_argument("--probe_type", required=True,
                   choices=["temporal_distance", "subtask_predicate", "oracle", "action_prediction", "action_delta", "goal_classification"])
    p.add_argument("--layers", default=None,
                   help="Layer range '0-32' or list '0,5,10'. Default: all")
    p.add_argument("--aggregation", default="last_preaction",
                   choices=["last_preaction", "image_mean"])
    p.add_argument("--label_file", default=None,
                   help="Predicate labels JSON (required for subtask_predicate)")
    p.add_argument("--data_dir", default="./data/libero/libero_10/",
                   help="LIBERO raw data directory (for action_prediction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--pairs_per_bin", type=int, default=50)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--probe_model", default="mlp",
                   choices=["linear", "mlp"],
                   help="Probe architecture: linear or mlp (default: mlp)")
    p.add_argument("--shuffle_cross_trajectory", action="store_true",
                   help="Control B: pair frames from different trajectories")
    p.add_argument("--pca_dim", type=int, default=None,
                   help="PCA reduce features to N dims before probe (fit on train only)")
    p.add_argument("--shuffle_labels", action="store_true",
                   help="Negative control: permute distance labels (permutation test null)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Use only first N frames per trajectory (early-frame analysis)")
    return p.parse_args()


# ============================================================
# Feature scanning & loading
# ============================================================

def scan_features(features_dir, aggregation):
    """Scan HDF5 files — return per-traj metadata, n_layers, hidden_dim."""
    primary, fallback = AGG_MAP[aggregation]
    traj_metas = []
    n_layers = hidden_dim = None

    for h5_path in sorted(glob.glob(os.path.join(features_dir, "*.h5"))):
        task = Path(h5_path).stem
        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            for hk in keys:
                if primary in f[hk]:
                    fk = primary
                elif fallback in f[hk]:
                    fk = fallback
                else:
                    raise KeyError(f"No feature key in {h5_path}/{hk}")

                shape = f[hk][fk].shape
                is_3d = len(shape) == 3
                T = shape[0]
                if n_layers is None:
                    n_layers = shape[1] if is_3d else 1
                    hidden_dim = shape[-1]

                traj_metas.append(dict(
                    task=task, h5_path=h5_path, h5_key=hk,
                    feat_key=fk, T=T, is_3d=is_3d,
                ))

    if not traj_metas:
        raise FileNotFoundError(f"No .h5 files found in {features_dir}")
    return traj_metas, n_layers, hidden_dim


def split_by_task(traj_metas):
    """80/20 split per task (first 80 % train, rest val)."""
    by_task = defaultdict(list)
    for i, m in enumerate(traj_metas):
        by_task[m["task"]].append(i)

    train_ids, val_ids = [], []
    for task in sorted(by_task):
        ids = by_task[task]
        n_train = int(len(ids) * 0.8)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:])
    return train_ids, val_ids


def load_layer_features(traj_metas, indices, layer):
    """Load one layer's features for selected trajectories.

    Returns list of (T, D) float32 arrays, aligned with `indices`.
    """
    by_file = defaultdict(list)
    for pos, idx in enumerate(indices):
        m = traj_metas[idx]
        by_file[m["h5_path"]].append((pos, m))

    feats = [None] * len(indices)
    for h5_path, entries in by_file.items():
        with h5py.File(h5_path, "r") as f:
            for pos, m in entries:
                ds = f[m["h5_key"]][m["feat_key"]]
                if m["is_3d"]:
                    data = ds[:, layer, :]
                else:
                    data = ds[:]
                feats[pos] = np.asarray(data, dtype=np.float32)
    return feats


def parse_layers(layers_str, max_layer):
    if layers_str is None:
        return list(range(max_layer + 1))
    if "-" in layers_str and "," not in layers_str:
        lo, hi = layers_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in layers_str.split(",")]


def apply_pca(X_tr, X_val, pca_dim):
    """Fit PCA on train set, transform both. Returns transformed tensors + variance info."""
    pca = SklearnPCA(n_components=pca_dim)
    X_tr_np = X_tr.numpy() if isinstance(X_tr, torch.Tensor) else X_tr
    X_val_np = X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val
    X_tr_pca = pca.fit_transform(X_tr_np)
    X_val_pca = pca.transform(X_val_np)
    evr = pca.explained_variance_ratio_
    info = {
        "cumulative_variance_ratio": float(evr.sum()),
        "top_10_components": [float(x) for x in evr[:10]],
    }
    return (torch.from_numpy(X_tr_pca.astype(np.float32)),
            torch.from_numpy(X_val_pca.astype(np.float32)),
            info)

# ============================================================
# Distance-stratified pair sampling
# ============================================================

def sample_pairs(traj_metas, indices, n_bins, pairs_per_bin, seed):
    """Within-trajectory distance-stratified pair sampling.

    Returns list of (local_pos, t_i, t_j, norm_dist).
    local_pos indexes into `indices`.
    """
    rng = np.random.RandomState(seed)
    all_pairs = []

    for pos, idx in enumerate(indices):
        T = traj_metas[idx]["T"]
        if T < 2:
            continue
        ii, jj = np.triu_indices(T, k=1)
        dists = (jj - ii).astype(np.float32) / T

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[-1] += 1e-6

        for b in range(n_bins):
            mask = (dists >= edges[b]) & (dists < edges[b + 1])
            valid = np.where(mask)[0]
            if len(valid) == 0:
                continue
            n = min(pairs_per_bin, len(valid))
            chosen = rng.choice(valid, size=n, replace=False)
            for c in chosen:
                all_pairs.append((pos, int(ii[c]), int(jj[c]), float(dists[c])))

    return all_pairs


def build_pair_tensors(feats, pairs):
    """Build |h(t_j) - h(t_i)| inputs and distance targets."""
    N = len(pairs)
    D = feats[0].shape[-1]
    X = np.empty((N, D), dtype=np.float32)
    y = np.empty(N, dtype=np.float32)
    for n, (pos, ti, tj, dist) in enumerate(pairs):
        X[n] = np.abs(feats[pos][tj] - feats[pos][ti])
        y[n] = dist
    return torch.from_numpy(X), torch.from_numpy(y)


def build_oracle_tensors(pairs):
    """Oracle: input = scalar distance (same as target)."""
    X = torch.tensor([[p[3]] for p in pairs], dtype=torch.float32)
    y = torch.tensor([p[3] for p in pairs], dtype=torch.float32)
    return X, y


def sample_cross_trajectory_pairs(traj_metas, indices, n_bins, pairs_per_bin, seed):
    """Cross-trajectory pair sampling (Control B negative control)."""
    rng = np.random.RandomState(seed)
    valid = [(pos, idx) for pos, idx in enumerate(indices)
             if traj_metas[idx]["T"] >= 2]
    if len(valid) < 2:
        return []

    n_v = len(valid)
    target_total = n_v * n_bins * pairs_per_bin
    n_cand = target_total * 5

    Ts = np.array([traj_metas[valid[i][1]]["T"] for i in range(n_v)])
    poss = np.array([valid[i][0] for i in range(n_v)])

    a_idx = rng.randint(n_v, size=n_cand)
    offsets = rng.randint(1, n_v, size=n_cand)
    b_idx = (a_idx + offsets) % n_v

    t_a = (rng.random(n_cand) * Ts[a_idx]).astype(int)
    t_b = (rng.random(n_cand) * Ts[b_idx]).astype(int)
    norm_dist = np.abs(t_b / Ts[b_idx] - t_a / Ts[a_idx])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-6

    all_pairs = []
    per_bin = target_total // n_bins
    for bi in range(n_bins):
        mask = np.where((norm_dist >= edges[bi]) & (norm_dist < edges[bi + 1]))[0]
        if len(mask) == 0:
            continue
        n = min(per_bin, len(mask))
        chosen = rng.choice(mask, size=n, replace=False)
        for c in chosen:
            all_pairs.append((
                int(poss[a_idx[c]]), int(t_a[c]),
                int(poss[b_idx[c]]), int(t_b[c]),
                float(norm_dist[c]),
            ))

    return all_pairs


def build_cross_pair_tensors(feats, pairs):
    """Build |h_j(t_j) - h_i(t_i)| for cross-trajectory pairs."""
    N = len(pairs)
    D = feats[0].shape[-1]
    X = np.empty((N, D), dtype=np.float32)
    y = np.empty(N, dtype=np.float32)
    for n, (pos_i, ti, pos_j, tj, dist) in enumerate(pairs):
        X[n] = np.abs(feats[pos_j][tj] - feats[pos_i][ti])
        y[n] = dist
    return torch.from_numpy(X), torch.from_numpy(y)


# ============================================================
# Probes
# ============================================================

class LinearProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)




class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MultiLabelProbe(nn.Module):
    def __init__(self, in_dim, n_labels):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_labels)

    def forward(self, x):
        return self.fc(x)

class GoalClassificationProbe(nn.Module):
    """MLP probe for multi-class goal classification."""
    def __init__(self, in_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)




def train_regression_probe(X_tr, y_tr, X_val, y_val, in_dim, args, device):
    probe_model = getattr(args, "probe_model", "mlp")
    if probe_model == "mlp":
        model = MLPProbe(in_dim).to(device)
    else:
        model = LinearProbe(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for s in range(0, len(X_tr), args.batch_size):
            idx = perm[s : s + args.batch_size]
            loss = crit(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_val), y_val).item()

        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                final_ep = ep + 1
                break
        final_ep = ep + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_val).cpu().numpy()
    y_np = y_val.cpu().numpy()

    rho, pval = spearmanr(pred, y_np)
    return dict(
        spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
        spearman_p=float(pval) if not np.isnan(pval) else 1.0,
        r2=float(r2_score(y_np, pred)),
        mae=float(mean_absolute_error(y_np, pred)),
        val_loss=best_loss,
        epochs_trained=final_ep,
    ), pred, y_np


def train_classification_probe(X_tr, y_tr, X_val, y_val, in_dim, n_labels, args, device):
    model = MultiLabelProbe(in_dim, n_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    bs = 512
    best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for s in range(0, len(X_tr), bs):
            idx = perm[s : s + bs]
            loss = crit(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_val), y_val).item()

        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                final_ep = ep + 1
                break
        final_ep = ep + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_val)).cpu().numpy()
    y_np = y_val.cpu().numpy()

    auc_per = {}
    for k in range(n_labels):
        col = y_np[:, k]
        if col.sum() > 0 and col.sum() < len(col):
            auc_per[k] = float(roc_auc_score(col, scores[:, k]))
        else:
            auc_per[k] = float("nan")

    valid = [v for v in auc_per.values() if not np.isnan(v)]
    return dict(
        auc_per_predicate={str(k): v for k, v in auc_per.items()},
        macro_auc=float(np.mean(valid)) if valid else float("nan"),
        val_loss=best_loss,
        epochs_trained=final_ep,
    )




def train_goal_classification_probe(X_tr, y_tr, X_val, y_val, in_dim, n_classes, args, device):
    """Train a multi-class classification probe with CrossEntropyLoss."""
    probe_model = getattr(args, "probe_model", "mlp")
    if probe_model == "linear":
        model = nn.Linear(in_dim, n_classes).to(device)
    else:
        model = GoalClassificationProbe(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    best_acc, best_state, wait, final_ep = 0.0, None, 0, 0

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for s in range(0, len(X_tr), args.batch_size):
            idx = perm[s : s + args.batch_size]
            loss = crit(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            preds = logits.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                final_ep = ep + 1
                break
        final_ep = ep + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_val).cpu()
        preds = logits.argmax(dim=1).numpy()
    y_np = y_val.cpu().numpy()

    acc = float(accuracy_score(y_np, preds))
    macro_f1 = float(f1_score(y_np, preds, average="macro"))
    cm = confusion_matrix(y_np, preds, labels=list(range(n_classes)))
    per_class_acc = {}
    for c in range(n_classes):
        mask = y_np == c
        if mask.sum() > 0:
            per_class_acc[c] = float((preds[mask] == c).mean())
        else:
            per_class_acc[c] = float("nan")

    return dict(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class_accuracy={str(k): v for k, v in per_class_acc.items()},
        confusion_matrix=cm.tolist(),
        epochs_trained=final_ep,
        probe_model=probe_model,
        val_preds=preds,
    )


# ============================================================
# Runners
# ============================================================


# ============================================================
# Action prediction
# ============================================================

def load_actions(traj_metas, indices, data_dir):
    """Load action arrays from LIBERO HDF5 for given trajectory indices."""
    actions = [None] * len(indices)
    by_task = defaultdict(list)
    for pos, idx in enumerate(indices):
        by_task[traj_metas[idx]["task"]].append((pos, idx))

    for task, entries in by_task.items():
        libero_path = os.path.join(data_dir, f"{task}_demo.hdf5")
        with h5py.File(libero_path, "r") as f:
            for pos, idx in entries:
                m = traj_metas[idx]
                demo_idx = int(m["h5_key"].split("_")[-1])
                demo_key = f"data/demo_{demo_idx}/actions"
                actions[pos] = np.asarray(f[demo_key], dtype=np.float32)
    return actions


def run_action_prediction(args, device):
    traj_metas, n_layers, hidden_dim = scan_features(args.features_dir, args.aggregation)
    layers = parse_layers(args.layers, n_layers - 1)
    train_ids, val_ids = split_by_task(traj_metas)

    print(f"Features: {len(traj_metas)} trajs, {n_layers} layers, dim={hidden_dim}")
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    predict_delta = args.probe_type == "action_delta"
    train_actions = load_actions(traj_metas, train_ids, args.data_dir)
    val_actions = load_actions(traj_metas, val_ids, args.data_dir)

    def build_shifted(feats, acts, ids_list):
        Xs, Ys, task_ids = [], [], []
        for i, (feat, act) in enumerate(zip(feats, acts)):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            Xs.append(torch.from_numpy(feat[:T - 1]))
            if predict_delta:
                Ys.append(torch.from_numpy(act[1:T] - act[:T-1]))
            else:
                Ys.append(torch.from_numpy(act[1:T]))
            task_ids.extend([traj_metas[ids_list[i]]["task"]] * (T - 1))
        return torch.cat(Xs), torch.cat(Ys), task_ids

    results = {}
    for li, layer in enumerate(layers):
        print(f"\n[Layer {layer}] ({li+1}/{len(layers)})", flush=True)
        t0 = time.time()

        train_feats = load_layer_features(traj_metas, train_ids, layer)
        val_feats = load_layer_features(traj_metas, val_ids, layer)

        X_tr, y_tr_raw, _ = build_shifted(train_feats, train_actions, train_ids)
        X_val, y_val_raw, val_tasks = build_shifted(val_feats, val_actions, val_ids)
        del train_feats, val_feats

        probe_dim = hidden_dim
        pca_info = None
        if args.pca_dim is not None:
            X_tr, X_val, pca_info = apply_pca(X_tr, X_val, args.pca_dim)
            probe_dim = args.pca_dim
            print(f"  PCA {hidden_dim} -> {args.pca_dim}: "
                  f"{pca_info['cumulative_variance_ratio']:.4f} variance retained")

        y_mean = y_tr_raw.mean(dim=0)
        y_std = y_tr_raw.std(dim=0).clamp(min=1e-8)
        y_tr = (y_tr_raw - y_mean) / y_std
        y_val = (y_val_raw - y_mean) / y_std

        print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")

        model = ActionMLPProbe(probe_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = nn.MSELoss()

        X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
        X_val_d, y_val_d = X_val.to(device), y_val.to(device)

        best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(X_tr_d), device=device)
            for s in range(0, len(X_tr_d), args.batch_size):
                idx = perm[s : s + args.batch_size]
                loss = crit(model(X_tr_d[idx]), y_tr_d[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = crit(model(X_val_d), y_val_d).item()

            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    final_ep = ep + 1
                    break
            final_ep = ep + 1

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_std = model(X_val_d).cpu()

        pred_raw = pred_std * y_std + y_mean
        y_val_np = y_val_raw.numpy()
        pred_np = pred_raw.numpy()

        dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        per_dim_r2 = {}
        for d in range(7):
            per_dim_r2[dim_names[d]] = float(r2_score(y_val_np[:, d], pred_np[:, d]))

        overall_r2 = float(r2_score(y_val_np.flatten(), pred_np.flatten()))
        raw_mse = float(np.mean((y_val_np - pred_np) ** 2))

        grip_pred = (pred_np[:, 6] > 0).astype(float)
        grip_true = (y_val_np[:, 6] > 0).astype(float)
        grip_acc = float(np.mean(grip_pred == grip_true))

        grip_delta_acc = None
        if predict_delta:
            grip_delta_pred = np.round(pred_np[:, 6]).clip(-1, 1).astype(int)
            grip_delta_true = np.round(y_val_np[:, 6]).clip(-1, 1).astype(int)
            grip_delta_acc = float(np.mean(grip_delta_pred == grip_delta_true))

        val_tasks_arr = np.array(val_tasks)
        unique_tasks = sorted(set(val_tasks))
        per_task_r2 = {}
        for task in unique_tasks:
            mask = val_tasks_arr == task
            if mask.sum() >= 2:
                per_task_r2[task] = float(r2_score(
                    y_val_np[mask].flatten(), pred_np[mask].flatten()))
            else:
                per_task_r2[task] = float("nan")

        valid_r2s = [v for v in per_task_r2.values() if not np.isnan(v)]
        per_task_mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0

        m = dict(
            layer=layer,
            per_dim_r2=per_dim_r2,
            overall_r2=overall_r2,
            per_task_r2=per_task_r2,
            per_task_mean_r2=per_task_mean_r2,
            gripper_accuracy=grip_acc,
            raw_mse=raw_mse,
            val_loss=best_loss,
            epochs_trained=final_ep,
        )
        if grip_delta_acc is not None:
            m["gripper_delta_accuracy"] = grip_delta_acc
        if pca_info is not None:
            m["pca_dim"] = args.pca_dim
            m["pca_info"] = pca_info
        results[layer] = m
        del X_tr, y_tr, X_val, y_val, X_tr_d, y_tr_d, X_val_d, y_val_d

        print(f"  R2={overall_r2:.4f}  mean_task_R2={per_task_mean_r2:.4f}  "
              f"grip_acc={grip_acc:.4f}  ep={final_ep}  {time.time()-t0:.1f}s")

    return results


def run_temporal_distance(args, device):
    traj_metas, n_layers, hidden_dim = scan_features(args.features_dir, args.aggregation)
    layers = parse_layers(args.layers, n_layers - 1)
    train_ids, val_ids = split_by_task(traj_metas)

    print(f"Features: {len(traj_metas)} trajs, {n_layers} layers, dim={hidden_dim}")
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    cross = getattr(args, "shuffle_cross_trajectory", False)
    shuffle_labels = getattr(args, "shuffle_labels", False)
    if cross:
        print("MODE: cross-trajectory shuffle (Control B)")
        train_pairs = sample_cross_trajectory_pairs(
            traj_metas, train_ids, args.n_bins, args.pairs_per_bin, args.seed)
        val_pairs = sample_cross_trajectory_pairs(
            traj_metas, val_ids, args.n_bins, args.pairs_per_bin, args.seed + 1)
    else:
        train_pairs = sample_pairs(traj_metas, train_ids, args.n_bins, args.pairs_per_bin, args.seed)
        val_pairs = sample_pairs(traj_metas, val_ids, args.n_bins, args.pairs_per_bin, args.seed + 1)
    if shuffle_labels:
        print("MODE: label permutation (negative control)")
    print(f"Pairs: {len(train_pairs)} train / {len(val_pairs)} val")

    # Build per-task grouping for val pairs
    val_pair_tasks = []
    if not cross:
        for pos, ti, tj, dist in val_pairs:
            val_pair_tasks.append(traj_metas[val_ids[pos]]["task"])
    else:
        for pos_i, ti, pos_j, tj, dist in val_pairs:
            val_pair_tasks.append(traj_metas[val_ids[pos_i]]["task"])
    val_pair_tasks = np.array(val_pair_tasks)
    unique_tasks = sorted(set(val_pair_tasks))

    pair_builder = build_cross_pair_tensors if cross else build_pair_tensors
    results = {}
    for li, layer in enumerate(layers):
        print(f"\n[Layer {layer}] ({li+1}/{len(layers)})", flush=True)
        t0 = time.time()

        train_feats = load_layer_features(traj_metas, train_ids, layer)
        X_tr, y_tr = pair_builder(train_feats, train_pairs)
        del train_feats

        val_feats = load_layer_features(traj_metas, val_ids, layer)
        X_val, y_val = pair_builder(val_feats, val_pairs)
        del val_feats

        if shuffle_labels:
            y_tr = y_tr[torch.randperm(len(y_tr))]
            y_val = y_val[torch.randperm(len(y_val))]

        td_probe_dim = hidden_dim
        td_pca_info = None
        if args.pca_dim is not None:
            X_tr, X_val, td_pca_info = apply_pca(X_tr, X_val, args.pca_dim)
            td_probe_dim = args.pca_dim
            if li == 0:
                print(f"  PCA {hidden_dim} -> {args.pca_dim}: "
                      f"{td_pca_info['cumulative_variance_ratio']:.4f} variance retained")

        m, pred, y_true = train_regression_probe(X_tr, y_tr, X_val, y_val, td_probe_dim, args, device)
        m["layer"] = layer
        if td_pca_info is not None:
            m["pca_dim"] = args.pca_dim
            m["pca_info"] = td_pca_info
        if cross:
            m["cross_trajectory"] = True

        # Per-task Spearman rho
        per_task_rho = {}
        per_task_n_pairs = {}
        for task in unique_tasks:
            mask = val_pair_tasks == task
            n_t = int(mask.sum())
            per_task_n_pairs[task] = n_t
            if n_t >= 3:
                rho_t, _ = spearmanr(pred[mask], y_true[mask])
                per_task_rho[task] = float(rho_t) if not np.isnan(rho_t) else 0.0
            else:
                per_task_rho[task] = float("nan")
        m["per_task_rho"] = per_task_rho
        m["per_task_n_pairs"] = per_task_n_pairs
        valid_rhos = [v for v in per_task_rho.values() if not np.isnan(v)]
        m["per_task_mean_rho"] = float(np.mean(valid_rhos)) if valid_rhos else 0.0

        results[layer] = m
        del X_tr, y_tr, X_val, y_val

        print(f"  rho={m['spearman_rho']:.4f}  mean_task_rho={m['per_task_mean_rho']:.4f}  "
              f"R2={m['r2']:.4f}  ep={m['epochs_trained']}  {time.time()-t0:.1f}s")

    return results


def run_oracle(args, device):
    traj_metas, _, _ = scan_features(args.features_dir, args.aggregation)
    train_ids, val_ids = split_by_task(traj_metas)

    train_pairs = sample_pairs(traj_metas, train_ids, args.n_bins, args.pairs_per_bin, args.seed)
    val_pairs = sample_pairs(traj_metas, val_ids, args.n_bins, args.pairs_per_bin, args.seed + 1)
    print(f"Oracle: {len(train_pairs)} train / {len(val_pairs)} val pairs")

    X_tr, y_tr = build_oracle_tensors(train_pairs)
    X_val, y_val = build_oracle_tensors(val_pairs)

    m, _, _ = train_regression_probe(X_tr, y_tr, X_val, y_val, 1, args, device)
    m["layer"] = "oracle"
    print(f"Oracle: rho={m['spearman_rho']:.4f}  R2={m['r2']:.4f}  MAE={m['mae']:.4f}")
    return {"oracle": m}


def run_subtask_predicate(args, device):
    if not args.label_file:
        raise ValueError("--label_file required for subtask_predicate")

    with open(args.label_file) as f:
        label_data = json.load(f)
    pred_names = label_data["predicate_names"]
    labels = label_data["labels"]
    n_pred = len(pred_names)
    print(f"Predicates ({n_pred}): {pred_names}")

    traj_metas, n_layers, hidden_dim = scan_features(args.features_dir, args.aggregation)
    layers = parse_layers(args.layers, n_layers - 1)
    train_ids, val_ids = split_by_task(traj_metas)

    results = {}
    for li, layer in enumerate(layers):
        print(f"\n[Layer {layer}] ({li+1}/{len(layers)})", flush=True)
        t0 = time.time()

        def gather(ids):
            Xs, Ys = [], []
            feats = load_layer_features(traj_metas, ids, layer)
            for pos, idx in enumerate(ids):
                m = traj_metas[idx]
                tk = m["h5_key"]
                if m["task"] not in labels or tk not in labels[m["task"]]:
                    continue
                lab = np.array(labels[m["task"]][tk], dtype=np.float32)
                T = min(feats[pos].shape[0], lab.shape[0])
                Xs.append(torch.from_numpy(feats[pos][:T]))
                Ys.append(torch.from_numpy(lab[:T]))
            if not Xs:
                return None, None
            return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)

        X_tr, y_tr = gather(train_ids)
        X_val, y_val = gather(val_ids)

        if X_tr is None or X_val is None:
            print("  No valid data, skipping")
            continue

        sp_probe_dim = hidden_dim
        sp_pca_info = None
        if args.pca_dim is not None:
            X_tr, X_val, sp_pca_info = apply_pca(X_tr, X_val, args.pca_dim)
            sp_probe_dim = args.pca_dim
            print(f"  PCA {hidden_dim} -> {args.pca_dim}: "
                  f"{sp_pca_info['cumulative_variance_ratio']:.4f} variance retained")

        print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")
        met = train_classification_probe(
            X_tr, y_tr, X_val, y_val, sp_probe_dim, n_pred, args, device
        )
        met["layer"] = layer
        if sp_pca_info is not None:
            met["pca_dim"] = args.pca_dim
            met["pca_info"] = sp_pca_info
        met["predicate_names"] = pred_names
        results[layer] = met

        print(f"  macro_AUC={met['macro_auc']:.4f}  ep={met['epochs_trained']}  "
              f"{time.time()-t0:.1f}s")

    return results



def run_goal_classification(args, device):
    """Classify which goal (task) a frame belongs to across all tasks.

    All tasks share the same visual scene; the probe tests whether hidden
    states encode instruction information.
    """
    traj_metas, n_layers, hidden_dim = scan_features(args.features_dir, args.aggregation)
    layers = parse_layers(args.layers, n_layers - 1)

    # Build task-to-index mapping
    task_names = sorted(set(m["task"] for m in traj_metas))
    task_to_idx = {t: i for i, t in enumerate(task_names)}
    n_classes = len(task_names)

    # Split by demo: reuse split_by_task (80/20 per task by trajectory order)
    train_ids, val_ids = split_by_task(traj_metas)

    print(f"Features: {len(traj_metas)} trajs, {n_layers} layers, dim={hidden_dim}")
    print(f"Tasks ({n_classes}): {task_names}")
    print(f"Split: {len(train_ids)} train trajs / {len(val_ids)} val trajs")
    if args.max_frames is not None:
        print(f"Early-frame mode: using first {args.max_frames} frame(s) per trajectory")

    results = {}
    for li, layer in enumerate(layers):
        print(f"\n[Layer {layer}] ({li+1}/{len(layers)})", flush=True)
        t0 = time.time()

        train_feats = load_layer_features(traj_metas, train_ids, layer)
        val_feats = load_layer_features(traj_metas, val_ids, layer)

        if args.max_frames is not None:
            train_feats = [f[:args.max_frames] for f in train_feats]
            val_feats = [f[:args.max_frames] for f in val_feats]

        # Flatten: each timestep is one sample, label = task index
        def flatten(feats, ids, return_traj_ids=False):
            Xs, Ys, Ts = [], [], []
            for feat, idx in zip(feats, ids):
                label = task_to_idx[traj_metas[idx]["task"]]
                T = feat.shape[0]
                Xs.append(torch.from_numpy(feat))
                Ys.append(torch.full((T,), label, dtype=torch.long))
                if return_traj_ids:
                    Ts.append(torch.full((T,), idx, dtype=torch.long))
            if return_traj_ids:
                return torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
            return torch.cat(Xs), torch.cat(Ys)

        X_tr, y_tr = flatten(train_feats, train_ids)
        X_val, y_val, val_traj_ids = flatten(val_feats, val_ids, return_traj_ids=True)
        del train_feats, val_feats

        probe_dim = hidden_dim
        pca_info = None
        if args.pca_dim is not None:
            X_tr, X_val, pca_info = apply_pca(X_tr, X_val, args.pca_dim)
            probe_dim = args.pca_dim
            print(f"  PCA {hidden_dim} -> {args.pca_dim}: "
                  f"{pca_info['cumulative_variance_ratio']:.4f} variance retained")

        print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")
        met = train_goal_classification_probe(
            X_tr, y_tr, X_val, y_val, probe_dim, n_classes, args, device
        )
        val_preds = met.pop("val_preds")
        val_traj_np = val_traj_ids.numpy()
        val_y_np = y_val.cpu().numpy()
        traj_correct, traj_total = 0, 0
        for tidx in np.unique(val_traj_np):
            mask = val_traj_np == tidx
            majority_pred = np.bincount(val_preds[mask], minlength=n_classes).argmax()
            traj_correct += int(majority_pred == val_y_np[mask][0])
            traj_total += 1
        met["traj_accuracy"] = traj_correct / traj_total if traj_total > 0 else 0.0
        met["layer"] = layer
        met["task_names"] = task_names
        met["n_classes"] = n_classes
        if args.max_frames is not None:
            met["max_frames"] = args.max_frames
        if pca_info is not None:
            met["pca_dim"] = args.pca_dim
            met["pca_info"] = pca_info
        results[layer] = met

        print(f"  acc={met['accuracy']:.4f}  traj_acc={met['traj_accuracy']:.4f}  "
              f"F1={met['macro_f1']:.4f}  ep={met['epochs_trained']}  {time.time()-t0:.1f}s")

    return results


# ============================================================
# Results I/O
# ============================================================

def save_results(results, output_dir, probe_type):
    out = os.path.join(output_dir, probe_type)
    os.makedirs(out, exist_ok=True)

    for key, m in results.items():
        with open(os.path.join(out, f"layer_{key}.json"), "w") as f:
            json.dump(m, f, indent=2)

    csv_path = os.path.join(out, "summary.csv")
    if probe_type in ("temporal_distance", "oracle"):
        fields = ["layer", "spearman_rho", "per_task_mean_rho", "spearman_p",
                   "r2", "mae", "val_loss", "epochs_trained"]
    elif probe_type == "action_prediction":
        fields = ["layer", "overall_r2", "per_task_mean_r2",
                   "gripper_accuracy", "raw_mse", "val_loss", "epochs_trained"]
    elif probe_type == "action_delta":
        fields = ["layer", "overall_r2", "per_task_mean_r2",
                   "gripper_accuracy", "gripper_delta_accuracy",
                   "raw_mse", "val_loss", "epochs_trained"]
    elif probe_type == "goal_classification":
        fields = ["layer", "accuracy", "traj_accuracy", "macro_f1", "epochs_trained"]
    else:
        fields = ["layer", "macro_auc", "val_loss", "epochs_trained"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(results.keys(), key=lambda x: (isinstance(x, str), x)):
            row = {k: results[key].get(k, "") for k in fields}
            w.writerow(row)

    summary = dict(probe_type=probe_type, n_entries=len(results),
                   results={str(k): v for k, v in results.items()})

    if probe_type == "temporal_distance" and results:
        # D002: best_layer selected by per-task mean ρ (not pooled ρ)
        best = max(results, key=lambda k: results[k]["per_task_mean_rho"])
        summary["best_layer"] = best
        summary["best_rho"] = results[best]["per_task_mean_rho"]
        summary["best_pooled_rho"] = results[best]["spearman_rho"]

    if probe_type in ("action_prediction", "action_delta") and results:
        best = max(results, key=lambda k: results[k]["per_task_mean_r2"])
        summary["best_layer"] = best
        summary["best_r2"] = results[best]["per_task_mean_r2"]
        summary["best_overall_r2"] = results[best]["overall_r2"]

    if probe_type == "goal_classification" and results:
        best = max(results, key=lambda k: results[k]["accuracy"])
        summary["best_layer"] = best
        summary["best_accuracy"] = results[best]["accuracy"]
        summary["best_macro_f1"] = results[best]["macro_f1"]
        summary["best_traj_accuracy"] = max(
            results[k].get("traj_accuracy", 0) for k in results
        )
        # Per-layer accuracy curve
        summary["accuracy_by_layer"] = {
            str(k): results[k]["accuracy"] for k in sorted(results.keys())
        }
        summary["traj_accuracy_by_layer"] = {
            str(k): results[k].get("traj_accuracy") for k in sorted(results.keys())
        }

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out}/")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.pca_dim is not None:
        args.output_dir = args.output_dir.rstrip("/") + f"_pca{args.pca_dim}"
        print(f"PCA mode: reducing to {args.pca_dim} dims")

    print(f"=== Probe Training: {args.probe_type} | device={device} ===")

    runners = dict(
        temporal_distance=run_temporal_distance,
        oracle=run_oracle,
        subtask_predicate=run_subtask_predicate,
        action_prediction=run_action_prediction,
        action_delta=run_action_prediction,
        goal_classification=run_goal_classification,
    )
    results = runners[args.probe_type](args, device)
    save_results(results, args.output_dir, args.probe_type)


if __name__ == "__main__":
    main()
