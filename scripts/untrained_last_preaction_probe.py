"""Untrained last_preaction action probe + action-delta probe.
Exact same pipeline as action_probe_projector.py (action) and run_action_delta_all.py (delta).
"""
import os, json, time, glob
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DEVICE = torch.device("cuda:1")

DATA_DIR = "./data/libero/libero_goal/"
UNTRAINED_DIR = "./features/untrained_libero_goal/"
OUT_PATH = "./results/functional_validation/untrained_last_preaction_probe.json"

DIM_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


class ActionMLPProbe(nn.Module):
    """2-layer MLP for action probe (same as action_probe_projector.py)."""
    def __init__(self, in_dim, out_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class DeltaMLPProbe(nn.Module):
    """3-layer MLP for delta probe (same as run_action_delta_all.py)."""
    def __init__(self, in_dim, out_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_features_and_actions(features_dir, feat_key, layer):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(features_dir, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            continue
        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            demo_keys = sorted(
                [k for k in ff.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1])
            )
            for dk in demo_keys:
                ds = ff[dk][feat_key]
                if len(ds.shape) == 3:
                    feat = np.asarray(ds[:, layer, :], dtype=np.float32)
                else:
                    feat = np.asarray(ds[:], dtype=np.float32)
                act = np.asarray(fa[f"data/{dk}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val_action(task_data):
    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list, val_tasks = [], [], []
    for task in sorted(task_data.keys()):
        demos = task_data[task]
        n_train = int(len(demos) * 0.8)
        for i, (feat, act) in enumerate(demos):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            x = torch.from_numpy(feat[:T-1])
            y = torch.from_numpy(act[1:T])
            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * (T - 1))
    return torch.cat(X_tr_list), torch.cat(y_tr_list), torch.cat(X_val_list), torch.cat(y_val_list), val_tasks


def build_train_val_delta(task_data):
    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list, val_tasks = [], [], []
    for task in sorted(task_data.keys()):
        demos = task_data[task]
        n_train = int(len(demos) * 0.8)
        for i, (feat, act) in enumerate(demos):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            x = torch.from_numpy(feat[:T-1])
            y = torch.from_numpy(act[1:T] - act[:T-1])
            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * (T - 1))
    return torch.cat(X_tr_list), torch.cat(y_tr_list), torch.cat(X_val_list), torch.cat(y_val_list), val_tasks


def train_probe(model_cls, X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, tag=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    in_dim = X_tr.shape[1]

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-8)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    model = model_cls(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    X_tr_d, y_tr_d = X_tr.to(DEVICE), y_tr.to(DEVICE)
    X_val_d, y_val_d = X_val.to(DEVICE), y_val.to(DEVICE)

    best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_d), device=DEVICE)
        for s in range(0, len(X_tr_d), BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
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
            if wait >= PATIENCE:
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

    per_dim_r2 = {}
    for d in range(7):
        per_dim_r2[DIM_NAMES[d]] = float(r2_score(y_val_np[:, d], pred_np[:, d]))

    overall_r2 = float(r2_score(y_val_np.flatten(), pred_np.flatten()))

    grip_pred = (pred_np[:, 6] > 0).astype(float)
    grip_true = (y_val_np[:, 6] > 0).astype(float)
    grip_acc = float(np.mean(grip_pred == grip_true))

    val_tasks_arr = np.array(val_tasks)
    per_task_r2 = {}
    for task in sorted(set(val_tasks)):
        mask = val_tasks_arr == task
        if mask.sum() >= 2:
            per_task_r2[task] = float(r2_score(y_val_np[mask].flatten(), pred_np[mask].flatten()))

    valid_r2s = [v for v in per_task_r2.values() if not np.isnan(v)]
    mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0

    del X_tr_d, y_tr_d, X_val_d, y_val_d, model
    torch.cuda.empty_cache()

    print(f"  [{tag}] R2={overall_r2:.4f}  mean_task={mean_r2:.4f}  grip={grip_acc:.4f}  ep={final_ep}")
    return {
        "per_task_R2": per_task_r2, "mean_R2": mean_r2, "overall_R2": overall_r2,
        "per_dim_R2": per_dim_r2, "gripper_acc": grip_acc,
        "val_loss": best_loss, "epochs_trained": final_ep,
    }


def train_delta_result(model_cls, X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, tag=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    in_dim = X_tr.shape[1]

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-8)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    model = model_cls(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    X_tr_d, y_tr_d = X_tr.to(DEVICE), y_tr.to(DEVICE)
    X_val_d, y_val_d = X_val.to(DEVICE), y_val.to(DEVICE)

    best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_d), device=DEVICE)
        for s in range(0, len(X_tr_d), BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
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
            if wait >= PATIENCE:
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

    per_dim_r2 = {}
    for d in range(7):
        per_dim_r2[DIM_NAMES[d]] = float(r2_score(y_val_np[:, d], pred_np[:, d]))

    overall_r2 = float(r2_score(y_val_np.flatten(), pred_np.flatten()))

    grip_delta_pred = np.round(pred_np[:, 6]).clip(-1, 1).astype(int)
    grip_delta_true = np.round(y_val_np[:, 6]).clip(-1, 1).astype(int)
    grip_delta_acc = float(np.mean(grip_delta_pred == grip_delta_true))

    val_tasks_arr = np.array(val_tasks)
    per_task_r2 = {}
    for task in sorted(set(val_tasks)):
        mask = val_tasks_arr == task
        if mask.sum() >= 2:
            per_task_r2[task] = float(r2_score(y_val_np[mask].flatten(), pred_np[mask].flatten()))

    valid_r2s = [v for v in per_task_r2.values() if not np.isnan(v)]
    mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0

    del X_tr_d, y_tr_d, X_val_d, y_val_d, model
    torch.cuda.empty_cache()

    print(f"  [{tag}] R2={overall_r2:.4f}  mean_task={mean_r2:.4f}  grip_delta={grip_delta_acc:.4f}  ep={final_ep}")
    return {
        "per_task_R2": per_task_r2, "mean_R2": mean_r2, "overall_R2": overall_r2,
        "per_dim_R2": per_dim_r2, "gripper_delta_accuracy": grip_delta_acc,
        "val_loss": best_loss, "epochs_trained": final_ep,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results = {}

    # 1. Action probe: L19
    print("=== Untrained last_preaction ACTION probe L19 ===")
    t0 = time.time()
    task_data = load_features_and_actions(UNTRAINED_DIR, "last_preaction", layer=19)
    X_tr, y_tr, X_val, y_val, val_tasks = build_train_val_action(task_data)
    print(f"  Samples: {len(X_tr)} train / {len(X_val)} val, dim={X_tr.shape[1]}")
    results["untrained_last_preaction_L19"] = train_probe(
        ActionMLPProbe, X_tr, y_tr, X_val, y_val, val_tasks, "action_L19")
    print(f"  Time: {time.time()-t0:.1f}s")
    del task_data, X_tr, y_tr, X_val, y_val

    # 2. Action probe: all 33 layers
    print("\n=== Untrained last_preaction ACTION probe ALL LAYERS ===")
    best_layer, best_r2 = -1, -999
    all_layers = {}
    for layer in range(33):
        t0 = time.time()
        task_data = load_features_and_actions(UNTRAINED_DIR, "last_preaction", layer=layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val_action(task_data)
        m = train_probe(ActionMLPProbe, X_tr, y_tr, X_val, y_val, val_tasks, f"action_L{layer}")
        all_layers[layer] = m
        if m["mean_R2"] > best_r2:
            best_r2 = m["mean_R2"]
            best_layer = layer
        del task_data, X_tr, y_tr, X_val, y_val
        print(f"    L{layer} time: {time.time()-t0:.1f}s")

    print(f"\n  Best layer: L{best_layer} mean_R2={best_r2:.4f}")
    results["untrained_last_preaction_best"] = all_layers[best_layer]
    results["untrained_last_preaction_best"]["best_layer"] = best_layer
    results["untrained_last_preaction_all_layers"] = {
        str(l): {"mean_R2": m["mean_R2"], "overall_R2": m["overall_R2"]}
        for l, m in all_layers.items()
    }

    # 3. Action-delta probe: L19
    print("\n=== Untrained last_preaction DELTA probe L19 ===")
    t0 = time.time()
    task_data = load_features_and_actions(UNTRAINED_DIR, "last_preaction", layer=19)
    X_tr, y_tr, X_val, y_val, val_tasks = build_train_val_delta(task_data)
    print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")
    results["untrained_last_preaction_delta_L19"] = train_delta_result(
        DeltaMLPProbe, X_tr, y_tr, X_val, y_val, val_tasks, "delta_L19")
    print(f"  Time: {time.time()-t0:.1f}s")
    del task_data, X_tr, y_tr, X_val, y_val

    # 4. Action-delta probe: best action-probe layer
    if best_layer != 19:
        print(f"\n=== Untrained last_preaction DELTA probe L{best_layer} (best action layer) ===")
        t0 = time.time()
        task_data = load_features_and_actions(UNTRAINED_DIR, "last_preaction", layer=best_layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val_delta(task_data)
        results[f"untrained_last_preaction_delta_L{best_layer}"] = train_delta_result(
            DeltaMLPProbe, X_tr, y_tr, X_val, y_val, val_tasks, f"delta_L{best_layer}")
        print(f"  Time: {time.time()-t0:.1f}s")
        del task_data, X_tr, y_tr, X_val, y_val

    # 5. Comparison summary
    results["comparison"] = {
        "trained_last_preaction_L19": 0.7309,
        "projector": 0.7063,
        "untrained_last_preaction_L19": results["untrained_last_preaction_L19"]["mean_R2"],
        "untrained_last_preaction_best_layer": best_layer,
        "untrained_last_preaction_best_mean_R2": best_r2,
    }
    u_l19 = results["untrained_last_preaction_L19"]["mean_R2"]
    if u_l19 > 0.7063:
        results["comparison"]["verdict"] = "architecture_property: untrained last_preaction > projector"
    else:
        results["comparison"]["verdict"] = "training_matters: untrained last_preaction < projector"

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  trained last_preaction L19:   mean_R2=0.7309")
    print(f"  projector:                    mean_R2=0.7063")
    print(f"  untrained last_preaction L19: mean_R2={u_l19:.4f}")
    print(f"  untrained best (L{best_layer}):       mean_R2={best_r2:.4f}")
    print(f"  verdict: {results['comparison']['verdict']}")


if __name__ == "__main__":
    main()
