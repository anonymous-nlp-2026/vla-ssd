"""DINO action probe on LIBERO-Goal: action + action-delta + bootstrap CI vs projector."""
import os, json, time, glob
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
DEVICE = torch.device("cuda:0")

DINO_DIR = "./features/dinov2_libero_goal/"
DATA_DIR = "./data/libero/libero_goal/"
OUT_DIR = "./results/functional_validation/"
PROJ_RESULTS = os.path.join(OUT_DIR, "action_probe_projector.json")


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


def load_dino_and_actions(feat_key="cls_token"):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(DINO_DIR, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            print(f"  WARNING: no action file for {task}, skipping")
            continue

        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            traj_keys = sorted(
                [k for k in ff.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[-1])
            )
            for tk in traj_keys:
                demo_idx = int(tk.split("_")[-1])
                dk = f"demo_{demo_idx}"
                feat = np.asarray(ff[tk][feat_key][:], dtype=np.float32)
                act = np.asarray(fa[f"data/{dk}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val(task_data, delta=False):
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
            if delta:
                y = torch.from_numpy(act[1:T] - act[:T-1])
            else:
                y = torch.from_numpy(act[1:T])

            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * (T - 1))

    return torch.cat(X_tr_list), torch.cat(y_tr_list), torch.cat(X_val_list), torch.cat(y_val_list), val_tasks


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, tag=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    in_dim = X_tr.shape[1]
    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-8)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    model = ActionMLPProbe(in_dim).to(DEVICE)
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

    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    per_dim_r2 = {}
    for d in range(7):
        per_dim_r2[dim_names[d]] = float(r2_score(y_val_np[:, d], pred_np[:, d]))

    overall_r2 = float(r2_score(y_val_np.flatten(), pred_np.flatten()))

    grip_pred = (pred_np[:, 6] > 0).astype(float)
    grip_true = (y_val_np[:, 6] > 0).astype(float)
    grip_acc = float(np.mean(grip_pred == grip_true))

    val_tasks_arr = np.array(val_tasks)
    unique_tasks = sorted(set(val_tasks))
    per_task_r2 = {}
    for task in unique_tasks:
        mask = val_tasks_arr == task
        if mask.sum() >= 2:
            per_task_r2[task] = float(r2_score(
                y_val_np[mask].flatten(), pred_np[mask].flatten()))

    valid_r2s = [v for v in per_task_r2.values() if not np.isnan(v)]
    per_task_mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else 0.0

    del X_tr_d, y_tr_d, X_val_d, y_val_d
    torch.cuda.empty_cache()

    print(f"  [{tag}] R²={overall_r2:.4f}  mean_task_R²={per_task_mean_r2:.4f}  "
          f"grip_acc={grip_acc:.4f}  ep={final_ep}")

    result = {
        "per_task_R2": per_task_r2,
        "mean_R2": per_task_mean_r2,
        "overall_R2": overall_r2,
        "per_dim_R2": per_dim_r2,
        "gripper_acc": grip_acc,
        "val_loss": best_loss,
        "epochs_trained": final_ep,
    }
    return result


def bootstrap_ci(r2_a, r2_b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    tasks = sorted(set(r2_a.keys()) & set(r2_b.keys()))
    diffs = np.array([r2_a[t] - r2_b[t] for t in tasks])
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    delta = float(diffs.mean())
    p_value = float(np.mean(boot_means < 0)) if delta > 0 else float(np.mean(boot_means > 0))
    return {"delta": delta, "ci_95": [ci_lo, ci_hi], "p_value": p_value}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    # 1. DINO action probe (cls_token)
    print("=== DINO cls_token action probe ===")
    t0 = time.time()
    task_data = load_dino_and_actions("cls_token")
    X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data, delta=False)
    print(f"  Samples: {len(X_tr)} train / {len(X_val)} val, dim={X_tr.shape[1]}")
    results["dino_action"] = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, "dino_cls_action")
    print(f"  Time: {time.time()-t0:.1f}s")

    # 2. DINO action probe (patch_mean)
    print("\n=== DINO patch_mean action probe ===")
    t0 = time.time()
    task_data_pm = load_dino_and_actions("patch_mean")
    X_tr_pm, y_tr_pm, X_val_pm, y_val_pm, val_tasks_pm = build_train_val(task_data_pm, delta=False)
    results["dino_action_patch_mean"] = train_and_eval(
        X_tr_pm, y_tr_pm, X_val_pm, y_val_pm, val_tasks_pm, "dino_pm_action")
    print(f"  Time: {time.time()-t0:.1f}s")
    del X_tr_pm, y_tr_pm, X_val_pm, y_val_pm

    # 3. DINO action-delta probe (cls_token)
    print("\n=== DINO cls_token action-delta probe ===")
    t0 = time.time()
    X_tr_d, y_tr_d, X_val_d, y_val_d, val_tasks_d = build_train_val(task_data, delta=True)
    print(f"  Samples: {len(X_tr_d)} train / {len(X_val_d)} val")
    results["dino_action_delta"] = train_and_eval(
        X_tr_d, y_tr_d, X_val_d, y_val_d, val_tasks_d, "dino_cls_delta")
    # action-delta doesn't have gripper_acc in the same sense, remove it
    results["dino_action_delta"].pop("gripper_acc", None)
    print(f"  Time: {time.time()-t0:.1f}s")
    del X_tr_d, y_tr_d, X_val_d, y_val_d

    # 4. Bootstrap CI: DINO vs projector
    print("\n=== Bootstrap CI: DINO vs Projector ===")
    with open(PROJ_RESULTS) as f:
        proj = json.load(f)
    proj_r2 = proj["projector"]["per_task_R2"]
    dino_r2 = results["dino_action"]["per_task_R2"]
    ci = bootstrap_ci(dino_r2, proj_r2, n_boot=10000)
    results["bootstrap_dino_vs_proj"] = ci
    print(f"  Delta(DINO-proj): {ci['delta']:.4f}, 95% CI: [{ci['ci_95'][0]:.4f}, {ci['ci_95'][1]:.4f}], "
          f"p={ci['p_value']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k in ["dino_action", "dino_action_patch_mean", "dino_action_delta"]:
        m = results[k]
        print(f"  {k:30s} mean_R²={m['mean_R2']:.4f}  overall_R²={m['overall_R2']:.4f}")

    out_path = os.path.join(OUT_DIR, "dino_action_probe.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
