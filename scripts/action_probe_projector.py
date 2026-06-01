"""Action prediction probe on SigLIP+projector, VLA image_mean, and VLA last_preaction features.

Replicates exact same pipeline from train_probes.py:
- ActionMLPProbe: Linear(in, 256) -> ReLU -> Linear(256, 7)
- 80/20 split per task (first 80% train, rest val)
- Adam lr=1e-3, batch_size=256, epochs=50, patience=5
- Normalize targets: y_mean, y_std from train
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
DEVICE = torch.device("cuda:0")

DATA_DIR = "./data/libero/libero_goal/"
SIGLIP_DIR = "./features/siglip_only_libero_goal/"
TRAINED_DIR = "./features/trained_libero_goal/"
UNTRAINED_DIR = "./features/untrained_libero_goal/"
OUT_DIR = "./results/functional_validation/"


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


def load_features_and_actions(features_dir, feat_key, layer=None):
    """Load features and actions for all tasks.

    Returns dict: task -> list of (feat_array, action_array) per demo.
    feat_key: 'last_preaction' or 'image_mean'
    layer: int or None (None for 2D features like projector)
    """
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(features_dir, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            print(f"  WARNING: no action file for {task}, skipping")
            continue

        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            demo_keys = sorted(
                [k for k in ff.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1])
            )
            for dk in demo_keys:
                ds = ff[dk][feat_key]
                if layer is not None and len(ds.shape) == 3:
                    feat = np.asarray(ds[:, layer, :], dtype=np.float32)
                else:
                    feat = np.asarray(ds[:], dtype=np.float32)
                demo_idx = int(dk.split("_")[-1])
                act = np.asarray(fa[f"data/{dk}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val(task_data):
    """80/20 split per task. Returns X_tr, y_tr, X_val, y_val, val_task_ids."""
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

    X_tr = torch.cat(X_tr_list)
    y_tr = torch.cat(y_tr_list)
    X_val = torch.cat(X_val_list)
    y_val = torch.cat(y_val_list)
    return X_tr, y_tr, X_val, y_val, val_tasks


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, tag=""):
    """Train ActionMLPProbe and evaluate. Returns metrics dict."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    in_dim = X_tr.shape[1]

    # Normalize targets
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

    # Per-dim R²
    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    per_dim_r2 = {}
    for d in range(7):
        per_dim_r2[dim_names[d]] = float(r2_score(y_val_np[:, d], pred_np[:, d]))

    # Overall R²
    overall_r2 = float(r2_score(y_val_np.flatten(), pred_np.flatten()))

    # Gripper accuracy
    grip_pred = (pred_np[:, 6] > 0).astype(float)
    grip_true = (y_val_np[:, 6] > 0).astype(float)
    grip_acc = float(np.mean(grip_pred == grip_true))

    # Per-task R²
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

    return {
        "per_task_R2": per_task_r2,
        "mean_R2": per_task_mean_r2,
        "overall_R2": overall_r2,
        "per_dim_R2": per_dim_r2,
        "gripper_acc": grip_acc,
        "val_loss": best_loss,
        "epochs_trained": final_ep,
    }


def bootstrap_ci(r2_a, r2_b, n_boot=10000, seed=42):
    """Bootstrap 95% CI for per-task R² difference (a - b). Task-level resampling."""
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

    # ==========================================
    # 1. SigLIP + Projector (single layer)
    # ==========================================
    print("\n=== SigLIP + Projector Action Probe ===")
    t0 = time.time()
    task_data = load_features_and_actions(SIGLIP_DIR, "last_preaction", layer=None)
    print(f"  Loaded {sum(len(v) for v in task_data.values())} demos from {len(task_data)} tasks")
    X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
    print(f"  Samples: {len(X_tr)} train / {len(X_val)} val, dim={X_tr.shape[1]}")
    results["projector"] = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, "projector")
    print(f"  Time: {time.time()-t0:.1f}s")
    del task_data, X_tr, y_tr, X_val, y_val

    # ==========================================
    # 2. VLA Trained image_mean L19, L23
    # ==========================================
    for layer in [19, 23]:
        print(f"\n=== VLA Trained image_mean L{layer} ===")
        t0 = time.time()
        task_data = load_features_and_actions(TRAINED_DIR, "image_mean", layer=layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
        print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")
        results[f"image_mean_trained_L{layer}"] = train_and_eval(
            X_tr, y_tr, X_val, y_val, val_tasks, f"im_trained_L{layer}")
        print(f"  Time: {time.time()-t0:.1f}s")
        del task_data, X_tr, y_tr, X_val, y_val

    # ==========================================
    # 3. VLA Trained last_preaction L19 (LLM baseline on same dataset)
    # ==========================================
    print(f"\n=== VLA Trained last_preaction L19 (LLM baseline) ===")
    t0 = time.time()
    task_data = load_features_and_actions(TRAINED_DIR, "last_preaction", layer=19)
    X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
    print(f"  Samples: {len(X_tr)} train / {len(X_val)} val")
    results["llm_trained_L19"] = train_and_eval(
        X_tr, y_tr, X_val, y_val, val_tasks, "llm_L19")
    print(f"  Time: {time.time()-t0:.1f}s")
    del task_data, X_tr, y_tr, X_val, y_val

    # ==========================================
    # 4. VLA Untrained image_mean — all 33 layers
    # ==========================================
    print(f"\n=== VLA Untrained image_mean (scanning all layers) ===")
    best_untrained_layer = None
    best_untrained_r2 = -999
    untrained_by_layer = {}

    for layer in range(33):
        t0 = time.time()
        task_data = load_features_and_actions(UNTRAINED_DIR, "image_mean", layer=layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
        m = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, f"untrained_im_L{layer}")
        untrained_by_layer[layer] = m
        if m["mean_R2"] > best_untrained_r2:
            best_untrained_r2 = m["mean_R2"]
            best_untrained_layer = layer
        del task_data, X_tr, y_tr, X_val, y_val
        print(f"    L{layer} time: {time.time()-t0:.1f}s")

    print(f"\n  Best untrained image_mean layer: L{best_untrained_layer} "
          f"mean_R²={best_untrained_r2:.4f}")
    results["image_mean_untrained_best"] = untrained_by_layer[best_untrained_layer]
    results["image_mean_untrained_best"]["best_layer"] = best_untrained_layer
    results["image_mean_untrained_all_layers"] = {
        str(l): {"mean_R2": m["mean_R2"], "overall_R2": m["overall_R2"]}
        for l, m in untrained_by_layer.items()
    }

    # ==========================================
    # 5. Bootstrap CI: projector vs LLM L19
    # ==========================================
    print("\n=== Bootstrap CI: projector vs LLM L19 ===")
    proj_r2 = results["projector"]["per_task_R2"]
    llm_r2 = results["llm_trained_L19"]["per_task_R2"]
    ci = bootstrap_ci(proj_r2, llm_r2, n_boot=10000)
    results["bootstrap_ci_proj_vs_llm"] = ci
    print(f"  Delta: {ci['delta']:.4f}, 95% CI: [{ci['ci_95'][0]:.4f}, {ci['ci_95'][1]:.4f}], "
          f"p={ci['p_value']:.4f}")

    # ==========================================
    # 6. Gate judgment
    # ==========================================
    proj_mean_r2 = results["projector"]["mean_R2"]
    if proj_mean_r2 >= 0.683:
        gate = "PASS"
    elif proj_mean_r2 > 0.676:
        gate = "PARTIAL"
    else:
        gate = "FAIL"
    results["gate_judgment"] = gate
    print(f"\n=== GATE: {gate} (projector mean_R²={proj_mean_r2:.4f}) ===")

    # ==========================================
    # Summary table
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k in ["projector", "image_mean_trained_L19", "image_mean_trained_L23",
              "llm_trained_L19", "image_mean_untrained_best"]:
        m = results[k]
        extra = f" (best_layer=L{m['best_layer']})" if "best_layer" in m else ""
        print(f"  {k:35s} mean_R²={m['mean_R2']:.4f}  overall_R²={m['overall_R2']:.4f}{extra}")

    # Save
    out_path = os.path.join(OUT_DIR, "action_probe_projector.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
