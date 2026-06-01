"""Run action-delta probes on projector, trained image_mean L14, untrained image_mean (all layers).
Then compute bootstrap CI and save final JSON.
"""
import sys
sys.path.insert(0, './scripts')

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import h5py
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA as SklearnPCA

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DATA_DIR = "./data/libero/libero_goal/"
DEVICE = torch.device("cuda:1")

PROJ_FEAT_DIR = "./features/siglip_only_libero_goal/"
TRAINED_FEAT_DIR = "./features/trained_libero_goal/"
UNTRAINED_FEAT_DIR = "./features/untrained_libero_goal/"
OUTPUT_PATH = "./results/functional_validation/action_delta_probe_projector.json"


class ActionMLPProbe(nn.Module):
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


def scan_features(features_dir, feat_key):
    """Scan HDF5 files — return per-traj metadata, n_layers, hidden_dim."""
    import glob
    traj_metas = []
    n_layers = hidden_dim = None
    for h5_path in sorted(glob.glob(os.path.join(features_dir, "*.h5"))):
        task = Path(h5_path).stem
        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            for hk in keys:
                if feat_key not in f[hk]:
                    raise KeyError(f"No '{feat_key}' in {h5_path}/{hk}")
                shape = f[hk][feat_key].shape
                is_3d = len(shape) == 3
                T = shape[0]
                if n_layers is None:
                    n_layers = shape[1] if is_3d else 1
                    hidden_dim = shape[-1]
                traj_metas.append(dict(
                    task=task, h5_path=h5_path, h5_key=hk,
                    feat_key=feat_key, T=T, is_3d=is_3d,
                ))
    return traj_metas, n_layers, hidden_dim


def split_by_task(traj_metas):
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


def load_actions(traj_metas, indices, data_dir):
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


def train_delta_probe(features_dir, feat_key, layer, device=DEVICE):
    """Train a single action-delta probe. Returns result dict."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    traj_metas, n_layers, hidden_dim = scan_features(features_dir, feat_key)
    train_ids, val_ids = split_by_task(traj_metas)

    train_actions = load_actions(traj_metas, train_ids, DATA_DIR)
    val_actions = load_actions(traj_metas, val_ids, DATA_DIR)

    train_feats = load_layer_features(traj_metas, train_ids, layer)
    val_feats = load_layer_features(traj_metas, val_ids, layer)

    def build_shifted(feats, acts, ids_list):
        Xs, Ys, task_ids = [], [], []
        for i, (feat, act) in enumerate(zip(feats, acts)):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            Xs.append(torch.from_numpy(feat[:T - 1]))
            Ys.append(torch.from_numpy(act[1:T] - act[:T-1]))
            task_ids.extend([traj_metas[ids_list[i]]["task"]] * (T - 1))
        return torch.cat(Xs), torch.cat(Ys), task_ids

    X_tr, y_tr_raw, _ = build_shifted(train_feats, train_actions, train_ids)
    X_val, y_val_raw, val_tasks = build_shifted(val_feats, val_actions, val_ids)
    del train_feats, val_feats

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-8)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    model = ActionMLPProbe(hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
    X_val_d, y_val_d = X_val.to(device), y_val.to(device)

    best_loss, best_state, wait, final_ep = float("inf"), None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_d), device=device)
        for s in range(0, len(X_tr_d), BATCH_SIZE):
            idx = perm[s : s + BATCH_SIZE]
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

    grip_delta_pred = np.round(pred_np[:, 6]).clip(-1, 1).astype(int)
    grip_delta_true = np.round(y_val_np[:, 6]).clip(-1, 1).astype(int)
    grip_delta_acc = float(np.mean(grip_delta_pred == grip_delta_true))

    val_tasks_arr = np.array(val_tasks)
    unique_tasks = sorted(set(val_tasks))
    per_task_r2 = {}
    per_task_r2_values = []
    for task in unique_tasks:
        mask = val_tasks_arr == task
        if mask.sum() >= 2:
            v = float(r2_score(y_val_np[mask].flatten(), pred_np[mask].flatten()))
            per_task_r2[task] = v
            per_task_r2_values.append(v)
        else:
            per_task_r2[task] = float("nan")

    per_task_mean_r2 = float(np.mean(per_task_r2_values)) if per_task_r2_values else 0.0

    del X_tr_d, y_tr_d, X_val_d, y_val_d, model
    torch.cuda.empty_cache()

    return dict(
        per_task_R2=per_task_r2,
        mean_R2=per_task_mean_r2,
        overall_R2=overall_r2,
        per_dim_R2=per_dim_r2,
        gripper_delta_accuracy=grip_delta_acc,
        val_loss=best_loss,
        epochs_trained=final_ep,
        per_task_r2_values=per_task_r2_values,
    )


def bootstrap_ci(values_a, values_b, n_boot=10000, seed=42):
    """Bootstrap 95% CI on difference of means (a - b)."""
    rng = np.random.RandomState(seed)
    a = np.array(values_a)
    b = np.array(values_b)
    n = len(a)
    assert len(b) == n
    diffs = a - b
    obs_delta = float(np.mean(diffs))
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_deltas.append(np.mean(diffs[idx]))
    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    p_value = float(np.mean(boot_deltas <= 0)) if obs_delta > 0 else float(np.mean(boot_deltas >= 0))
    return dict(delta=obs_delta, ci_95=[ci_lo, ci_hi], p_value=p_value)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Projector features (single layer)
    print("=" * 60)
    print("1. SigLIP+Projector action-delta probe")
    print("=" * 60)
    t0 = time.time()
    proj_result = train_delta_probe(PROJ_FEAT_DIR, "last_preaction", layer=0)
    print(f"   mean_R2={proj_result['mean_R2']:.4f}  overall_R2={proj_result['overall_R2']:.4f}  "
          f"time={time.time()-t0:.1f}s")

    # 2. Trained VLA image_mean L14
    print("\n" + "=" * 60)
    print("2. Trained VLA image_mean L14 action-delta probe")
    print("=" * 60)
    t0 = time.time()
    trained_l14 = train_delta_probe(TRAINED_FEAT_DIR, "image_mean", layer=14)
    print(f"   mean_R2={trained_l14['mean_R2']:.4f}  overall_R2={trained_l14['overall_R2']:.4f}  "
          f"time={time.time()-t0:.1f}s")

    # 3. Untrained VLA image_mean — scan all layers, find best
    print("\n" + "=" * 60)
    print("3. Untrained VLA image_mean action-delta probe (all layers)")
    print("=" * 60)

    # First scan to get n_layers
    import glob
    sample_h5 = sorted(glob.glob(os.path.join(UNTRAINED_FEAT_DIR, "*.h5")))[0]
    with h5py.File(sample_h5, "r") as f:
        demo_key = sorted(f.keys())[0]
        n_layers = f[demo_key]["image_mean"].shape[1]
    print(f"   Scanning {n_layers} layers...")

    untrained_results = {}
    for layer in range(n_layers):
        t0 = time.time()
        res = train_delta_probe(UNTRAINED_FEAT_DIR, "image_mean", layer=layer)
        untrained_results[layer] = res
        print(f"   L{layer:02d}: mean_R2={res['mean_R2']:.4f}  overall_R2={res['overall_R2']:.4f}  "
              f"time={time.time()-t0:.1f}s")

    best_untrained_layer = max(untrained_results, key=lambda l: untrained_results[l]["mean_R2"])
    untrained_best = untrained_results[best_untrained_layer]
    print(f"\n   Best untrained layer: L{best_untrained_layer} (mean_R2={untrained_best['mean_R2']:.4f})")

    # 4. Bootstrap CI: projector vs LLM trained best (L14)
    print("\n" + "=" * 60)
    print("4. Bootstrap CI: projector vs trained_L14")
    print("=" * 60)

    # Align per-task R2 values
    proj_tasks = sorted(proj_result["per_task_R2"].keys())
    trained_tasks = sorted(trained_l14["per_task_R2"].keys())
    common_tasks = sorted(set(proj_tasks) & set(trained_tasks))

    proj_vals = [proj_result["per_task_R2"][t] for t in common_tasks
                 if not np.isnan(proj_result["per_task_R2"][t]) and not np.isnan(trained_l14["per_task_R2"][t])]
    llm_vals = [trained_l14["per_task_R2"][t] for t in common_tasks
                if not np.isnan(proj_result["per_task_R2"][t]) and not np.isnan(trained_l14["per_task_R2"][t])]

    boot_result = bootstrap_ci(proj_vals, llm_vals)
    print(f"   delta={boot_result['delta']:.4f}  CI95={boot_result['ci_95']}  p={boot_result['p_value']:.4f}")

    # 5. Gate judgment
    proj_r2 = proj_result["mean_R2"]
    llm_r2 = trained_l14["mean_R2"]
    dino_r2 = 0.0609  # baseline from task description

    if proj_r2 >= llm_r2:
        gate = f"PASS: projector delta R2 ({proj_r2:.4f}) >= LLM best ({llm_r2:.4f})"
    elif abs(proj_r2 - dino_r2) < 0.01:
        gate = f"CONSISTENT: projector delta R2 ({proj_r2:.4f}) ~ DINO ({dino_r2:.4f})"
    else:
        gate = f"BELOW: projector delta R2 ({proj_r2:.4f}) < LLM best ({llm_r2:.4f})"

    # Clean up internal values before saving
    def clean_result(r):
        r2 = dict(r)
        r2.pop("per_task_r2_values", None)
        return r2

    final = {
        "projector": clean_result(proj_result),
        "image_mean_trained_L14": clean_result(trained_l14),
        "image_mean_untrained_best": {
            **clean_result(untrained_best),
            "best_layer": best_untrained_layer,
        },
        "untrained_all_layers": {
            str(l): {"mean_R2": r["mean_R2"], "overall_R2": r["overall_R2"]}
            for l, r in untrained_results.items()
        },
        "bootstrap_ci_proj_vs_llm": boot_result,
        "gate_judgment": gate,
        "baselines": {
            "llm_trained_best_L14_delta_mean_R2": 0.0594,
            "llm_untrained_best_L32_delta_mean_R2": 0.0485,
            "dino_delta_mean_R2": 0.0609,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\n=== SUMMARY ===")
    print(f"Projector:               mean_R2={proj_result['mean_R2']:.4f}")
    print(f"Trained image_mean L14:  mean_R2={trained_l14['mean_R2']:.4f}")
    print(f"Untrained best (L{best_untrained_layer:02d}):    mean_R2={untrained_best['mean_R2']:.4f}")
    print(f"Bootstrap delta (proj-llm): {boot_result['delta']:.4f} CI95={boot_result['ci_95']}")
    print(f"Gate: {gate}")


if __name__ == "__main__":
    main()
