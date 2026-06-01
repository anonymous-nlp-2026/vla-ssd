"""Action probe for patch_mean features (CLIP + DINO) + DINO RSA with patch_mean.

Matches unified_action_probe.py hyperparams exactly:
  epochs=50, patience=5, lr=1e-3, batch=256, hidden=256, seed=42
  MLP(1024→256→7), target=a_{t+1}, z-score normalization, 80/20 per-task split
"""
import os, json, time, glob, itertools
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
HIDDEN_DIM = 256
DEVICE = torch.device("cuda:0")

DATA_DIR = "./data/libero/libero_goal/"
FEATURE_BASE = "./features/"
UNIFIED_PATH = "./results/functional_validation/unified_action_probe.json"
OUT_PATH = "./results/functional_validation/patch_mean_probe.json"

FEATURE_CONFIGS = {
    "clip_patch_mean": {
        "dir": "clip_libero_goal",
        "key_prefix": "traj",
        "feat_key": "patch_mean",
        "layer": None,
        "input_dim": 1024,
    },
    "dino_patch_mean": {
        "dir": "dinov2_libero_goal",
        "key_prefix": "traj",
        "feat_key": "patch_mean",
        "layer": None,
        "input_dim": 1024,
    },
}

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


class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim=7, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_features_and_actions(cfg):
    feat_dir = os.path.join(FEATURE_BASE, cfg["dir"])
    prefix = cfg["key_prefix"]
    feat_key = cfg["feat_key"]
    layer = cfg["layer"]

    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(feat_dir, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            print(f"  WARNING: no action file for {task}, skipping")
            continue

        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            keys = sorted(
                [k for k in ff.keys() if k.startswith(f"{prefix}_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            for k in keys:
                idx = int(k.split("_")[-1])
                ds = ff[k][feat_key]
                if layer is not None and len(ds.shape) == 3:
                    feat = np.asarray(ds[:, layer, :], dtype=np.float32)
                else:
                    feat = np.asarray(ds[:], dtype=np.float32)
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
            x = torch.from_numpy(feat[: T - 1])
            y = torch.from_numpy(act[1:T])

            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * len(x))

    X_tr = torch.cat(X_tr_list, dim=0)
    y_tr = torch.cat(y_tr_list, dim=0)
    X_val = torch.cat(X_val_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)
    return X_tr, y_tr, X_val, y_val, val_tasks


def train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, input_dim, tag):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp(min=1e-8)
    y_tr_n = (y_tr - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    model = ActionMLPProbe(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n = len(X_tr)
    best_val_loss = float("inf")
    patience_cnt = 0
    epochs_trained = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batch = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb = X_tr[idx].to(DEVICE)
            yb = y_tr_n[idx].to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1

        model.eval()
        with torch.no_grad():
            val_pred = []
            for start in range(0, len(X_val), BATCH_SIZE):
                xb = X_val[start : start + BATCH_SIZE].to(DEVICE)
                val_pred.append(model(xb).cpu())
            val_pred = torch.cat(val_pred, dim=0)
            val_loss = loss_fn(val_pred, y_val_n).item()

        epochs_trained = ep
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = []
        for start in range(0, len(X_val), BATCH_SIZE):
            xb = X_val[start : start + BATCH_SIZE].to(DEVICE)
            p = model(xb).cpu()
            pred.append(p * y_std + y_mean)
        pred = torch.cat(pred, dim=0)

    y_true = y_val.numpy()
    pred_np = pred.numpy()

    ss_res = np.sum((y_true - pred_np) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot)

    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    per_dim_r2 = {}
    for j, name in enumerate(dim_names):
        ss_r = np.sum((y_true[:, j] - pred_np[:, j]) ** 2)
        ss_t = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)
        per_dim_r2[name] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0

    gripper_acc = float(np.mean((pred_np[:, 6] >= 0) == (y_true[:, 6] >= 0)))

    tasks_arr = np.array(val_tasks)
    unique_tasks = sorted(set(val_tasks))
    per_task_r2 = {}
    for t in unique_tasks:
        mask = tasks_arr == t
        yt = y_true[mask]
        yp = pred_np[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean(axis=0)) ** 2)
        per_task_r2[t] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0

    mean_r2 = float(np.mean(list(per_task_r2.values())))

    print(f"  [{tag}] mean_R2={mean_r2:.4f}  overall_R2={overall_r2:.4f}  "
          f"gripper_acc={gripper_acc:.4f}  epochs={epochs_trained}")

    return {
        "mean_R2": mean_r2,
        "overall_R2": overall_r2,
        "per_task_R2": per_task_r2,
        "per_dim_R2": per_dim_r2,
        "gripper_acc": gripper_acc,
        "epochs_trained": epochs_trained,
        "val_loss": float(best_val_loss),
    }


def bootstrap_pairwise(r2_a, r2_b, n_boot=10000):
    np.random.seed(SEED)
    tasks = sorted(r2_a.keys())
    a = np.array([r2_a[t] for t in tasks])
    b = np.array([r2_b[t] for t in tasks])
    observed = float(a.mean() - b.mean())

    n = len(tasks)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_one = float(np.mean(deltas <= 0)) if observed > 0 else float(np.mean(deltas >= 0))
    p_value = float(min(p_one * 2, 1.0))

    return {"delta": observed, "ci_95": [ci_lo, ci_hi], "p_value": p_value}


# --- DINO patch_mean RSA (same methodology as rsa_clip.py) ---

def subsample_to_length(arr, target_len):
    T = arr.shape[0]
    if T <= target_len:
        return arr
    idx = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[idx]


def mean_cosine_distance(feat_a, feat_b):
    a = subsample_to_length(feat_a, N_SAMPLE)
    b = subsample_to_length(feat_b, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([cosine_dist(a[i], b[i]) for i in range(n)]))


def mean_action_distance(act_a, act_b):
    a = subsample_to_length(act_a, N_SAMPLE)
    b = subsample_to_length(act_b, N_SAMPLE)
    n = min(len(a), len(b))
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in range(n)]))


def compute_rsa(feats, actions):
    common = sorted(set(feats.keys()) & set(actions.keys()))
    if len(common) < 2:
        return None
    pairs = list(itertools.combinations(common, 2))
    rep_dists = [mean_cosine_distance(feats[a], feats[b]) for a, b in pairs]
    act_dists = [mean_action_distance(actions[a], actions[b]) for a, b in pairs]
    rsa, _ = spearmanr(rep_dists, act_dists)
    return float(rsa)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def run_dino_rsa(feature_key):
    """Run RSA for DINO features with specified feature_key."""
    feat_dir = Path(FEATURE_BASE) / "dinov2_libero_goal"
    data_dir = Path(DATA_DIR)

    per_task = {}
    rsa_values = []

    for task in TASKS:
        feat_path = feat_dir / f"{task}.h5"
        action_path = data_dir / f"{task}_demo.hdf5"
        if not feat_path.exists() or not action_path.exists():
            per_task[task] = None
            continue

        feats = {}
        with h5py.File(feat_path, "r") as f:
            for key in f.keys():
                if key.startswith("traj_"):
                    did = int(key.split("_")[1])
                    feats[did] = f[key][feature_key][:].astype(np.float32)

        actions = {}
        with h5py.File(action_path, "r") as f:
            for key in f["data"].keys():
                if key.startswith("demo_"):
                    did = int(key.split("_")[1])
                    actions[did] = f[f"data/{key}/actions"][:].astype(np.float32)

        rsa = compute_rsa(feats, actions)
        per_task[task] = rsa
        if rsa is not None:
            rsa_values.append(rsa)
        tag = f"{rsa:.4f}" if rsa is not None else "SKIP"
        print(f"  {task}: {tag}")

    if not rsa_values:
        return None, None, None, per_task

    mean_rsa, ci_lo, ci_hi = bootstrap_ci(rsa_values)
    return mean_rsa, ci_lo, ci_hi, per_task


def main():
    # Load existing results for bootstrap comparisons
    with open(UNIFIED_PATH) as f:
        unified = json.load(f)

    results = {}
    results["hyperparams"] = {
        "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
        "batch_size": BATCH_SIZE, "hidden": HIDDEN_DIM, "seed": SEED,
        "target": "next_action_a_{t+1}", "normalization": "z-score",
        "split": "80/20_per_task_by_demo_index",
        "note": "patch_mean supplement to unified_action_probe.json",
    }

    # --- Action probes ---
    for name, cfg in FEATURE_CONFIGS.items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        task_data = load_features_and_actions(cfg)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
        print(f"  Samples: {len(X_tr)} train / {len(X_val)} val, dim={X_tr.shape[1]}")
        metrics = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, cfg["input_dim"], name)
        results[name] = metrics
        print(f"  Time: {time.time() - t0:.1f}s")
        del task_data, X_tr, y_tr, X_val, y_val
        torch.cuda.empty_cache()

    # --- DINO RSA with patch_mean and cls_token ---
    print("\n=== DINO RSA ===")
    np.random.seed(SEED)
    for fkey in ["cls_token", "patch_mean"]:
        print(f"\n--- DINO feature_key={fkey} ---")
        mean_rsa, ci_lo, ci_hi, per_task = run_dino_rsa(fkey)
        results[f"dino_rsa_{fkey}"] = {
            "rsa": round(mean_rsa, 4) if mean_rsa is not None else None,
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)] if ci_lo is not None else None,
            "per_task": {k: round(v, 4) if v is not None else None for k, v in per_task.items()},
        }
        if mean_rsa is not None:
            print(f"  DINO ({fkey}) RSA: {mean_rsa:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # --- Bootstrap pairwise comparisons ---
    print("\n=== Bootstrap Pairwise Comparisons ===")
    bootstrap = {}

    pairs = [
        ("clip_patch_mean", "dino_patch_mean", results["clip_patch_mean"]["per_task_R2"], results["dino_patch_mean"]["per_task_R2"]),
        ("clip_patch_mean", "projector", results["clip_patch_mean"]["per_task_R2"], unified["projector"]["per_task_R2"]),
        ("dino_patch_mean", "dino_cls", results["dino_patch_mean"]["per_task_R2"], unified["dino_cls"]["per_task_R2"]),
        ("dino_patch_mean", "projector", results["dino_patch_mean"]["per_task_R2"], unified["projector"]["per_task_R2"]),
        ("clip_patch_mean", "clip_cls", results["clip_patch_mean"]["per_task_R2"], unified["clip_cls"]["per_task_R2"]),
    ]
    for a_name, b_name, r2_a, r2_b in pairs:
        key = f"{a_name}_vs_{b_name}"
        ci = bootstrap_pairwise(r2_a, r2_b)
        bootstrap[key] = ci
        print(f"  {key}: delta={ci['delta']:.4f}, CI={ci['ci_95']}, p={ci['p_value']:.4f}")
    results["bootstrap_pairwise"] = bootstrap

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  clip_patch_mean   mean_R2={results['clip_patch_mean']['mean_R2']:.4f}")
    print(f"  dino_patch_mean   mean_R2={results['dino_patch_mean']['mean_R2']:.4f}")
    print(f"  (existing) clip_cls       mean_R2={unified['clip_cls']['mean_R2']:.4f}")
    print(f"  (existing) dino_cls       mean_R2={unified['dino_cls']['mean_R2']:.4f}")
    print(f"  (existing) projector      mean_R2={unified['projector']['mean_R2']:.4f}")
    for fkey in ["cls_token", "patch_mean"]:
        rk = f"dino_rsa_{fkey}"
        if results[rk]["rsa"] is not None:
            print(f"  DINO RSA ({fkey}): {results[rk]['rsa']:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
