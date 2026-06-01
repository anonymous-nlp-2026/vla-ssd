"""Three-condition multi-seed action R2 probe (v2).

Fixes from v1:
  1. Train/val split is sequential (demo 0-39 train, 40-49 val), matching
     fulllayer_action_probe.py. No shuffle. Split is identical across seeds.
  2. R2 computed in original (denormalized) action space.
  3. Only probe weight initialization varies across seeds.
"""
import os, json, time, glob
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn

SEEDS = [42, 123, 456]
N_LAYERS = 33
INPUT_DIM = 4096
HIDDEN_DIM = 256
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5

DATA_DIR = "./data/libero/libero_goal/"
FEATURE_BASE = "./features/"

CONDITIONS = {
    "untrained": "untrained_libero_goal",
    "llama2base": "llama2base_libero_goal",
    "trained": "trained_libero_goal",
}

OUT_JSON = "./results/threecond_multiseed_v2.json"
OUT_FIG_DIR = "./results/figures/"


class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, out_dim=7, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_features_and_actions(feat_dir, layer):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(feat_dir, "*.h5"))):
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
    """Sequential 80/20 split by demo index (no shuffle). Matches fulllayer_action_probe.py."""
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
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
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


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = list(range(N_LAYERS))

    colors = {"untrained": "#4878CF", "llama2base": "#E8833A", "trained": "#6ACC65"}
    labels = {"untrained": "Untrained", "llama2base": "Llama-2 Base", "trained": "Trained"}

    for cond in ["untrained", "llama2base", "trained"]:
        means, stds = [], []
        for l in range(N_LAYERS):
            lk = f"L{l}"
            vals = results[cond][lk]["seeds"]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(layers, means, color=colors[cond], label=labels[cond], linewidth=1.8)
        ax.fill_between(layers, means - stds, means + stds,
                         color=colors[cond], alpha=0.15)

    l0_vals = [results[c]["L0"]["mean"] for c in ["untrained", "llama2base", "trained"]]
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.annotate("L0 (shared\nbaseline)", xy=(0, max(l0_vals)),
                xytext=(2.5, max(l0_vals) + 0.02),
                fontsize=8, color="gray", ha="left", va="bottom")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Action Probe R2", fontsize=11)
    ax.set_title("Per-Layer Action R2 (3 seeds, fixed split, v2)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, N_LAYERS - 1)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, "fig_threecond_multiseed_v2.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT_FIG_DIR, "fig_threecond_multiseed_v2.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUT_FIG_DIR}")


def main():
    device = torch.device("cuda:0")
    t_start = time.time()

    print("=" * 60)
    print("Three-condition multi-seed action R2 probe (v2)")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Layers: {N_LAYERS}")
    print(f"Device: {device}")
    print(f"Split: sequential 80/20 (no shuffle)")
    print(f"R2: original (denormalized) action space")
    print("=" * 60)

    results = {}

    for cond, feat_subdir in CONDITIONS.items():
        feat_dir = os.path.join(FEATURE_BASE, feat_subdir)
        results[cond] = {}

        print(f"\n{'='*40}")
        print(f"Condition: {cond} ({feat_subdir})")
        print(f"{'='*40}")

        for layer in range(N_LAYERS):
            t0 = time.time()
            task_data = load_features_and_actions(feat_dir, layer)
            X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)

            seed_r2s = []
            for seed in SEEDS:
                r2 = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, device, seed)
                seed_r2s.append(r2)
                torch.cuda.empty_cache()

            results[cond][f"L{layer}"] = {
                "seeds": seed_r2s,
                "mean": float(np.mean(seed_r2s)),
                "std": float(np.std(seed_r2s)),
            }

            elapsed = time.time() - t0
            print(f"  {cond} L{layer}: mean={np.mean(seed_r2s):.4f} +/- {np.std(seed_r2s):.4f} "
                  f"({seed_r2s}) {elapsed:.1f}s")

            del task_data, X_tr, y_tr, X_val, y_val

    output = {
        "conditions": list(CONDITIONS.keys()),
        "seeds": SEEDS,
        "split": "sequential_80_20_no_shuffle",
        "r2_space": "original_denormalized",
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {OUT_JSON}")

    make_figure(results)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
