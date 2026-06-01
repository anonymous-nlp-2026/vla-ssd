"""Linear probe baseline: Linear(4096->7) per layer, trained vs untrained.
Validates that inverted-U pattern reproduces without MLP nonlinearity.
"""
import os, json, time, glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
N_LAYERS = 33
INPUT_DIM = 4096
ACTION_DIM = 7

DATA_DIR = "./data/libero/libero_goal/"
FEATURE_BASE = "./features/"
MLP_JSON = "./results/functional_validation/fulllayer_action_probe.json"
OUT_JSON = "./results/linear_probe_results.json"
FIG_DIR = "./results/figures/"

CONDITIONS = {
    "trained": "trained_libero_goal",
    "untrained": "untrained_libero_goal",
}
AGG_KEY = "image_mean"


def load_features_and_actions(feat_dir, layer):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(feat_dir, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            continue
        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            keys = sorted(
                [k for k in ff.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            for k in keys:
                idx = int(k.split("_")[-1])
                feat = np.asarray(ff[k][AGG_KEY][:, layer, :], dtype=np.float32)
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
            x = torch.from_numpy(feat[:T-1])
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


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, device="cpu"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-6)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    model = nn.Linear(INPUT_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = len(X_tr)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    epochs_trained = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            pred = model(X_tr[idx])
            loss = criterion(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        epochs_trained = epoch + 1
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_val)
    pred = pred_norm * y_std + y_mean
    y_true = y_val_raw.numpy()
    pred_np = pred.numpy()

    ss_res = np.sum((y_true - pred_np) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot)

    tasks_arr = np.array(val_tasks)
    per_task_r2 = {}
    for t in sorted(set(val_tasks)):
        mask = tasks_arr == t
        yt, yp = y_true[mask], pred_np[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean(axis=0)) ** 2)
        per_task_r2[t] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0

    mean_r2 = float(np.mean(list(per_task_r2.values())))
    return {
        "mean_R2": mean_r2,
        "overall_R2": overall_r2,
        "per_task_R2": per_task_r2,
        "epochs_trained": epochs_trained,
    }


def main():
    results = {
        "hyperparams": {
            "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
            "batch_size": BATCH_SIZE, "seed": SEED,
            "target": "next_action_a_{t+1}",
            "normalization": "z-score",
            "split": "80/20_per_task_by_demo_index",
            "model": "Linear(4096,7)",
            "readout": "image_mean",
        }
    }

    for cond, feat_subdir in CONDITIONS.items():
        feat_dir = os.path.join(FEATURE_BASE, feat_subdir)
        cond_results = {}
        for layer in range(N_LAYERS):
            t0 = time.time()
            task_data = load_features_and_actions(feat_dir, layer)
            X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
            metrics = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks)
            elapsed = time.time() - t0
            cond_results[f"layer_{layer}"] = metrics
            print(f"  {cond}/L{layer}: mean_R2={metrics['mean_R2']:.4f} "
                  f"({elapsed:.1f}s, ep={metrics['epochs_trained']})")
        results[cond] = {"image_mean": cond_results}

    with open(MLP_JSON, "r") as f:
        mlp_data = json.load(f)

    comparison = {"trained": {}, "untrained": {}}
    for cond in ["trained", "untrained"]:
        for layer in range(N_LAYERS):
            lk = f"layer_{layer}"
            lin_r2 = results[cond]["image_mean"][lk]["mean_R2"]
            mlp_r2 = mlp_data[cond]["image_mean"][lk]["mean_R2"]
            comparison[cond][lk] = {
                "linear_R2": lin_r2,
                "mlp_R2": mlp_r2,
                "delta": lin_r2 - mlp_r2,
            }
    results["comparison_linear_vs_mlp"] = comparison

    lin_trained = [results["trained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in range(N_LAYERS)]
    lin_untrained = [results["untrained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in range(N_LAYERS)]
    results["summary"] = {
        "trained_peak_layer": int(np.argmax(lin_trained)),
        "trained_peak_R2": float(np.max(lin_trained)),
        "untrained_peak_layer": int(np.argmax(lin_untrained)),
        "untrained_peak_R2": float(np.max(lin_untrained)),
        "inverted_U_trained": bool(np.argmax(lin_trained) not in [0, N_LAYERS-1]),
        "trained_gt_untrained_at_peak": bool(np.max(lin_trained) > np.max(lin_untrained)),
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")

    make_figure(results, mlp_data)


def make_figure(results, mlp_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(N_LAYERS))
    lin_tr = [results["trained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in layers]
    lin_un = [results["untrained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in layers]
    mlp_tr = [mlp_data["trained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in layers]
    mlp_un = [mlp_data["untrained"]["image_mean"][f"layer_{i}"]["mean_R2"] for i in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, mlp_tr, "o-", color="#2166ac", ms=4, lw=1.5, label="Trained (MLP)")
    ax.plot(layers, lin_tr, "s--", color="#2166ac", ms=4, lw=1.5, alpha=0.7, label="Trained (Linear)")
    ax.plot(layers, mlp_un, "o-", color="#b2182b", ms=4, lw=1.5, label="Untrained (MLP)")
    ax.plot(layers, lin_un, "s--", color="#b2182b", ms=4, lw=1.5, alpha=0.7, label="Untrained (Linear)")

    tr_peak = int(np.argmax(lin_tr))
    ax.axvline(tr_peak, color="#2166ac", ls=":", alpha=0.4)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Action R² (mean over tasks)", fontsize=10)
    ax.set_title("Linear vs MLP Probe: Per-Layer Action R²", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig_linear_vs_mlp_probe.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig_linear_vs_mlp_probe.png"), dpi=150)
    print(f"Saved figure: {FIG_DIR}fig_linear_vs_mlp_probe.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
