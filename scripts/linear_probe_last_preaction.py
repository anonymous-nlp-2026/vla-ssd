"""Linear probe on last_preaction token: L0-L32 x trained/untrained.
Probe: Linear(4096, 7), no hidden layer.
Hyperparams match fulllayer_action_probe.py except probe architecture.
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
OUTPUT_DIM = 7

DATA_DIR = "./data/libero/libero_goal/"
FEATURE_BASE = "./features/"
MLP_JSON = "./results/functional_validation/fulllayer_action_probe.json"
OUT_PATH = "./results/linear_probe_last_preaction.json"

CONDITIONS = {
    "trained": "trained_libero_goal",
    "untrained": "untrained_libero_goal",
}
AGG_KEY = "last_preaction"


class LinearProbe(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, out_dim=OUTPUT_DIM):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


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


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-6)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    X_tr_d = X_tr.to(device)
    y_tr_d = y_tr.to(device)
    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    model = LinearProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = len(X_tr_d)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    epochs_trained = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            pred = model(X_tr_d[idx])
            loss = criterion(pred, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_d)
            val_loss = criterion(val_pred, y_val_d).item()

        epochs_trained = epoch + 1
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
    pred = pred_norm * y_std + y_mean
    y_true = y_val_raw.numpy()
    pred_np = pred.numpy()

    ss_res = np.sum((y_true - pred_np) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot)

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
    return {
        "mean_R2": mean_r2,
        "overall_R2": overall_r2,
        "per_task_R2": per_task_r2,
        "epochs_trained": epochs_trained,
    }


def find_peak(layer_results):
    best_layer, best_r2 = -1, -999
    for lk, metrics in layer_results.items():
        if metrics["mean_R2"] > best_r2:
            best_r2 = metrics["mean_R2"]
            best_layer = int(lk.split("_")[1])
    return {"layer": best_layer, "mean_R2": best_r2}


def check_inverted_u(layer_results, n_layers=N_LAYERS):
    r2s = [layer_results[f"layer_{i}"]["mean_R2"] for i in range(n_layers)]
    peak_idx = int(np.argmax(r2s))
    if peak_idx == 0 or peak_idx == n_layers - 1:
        return False
    left_lower = any(r2s[i] < r2s[peak_idx] for i in range(peak_idx))
    right_lower = any(r2s[i] < r2s[peak_idx] for i in range(peak_idx+1, n_layers))
    return left_lower and right_lower


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


def main():
    print("=" * 60)
    print("Linear probe (last_preaction): L0-L32 x trained/untrained")
    print(f"Probe: Linear({INPUT_DIM}, {OUTPUT_DIM})")
    print("=" * 60)
    t_start = time.time()

    device = torch.device("cuda:0")

    results = {}
    for condition, feat_subdir in CONDITIONS.items():
        feat_dir = os.path.join(FEATURE_BASE, feat_subdir)
        layer_results = {}
        for layer in range(N_LAYERS):
            t0 = time.time()
            task_data = load_features_and_actions(feat_dir, layer)
            X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)
            metrics = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, device)
            elapsed = time.time() - t0
            layer_results[f"layer_{layer}"] = metrics
            print(f"  {condition}/L{layer}: mean_R2={metrics['mean_R2']:.4f} "
                  f"({elapsed:.1f}s, ep={metrics['epochs_trained']})")
            del task_data, X_tr, y_tr, X_val, y_val
            torch.cuda.empty_cache()
        results[condition] = layer_results

    trained_peak = find_peak(results["trained"])
    untrained_peak = find_peak(results["untrained"])
    inverted_u_trained = check_inverted_u(results["trained"])
    inverted_u_untrained = check_inverted_u(results["untrained"])

    tr_peak_r2 = results["trained"][f"layer_{trained_peak['layer']}"]["per_task_R2"]
    un_peak_r2 = results["untrained"][f"layer_{untrained_peak['layer']}"]["per_task_R2"]
    trained_gt_untrained = trained_peak["mean_R2"] > untrained_peak["mean_R2"]

    print(f"\nPeak trained: L{trained_peak['layer']} R2={trained_peak['mean_R2']:.4f}")
    print(f"Peak untrained: L{untrained_peak['layer']} R2={untrained_peak['mean_R2']:.4f}")
    print(f"Inverted-U trained: {inverted_u_trained}")
    print(f"Trained > untrained at peak: {trained_gt_untrained}")

    bootstrap = {}
    comp = bootstrap_pairwise(tr_peak_r2, un_peak_r2)
    bootstrap["trained_vs_untrained_best_vs_best"] = comp
    print(f"\nBootstrap trained(L{trained_peak['layer']}) vs untrained(L{untrained_peak['layer']}): "
          f"delta={comp['delta']:.4f}, CI={comp['ci_95']}, p={comp['p_value']:.4f}")

    tr_sl = results["trained"][f"layer_{trained_peak['layer']}"]["per_task_R2"]
    un_sl = results["untrained"][f"layer_{trained_peak['layer']}"]["per_task_R2"]
    comp_sl = bootstrap_pairwise(tr_sl, un_sl)
    bootstrap["trained_vs_untrained_same_layer"] = comp_sl
    print(f"Bootstrap same-layer(L{trained_peak['layer']}): "
          f"delta={comp_sl['delta']:.4f}, CI={comp_sl['ci_95']}, p={comp_sl['p_value']:.4f}")

    mlp_comparison = None
    if os.path.exists(MLP_JSON):
        with open(MLP_JSON, "r") as f:
            mlp_data = json.load(f)
        if "trained" in mlp_data and "last_preaction" in mlp_data["trained"]:
            mlp_tr = mlp_data["trained"]["last_preaction"]
            mlp_un = mlp_data["untrained"]["last_preaction"]
            mlp_tr_peak = find_peak(mlp_tr)
            mlp_un_peak = find_peak(mlp_un)
            mlp_comparison = {
                "mlp_trained_peak": mlp_tr_peak,
                "mlp_untrained_peak": mlp_un_peak,
                "linear_trained_peak": trained_peak,
                "linear_untrained_peak": untrained_peak,
                "trained_linear_vs_mlp_delta": trained_peak["mean_R2"] - mlp_tr_peak["mean_R2"],
                "untrained_linear_vs_mlp_delta": untrained_peak["mean_R2"] - mlp_un_peak["mean_R2"],
            }
            print(f"\nMLP comparison (last_preaction):")
            print(f"  MLP trained peak: L{mlp_tr_peak['layer']} R2={mlp_tr_peak['mean_R2']:.4f}")
            print(f"  Linear trained peak: L{trained_peak['layer']} R2={trained_peak['mean_R2']:.4f}")
            print(f"  MLP untrained peak: L{mlp_un_peak['layer']} R2={mlp_un_peak['mean_R2']:.4f}")
            print(f"  Linear untrained peak: L{untrained_peak['layer']} R2={untrained_peak['mean_R2']:.4f}")

    trained_r2_dict = {f"L{i}": results["trained"][f"layer_{i}"]["mean_R2"] for i in range(N_LAYERS)}
    untrained_r2_dict = {f"L{i}": results["untrained"][f"layer_{i}"]["mean_R2"] for i in range(N_LAYERS)}

    output = {
        "readout": "last_preaction",
        "probe_type": "linear",
        "hyperparams": {
            "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
            "batch_size": BATCH_SIZE, "seed": SEED,
            "target": "next_action_a_{t+1}", "normalization": "z-score",
            "split": "80/20_per_task_by_demo_index",
            "model": f"Linear({INPUT_DIM},{OUTPUT_DIM})",
        },
        "results": {
            "trained": trained_r2_dict,
            "untrained": untrained_r2_dict,
        },
        "full_results": {
            "trained": results["trained"],
            "untrained": results["untrained"],
        },
        "summary": {
            "trained_peak_layer": trained_peak["layer"],
            "trained_peak_R2": trained_peak["mean_R2"],
            "untrained_peak_layer": untrained_peak["layer"],
            "untrained_peak_R2": untrained_peak["mean_R2"],
            "inverted_U_trained": inverted_u_trained,
            "inverted_U_untrained": inverted_u_untrained,
            "trained_gt_untrained_at_peak": trained_gt_untrained,
        },
        "bootstrap_comparisons": bootstrap,
    }
    if mlp_comparison:
        output["mlp_comparison"] = mlp_comparison

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
