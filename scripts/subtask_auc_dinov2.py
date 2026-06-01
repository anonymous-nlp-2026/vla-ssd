"""Compute subtask AUC for DINO-V2 features using sklearn LogisticRegression.

Faster than train_probes.py for this specific task.
"""

import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_data(features_dir, label_path):
    with open(label_path) as f:
        label_data = json.load(f)
    pred_names = label_data["predicate_names"]
    labels = label_data["labels"]
    n_pred = len(pred_names)

    by_task = defaultdict(list)

    for h5_path in sorted(glob.glob(os.path.join(features_dir, "*.h5"))):
        task = Path(h5_path).stem
        if task not in labels:
            continue

        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            for tk in traj_keys:
                if tk not in labels[task]:
                    continue
                if "cls_token" in f[tk]:
                    feat = f[tk]["cls_token"][:]
                elif "patch_mean" in f[tk]:
                    feat = f[tk]["patch_mean"][:]
                else:
                    continue
                lab = np.array(labels[task][tk], dtype=np.float32)
                T = min(feat.shape[0], lab.shape[0])
                feat = feat[:T].astype(np.float32)
                lab = lab[:T]
                by_task[task].append((feat, lab))

    return by_task, pred_names, n_pred


def main():
    features_dir = "./features/dinov2/"
    label_path = "./subtask_labels.json"
    output_dir = "./results/dinov2/subtask_predicate/"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...", flush=True)
    t0 = time.time()
    by_task, pred_names, n_pred = load_data(features_dir, label_path)
    print(f"  Loaded {sum(len(v) for v in by_task.values())} trajectories "
          f"from {len(by_task)} tasks in {time.time()-t0:.1f}s", flush=True)

    # 80/20 split per task
    tasks = sorted(by_task.keys())
    X_train, Y_train, X_val, Y_val = [], [], [], []
    task_val_indices = {}
    val_offset = 0

    for task in tasks:
        trajs = by_task[task]
        n = len(trajs)
        n_train = int(n * 0.8)
        for feat, lab in trajs[:n_train]:
            X_train.append(feat)
            Y_train.append(lab)
        val_start = val_offset
        for feat, lab in trajs[n_train:]:
            X_val.append(feat)
            Y_val.append(lab)
            val_offset += feat.shape[0]
        task_val_indices[task] = (val_start, val_offset)

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)

    print(f"  Train: {X_train.shape[0]} frames, Val: {X_val.shape[0]} frames", flush=True)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train per-predicate classifiers
    results = {"predicate_names": pred_names}
    auc_per_pred = {}

    for pi, pname in enumerate(pred_names):
        print(f"\nTraining {pname}...", flush=True)
        t1 = time.time()

        y_tr = Y_train[:, pi]
        y_va = Y_val[:, pi]

        # Check label distribution
        tr_pos = y_tr.sum() / len(y_tr) * 100
        va_pos = y_va.sum() / len(y_va) * 100
        print(f"  Label balance: train {tr_pos:.1f}% pos, val {va_pos:.1f}% pos", flush=True)

        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            print(f"  SKIP: degenerate labels")
            auc_per_pred[pname] = float("nan")
            continue

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1
        )
        clf.fit(X_train, y_tr)

        scores = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_va, scores)
        auc_per_pred[pname] = float(auc)

        print(f"  AUC = {auc:.4f} ({time.time()-t1:.1f}s)", flush=True)

        # Per-task AUC
        task_aucs = {}
        for task, (s, e) in task_val_indices.items():
            y_t = y_va[s:e]
            sc_t = scores[s:e]
            if y_t.sum() > 0 and y_t.sum() < len(y_t):
                task_aucs[task] = float(roc_auc_score(y_t, sc_t))
            else:
                task_aucs[task] = float("nan")
        results[f"{pname}_per_task"] = task_aucs

        valid_task_aucs = [v for v in task_aucs.values() if not np.isnan(v)]
        if valid_task_aucs:
            print(f"  Per-task AUC: min={min(valid_task_aucs):.4f} "
                  f"max={max(valid_task_aucs):.4f} "
                  f"mean={np.mean(valid_task_aucs):.4f}", flush=True)

    valid_aucs = [v for v in auc_per_pred.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    results["auc_per_predicate"] = auc_per_pred
    results["macro_auc"] = macro_auc
    results["n_train"] = int(X_train.shape[0])
    results["n_val"] = int(X_val.shape[0])
    results["n_tasks"] = len(tasks)

    # Save
    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DINO-V2 Subtask AUC ===", flush=True)
    for pname, auc in auc_per_pred.items():
        print(f"  {pname}: AUC = {auc:.4f}", flush=True)
    print(f"  Macro AUC = {macro_auc:.4f}", flush=True)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
