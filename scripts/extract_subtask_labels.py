"""Extract subtask completion labels from LIBERO demo MuJoCo states.

For each LIBERO-Long task (2 subtasks connected by 'and'):
1. Detect object-state dimensions with significant step-like transitions
2. Cluster these dims into 2 groups by transition timing (early/late)
3. For each demo, detect subtask completion times from state thresholds
4. Output: labels.json for train_probes.py --probe_type subtask_predicate
"""

import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans


ROBOT_DIMS = 8


def detect_step_dims(states_list, min_demos_ratio=0.4, min_change=0.03):
    n_dims = states_list[0].shape[1]
    dim_info = {}

    for states in states_list:
        T = states.shape[0]
        for dim in range(ROBOT_DIMS, n_dims):
            vals = states[:, dim]
            total = abs(vals[-1] - vals[0])
            if total < min_change:
                continue
            cs = np.cumsum(np.abs(np.diff(vals)))
            if cs[-1] == 0:
                continue
            mid = np.searchsorted(cs, cs[-1] * 0.5) / T
            # Filter out initial settling (< 5%) and final noise (> 98%)
            if mid < 0.05 or mid > 0.98:
                continue
            dim_info.setdefault(dim, []).append(mid)

    n_demos = len(states_list)
    result = []
    for dim, pcts in dim_info.items():
        if len(pcts) >= n_demos * min_demos_ratio:
            m, s = np.mean(pcts), np.std(pcts)
            if s < 0.25:
                result.append((dim, m))
    return result


def cluster_subtasks(dim_pcts, n_subtasks=2):
    if len(dim_pcts) < n_subtasks:
        return None
    pcts = np.array([p for _, p in dim_pcts]).reshape(-1, 1)
    km = KMeans(n_clusters=n_subtasks, random_state=42, n_init=10)
    labels = km.fit_predict(pcts)
    order = np.argsort([np.mean(pcts[labels == c]) for c in range(n_subtasks)])
    groups = {}
    for ci, cid in enumerate(order):
        groups[ci] = [dim_pcts[i][0] for i in range(len(dim_pcts)) if labels[i] == cid]
    return groups


def compute_completion_time(states, dims, threshold=0.8):
    times = []
    for dim in dims:
        vals = states[:, dim]
        total = abs(vals[-1] - vals[0])
        if total < 0.01:
            continue
        cs = np.cumsum(np.abs(np.diff(vals)))
        if cs[-1] > 0:
            ct = np.searchsorted(cs, cs[-1] * threshold)
            times.append(ct)
    return int(np.median(times)) if times else None


def extract_task_labels(demo_path, feature_task_name):
    with h5py.File(demo_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

        sample_idx = np.linspace(0, len(demo_keys) - 1, min(15, len(demo_keys)), dtype=int)
        sample_states = [f[f"data/{demo_keys[i]}/states"][:] for i in sample_idx]

        dim_pcts = detect_step_dims(sample_states)
        if len(dim_pcts) < 2:
            return None, None

        groups = cluster_subtasks(dim_pcts)
        if groups is None:
            return None, None

        early_mean = np.mean([p for d, p in dim_pcts if d in groups[0]])
        late_mean = np.mean([p for d, p in dim_pcts if d in groups[1]])
        if late_mean - early_mean < 0.1:
            return None, None

        pred_names = ["subtask_0", "subtask_1"]
        demo_labels = {}

        for dk in demo_keys:
            states = f[f"data/{dk}/states"][:]
            T = states.shape[0]
            demo_idx = int(dk.split("_")[-1])
            traj_key = f"traj_{demo_idx}"

            lab = np.zeros((T, 2), dtype=np.float32)
            for si in range(2):
                ct = compute_completion_time(states, groups[si])
                if ct is not None and ct < T:
                    lab[ct:, si] = 1.0

            demo_labels[traj_key] = lab.tolist()

        return pred_names, demo_labels


def main():
    demo_dir = "./data/libero/libero_10/"
    feature_dir = "./features/dinov2/"
    output_path = "./subtask_labels.json"

    feature_tasks = {Path(p).stem for p in glob.glob(os.path.join(feature_dir, "*.h5"))}

    all_labels = {}
    pred_names = None

    for demo_file in sorted(glob.glob(os.path.join(demo_dir, "*.hdf5"))):
        task_name = Path(demo_file).stem.replace("_demo", "")
        if task_name not in feature_tasks:
            print(f"SKIP {task_name} (no features)")
            continue

        print(f"Processing: {task_name}")
        pn, dl = extract_task_labels(demo_file, task_name)
        if pn is None:
            print("  WARNING: could not extract labels")
            continue

        all_labels[task_name] = dl
        if pred_names is None:
            pred_names = pn

        n = len(dl)
        # Stats from first demo
        example = list(dl.values())[0]
        T = len(example)
        onset_0 = next((i for i, row in enumerate(example) if row[0] > 0.5), T)
        onset_1 = next((i for i, row in enumerate(example) if row[1] > 0.5), T)
        print(f"  {n} demos, T={T}, subtask_0 @{onset_0} ({onset_0/T*100:.0f}%), "
              f"subtask_1 @{onset_1} ({onset_1/T*100:.0f}%)")

    result = {"predicate_names": pred_names, "labels": all_labels}
    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"\nSaved: {output_path}")
    print(f"Tasks: {len(all_labels)}, Predicates: {pred_names}")


if __name__ == "__main__":
    main()
