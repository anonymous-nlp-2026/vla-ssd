import json
import time
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

TRAINED_PATH = "./results/cross_instruction/trained_cross_inst.h5"
UNTRAINED_PATH = "./results/cross_instruction/untrained_cross_inst.h5"
OUTPUT_PATH = "./results/cross_instruction/loocv_results.json"

N_TASKS = 10
N_FRAMES = 50
N_INSTRUCTIONS = 10
TOKEN_POSITIONS = ["image_mean", "last_preaction"]
PCA_COMPONENTS = 256
SEED = 42


def load_all_features(path):
    """Load features grouped by task.
    Returns: dict[token_pos][layer_name] -> (10, 500, 4096)
    Also returns labels: (500,) with instruction_idx repeated 50 times
    """
    with h5py.File(path, "r") as f:
        layers = f["metadata/layers"][:]
        layer_names = [f"L{l}" for l in layers]

        data = {tp: {ln: [] for ln in layer_names} for tp in TOKEN_POSITIONS}

        for task_idx in range(N_TASKS):
            task_samples = {tp: {ln: [] for ln in layer_names} for tp in TOKEN_POSITIONS}
            for frame_idx in range(N_FRAMES):
                for inst_idx in range(N_INSTRUCTIONS):
                    grp = f[f"task_{task_idx}/frame_{frame_idx}/instruction_{inst_idx}"]
                    for tp in TOKEN_POSITIONS:
                        arr = grp[tp][:]  # (5, 4096)
                        for li, ln in enumerate(layer_names):
                            task_samples[tp][ln].append(arr[li])
            for tp in TOKEN_POSITIONS:
                for ln in layer_names:
                    data[tp][ln].append(np.array(task_samples[tp][ln], dtype=np.float32))

        for tp in TOKEN_POSITIONS:
            for ln in layer_names:
                data[tp][ln] = np.array(data[tp][ln])  # (10, 500, 4096)

    # Labels: instruction index (0-9), repeated for each frame
    # Order in 500 samples: frame0_inst0, frame0_inst1, ..., frame0_inst9, frame1_inst0, ...
    labels = np.tile(np.arange(N_INSTRUCTIONS), N_FRAMES)  # (500,)

    return data, layer_names, labels


def run_loocv(data, layer_names, labels, condition_name):
    """Leave-one-task-out CV. Predict instruction label (10-class)."""
    results = {}
    for tp in TOKEN_POSITIONS:
        results[tp] = {}
        for ln in layer_names:
            fold_accs = []
            all_data = data[tp][ln]  # (10, 500, 4096)

            for test_task in range(N_TASKS):
                train_mask = [i for i in range(N_TASKS) if i != test_task]
                X_train = all_data[train_mask].reshape(-1, all_data.shape[-1])  # (4500, 4096)
                y_train = np.tile(labels, len(train_mask))  # (4500,)
                X_test = all_data[test_task]  # (500, 4096)
                y_test = labels  # (500,)

                pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
                clf.fit(X_train_pca, y_train)
                acc = clf.score(X_test_pca, y_test)
                fold_accs.append(round(float(acc), 4))

            results[tp][ln] = {
                "mean": round(float(np.mean(fold_accs)), 4),
                "std": round(float(np.std(fold_accs)), 4),
                "per_fold": fold_accs,
            }
            print(f"  {condition_name} | {tp:15s} | {ln:3s} | "
                  f"mean={results[tp][ln]['mean']:.4f} std={results[tp][ln]['std']:.4f}",
                  flush=True)
    return results


def main():
    t0 = time.time()

    print("Loading trained features...", flush=True)
    trained_data, layer_names, labels = load_all_features(TRAINED_PATH)
    print(f"  Done. Shape per layer: {trained_data['image_mean'][layer_names[0]].shape}", flush=True)

    print("Loading untrained features...", flush=True)
    untrained_data, _, _ = load_all_features(UNTRAINED_PATH)
    print(f"  Done.", flush=True)

    print("\n=== LOOCV (10-fold, leave-one-task-out) ===", flush=True)
    print("  Classifying instruction identity (10-class) across tasks\n", flush=True)

    print("--- Trained ---", flush=True)
    trained_results = run_loocv(trained_data, layer_names, labels, "trained")

    print("\n--- Untrained ---", flush=True)
    untrained_results = run_loocv(untrained_data, layer_names, labels, "untrained")

    output = {
        "trained": trained_results,
        "untrained": untrained_results,
        "n_folds": 10,
        "samples_per_fold_train": 4500,
        "samples_per_fold_test": 500,
        "pca_components": PCA_COMPONENTS,
        "layers": layer_names,
        "classifier": "LogisticRegression(max_iter=2000, C=1.0)",
        "task": "instruction_classification_10class",
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
