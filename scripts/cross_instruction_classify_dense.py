"""Cross-instruction classification probe for dense L0-L8 features.

10-way classification: predict which instruction was given from VLA features.
Train on tasks 0-7, test on tasks 8-9 (cross-task generalization).
Only last_preaction position (image_mean is always at chance).
"""

import json
import time
import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAINED_PATH = "./results/cross_instruction_dense/trained_cross_inst_dense.h5"
UNTRAINED_PATH = "./results/cross_instruction_dense/untrained_cross_inst_dense.h5"
OUTPUT_PATH = "./results/cross_instruction_dense/classification_results.json"

SEED = 42
N_TASKS = 10
N_FRAMES = 50
N_INSTRUCTIONS = 10
TRAIN_TASKS = list(range(8))
TEST_TASKS = list(range(8, 10))


def load_features(h5_path):
    """Load last_preaction features for all layers.
    
    Returns:
        features: dict[layer_name] = {'train_X': ndarray, 'train_y': ndarray, 'test_X': ndarray, 'test_y': ndarray}
        layer_names: list of 'L0', 'L1', ..., 'L8'
    """
    print(f"Loading {h5_path}...", flush=True)
    f = h5py.File(h5_path, "r")
    layers = list(f.attrs.get("layers", [0, 1, 2, 3, 4, 5, 6, 7, 8]))
    if "metadata/layers" in f:
        layers = f["metadata/layers"][:].tolist()
    layer_names = [f"L{l}" for l in layers]
    n_layers = len(layers)

    features = {ln: {"train_X": [], "train_y": [], "test_X": [], "test_y": []}
                for ln in layer_names}

    for task_id in range(N_TASKS):
        split = "train" if task_id in TRAIN_TASKS else "test"
        for frame_id in range(N_FRAMES):
            for inst_id in range(N_INSTRUCTIONS):
                key = f"task_{task_id}/frame_{frame_id}/instruction_{inst_id}/last_preaction"
                data = f[key][:]  # shape (n_layers, 4096)
                for li, ln in enumerate(layer_names):
                    features[ln][f"{split}_X"].append(data[li])
                    features[ln][f"{split}_y"].append(inst_id)

    for ln in layer_names:
        for key in ["train_X", "test_X"]:
            features[ln][key] = np.array(features[ln][key], dtype=np.float32)
        for key in ["train_y", "test_y"]:
            features[ln][key] = np.array(features[ln][key], dtype=np.int64)

    f.close()
    n_train = features[layer_names[0]]["train_X"].shape[0]
    n_test = features[layer_names[0]]["test_X"].shape[0]
    print(f"  Loaded. Train: {n_train}, Test: {n_test}", flush=True)
    return features, layer_names


def classify(features, layer_names):
    """Run PCA + LogReg for each layer."""
    results = {}
    for ln in layer_names:
        d = features[ln]
        train_X, train_y = d["train_X"], d["train_y"]
        test_X, test_y = d["test_X"], d["test_y"]

        n_components = min(256, train_X.shape[0], train_X.shape[1])
        pca = PCA(n_components=n_components, random_state=SEED)
        train_X_pca = pca.fit_transform(train_X)
        test_X_pca = pca.transform(test_X)

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, n_jobs=-1)
        clf.fit(train_X_pca, train_y)
        pred = clf.predict(test_X_pca)
        acc = accuracy_score(test_y, pred)
        results[ln] = round(float(acc), 4)
        print(f"  {ln}: {acc:.4f}", flush=True)

    return results


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["all", "llama2base"], default="all",
                   help="'all' runs trained+untrained; 'llama2base' adds llama2base comparison")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)

    BASE_DIR = "./results/cross_instruction_dense/"
    LLAMA2BASE_PATH = os.path.join(BASE_DIR, "llama2base_cross_inst_dense.h5")

    conditions_to_run = []
    if args.condition == "all":
        conditions_to_run = [("trained", TRAINED_PATH), ("untrained", UNTRAINED_PATH)]
    elif args.condition == "llama2base":
        conditions_to_run = [("trained", TRAINED_PATH), ("untrained", UNTRAINED_PATH), ("llama2base", LLAMA2BASE_PATH)]

    for path_name, path in conditions_to_run:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found ({path_name})")

    all_results = {}
    layer_names = None
    for cond_name, cond_path in conditions_to_run:
        print(f"\n=== {cond_name.upper()} MODEL (last_preaction) ===", flush=True)
        features, ln = load_features(cond_path)
        if layer_names is None:
            layer_names = ln
        results = classify(features, layer_names)
        all_results[f"{cond_name}_last_preaction"] = results
        del features

    all_results["chance"] = 0.1
    all_results["metadata"] = {
        "n_train_samples": len(TRAIN_TASKS) * N_FRAMES * N_INSTRUCTIONS,
        "n_test_samples": len(TEST_TASKS) * N_FRAMES * N_INSTRUCTIONS,
        "n_classes": N_INSTRUCTIONS,
        "layers": layer_names,
        "train_tasks": TRAIN_TASKS,
        "test_tasks": TEST_TASKS,
        "pca_components": 256,
        "classifier": "LogisticRegression(C=1.0, max_iter=2000)",
        "seed": SEED,
        "conditions": [c[0] for c in conditions_to_run],
    }

    output_suffix = "_with_llama2base" if args.condition == "llama2base" else ""
    output_path = os.path.join(BASE_DIR, f"classification_results{output_suffix}.json")
    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)

    print("\n=== SUMMARY TABLE ===", flush=True)
    cond_names = [c[0] for c in conditions_to_run]
    header = f"{'Layer':<6}" + "".join(f" {c:<12}" for c in cond_names)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for ln in layer_names:
        row = f"{ln:<6}"
        for c in cond_names:
            row += f" {all_results[f'{c}_last_preaction'][ln]:<12.4f}"
        print(row, flush=True)
    print(f"{'Chance':<6} {'0.1000':<12}", flush=True)


if __name__ == "__main__":
    main()
