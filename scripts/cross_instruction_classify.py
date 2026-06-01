"""Cross-instruction classification probe.

10-way classification: predict which instruction was given from VLA features.
Train on tasks 0-7, test on tasks 8-9 (cross-task generalization).
"""

import json
import time
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAINED_PATH = "./results/cross_instruction/trained_cross_inst.h5"
UNTRAINED_PATH = "./results/cross_instruction/untrained_cross_inst.h5"
OUTPUT_PATH = "./results/cross_instruction/classification_results.json"

SEED = 42
N_TASKS = 10
N_FRAMES = 50
N_INSTRUCTIONS = 10
TRAIN_TASKS = list(range(8))
TEST_TASKS = list(range(8, 10))
TOKEN_POSITIONS = ["image_mean", "last_preaction"]


def wait_for_file(path, timeout=300, interval=30):
    import os
    start = time.time()
    while not os.path.exists(path):
        elapsed = time.time() - start
        if elapsed > timeout:
            raise FileNotFoundError(f"{path} not found after {timeout}s")
        print(f"Waiting for {path}... ({elapsed:.0f}s)", flush=True)
        time.sleep(interval)


def load_features(h5_path):
    """Load features into dict[token_pos][layer_idx] -> (X, y) arrays.
    
    Returns:
        features: dict with structure features[token_pos][layer_idx] = {
            'train_X': ndarray, 'train_y': ndarray,
            'test_X': ndarray, 'test_y': ndarray
        }
        layer_names: list of layer name strings like 'L0', 'L8', etc.
    """
    print(f"Loading {h5_path}...", flush=True)
    f = h5py.File(h5_path, "r")
    layers = f["metadata/layers"][:]
    layer_names = [f"L{l}" for l in layers]
    n_layers = len(layers)

    features = {tp: {ln: {"train_X": [], "train_y": [], "test_X": [], "test_y": []}
                      for ln in layer_names}
                for tp in TOKEN_POSITIONS}

    for task_id in range(N_TASKS):
        split = "train" if task_id in TRAIN_TASKS else "test"
        for frame_id in range(N_FRAMES):
            for inst_id in range(N_INSTRUCTIONS):
                grp = f[f"task_{task_id}/frame_{frame_id}/instruction_{inst_id}"]
                for tp in TOKEN_POSITIONS:
                    data = grp[tp][:]  # shape (n_layers, 4096)
                    for li, ln in enumerate(layer_names):
                        features[tp][ln][f"{split}_X"].append(data[li])
                        features[tp][ln][f"{split}_y"].append(inst_id)

    # Convert to arrays
    for tp in TOKEN_POSITIONS:
        for ln in layer_names:
            for key in ["train_X", "test_X"]:
                features[tp][ln][key] = np.array(features[tp][ln][key], dtype=np.float32)
            for key in ["train_y", "test_y"]:
                features[tp][ln][key] = np.array(features[tp][ln][key], dtype=np.int64)

    f.close()
    print(f"  Loaded. Train: {features[TOKEN_POSITIONS[0]][layer_names[0]]['train_X'].shape[0]}, "
          f"Test: {features[TOKEN_POSITIONS[0]][layer_names[0]]['test_X'].shape[0]}", flush=True)
    return features, layer_names


def classify(features, layer_names):
    """Run PCA + LogReg for each token position and layer."""
    results = {tp: {} for tp in TOKEN_POSITIONS}

    for tp in TOKEN_POSITIONS:
        for ln in layer_names:
            d = features[tp][ln]
            train_X, train_y = d["train_X"], d["train_y"]
            test_X, test_y = d["test_X"], d["test_y"]

            # PCA
            n_components = min(256, train_X.shape[0], train_X.shape[1])
            pca = PCA(n_components=n_components, random_state=SEED)
            train_X_pca = pca.fit_transform(train_X)
            test_X_pca = pca.transform(test_X)

            # LogReg
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, n_jobs=-1)
            clf.fit(train_X_pca, train_y)
            pred = clf.predict(test_X_pca)
            acc = accuracy_score(test_y, pred)
            results[tp][ln] = round(float(acc), 4)
            print(f"  {tp} {ln}: {acc:.4f}", flush=True)

    return results


def main():
    np.random.seed(SEED)

    # Wait for files
    wait_for_file(TRAINED_PATH)
    wait_for_file(UNTRAINED_PATH)

    # Trained model
    print("\n=== TRAINED MODEL ===", flush=True)
    trained_features, layer_names = load_features(TRAINED_PATH)
    trained_results = classify(trained_features, layer_names)
    del trained_features

    # Untrained model
    print("\n=== UNTRAINED MODEL ===", flush=True)
    untrained_features, _ = load_features(UNTRAINED_PATH)
    untrained_results = classify(untrained_features, layer_names)

    # DINOv2 baseline: use untrained L0 image_mean
    # At L0 the untrained model just passes through projected visual tokens
    # without any language conditioning, so this approximates DINOv2 features
    dino_baseline = untrained_results["image_mean"]["L0"]
    print(f"\nDINOv2 baseline (untrained L0 image_mean): {dino_baseline:.4f}", flush=True)

    del untrained_features

    # Compile results
    n_train = len(TRAIN_TASKS) * N_FRAMES * N_INSTRUCTIONS
    n_test = len(TEST_TASKS) * N_FRAMES * N_INSTRUCTIONS
    output = {
        "trained": trained_results,
        "untrained": untrained_results,
        "dino_baseline": dino_baseline,
        "chance": 0.1,
        "metadata": {
            "n_train_frames": len(TRAIN_TASKS) * N_FRAMES,
            "n_test_frames": len(TEST_TASKS) * N_FRAMES,
            "n_train_samples": n_train,
            "n_test_samples": n_test,
            "n_classes": N_INSTRUCTIONS,
            "layers": layer_names,
            "train_tasks": TRAIN_TASKS,
            "test_tasks": TEST_TASKS,
            "pca_components": 256,
            "classifier": "LogisticRegression(C=1.0, max_iter=2000)",
            "seed": SEED,
        },
    }

    with open(OUTPUT_PATH, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
