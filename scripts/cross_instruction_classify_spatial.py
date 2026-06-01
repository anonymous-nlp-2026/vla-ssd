"""Cross-instruction classification for LIBERO-Spatial (all layers, both token positions)."""

import json
import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = "./results/cross_instruction_spatial/"
TRAINED_PATH = os.path.join(BASE_DIR, "trained_cross_inst_L0-L31.h5")
UNTRAINED_PATH = os.path.join(BASE_DIR, "untrained_cross_inst_L0-L31.h5")
OUTPUT_PATH = os.path.join(BASE_DIR, "classification_results.json")

SEED = 42
N_TASKS = 10
N_FRAMES = 50
N_INSTRUCTIONS = 10
TRAIN_TASKS = list(range(8))
TEST_TASKS = list(range(8, 10))

REPORT_LAYERS = [0, 1, 4, 8, 12, 16, 20, 24, 28, 31]
TOKEN_POSITIONS = ["image_mean", "last_preaction"]


def load_features(h5_path, position):
    print(f"Loading {h5_path} [{position}]...", flush=True)
    f = h5py.File(h5_path, "r")
    layers = list(f.attrs.get("layers", list(range(32))))
    if "metadata/layers" in f:
        layers = f["metadata/layers"][:].tolist()
    layer_names = [f"L{l}" for l in layers]

    features = {ln: {"train_X": [], "train_y": [], "test_X": [], "test_y": []}
                for ln in layer_names}

    for task_id in range(N_TASKS):
        split = "train" if task_id in TRAIN_TASKS else "test"
        for frame_id in range(N_FRAMES):
            for inst_id in range(N_INSTRUCTIONS):
                key = f"task_{task_id}/frame_{frame_id}/instruction_{inst_id}/{position}"
                data = f[key][:]
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


def main():
    np.random.seed(SEED)

    conditions = [("trained", TRAINED_PATH), ("untrained", UNTRAINED_PATH)]
    for name, path in conditions:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found ({name})")

    all_results = {}

    for cond_name, cond_path in conditions:
        for position in TOKEN_POSITIONS:
            key = f"{cond_name}_{position}"
            print(f"\n=== {key.upper()} ===", flush=True)
            features, layer_names = load_features(cond_path, position)
            report_layer_names = [f"L{l}" for l in REPORT_LAYERS if f"L{l}" in layer_names]
            results = classify(features, report_layer_names)
            all_results[key] = results
            del features

    all_results["chance"] = 0.1
    all_results["metadata"] = {
        "n_train_samples": len(TRAIN_TASKS) * N_FRAMES * N_INSTRUCTIONS,
        "n_test_samples": len(TEST_TASKS) * N_FRAMES * N_INSTRUCTIONS,
        "n_classes": N_INSTRUCTIONS,
        "report_layers": [f"L{l}" for l in REPORT_LAYERS],
        "train_tasks": TRAIN_TASKS,
        "test_tasks": TEST_TASKS,
        "pca_components": 256,
        "classifier": "LogisticRegression(C=1.0, max_iter=2000)",
        "seed": SEED,
    }

    with open(OUTPUT_PATH, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}", flush=True)

    print("\n=== LIBERO-Spatial Cross-Instruction 10-way Classification ===", flush=True)
    report_ln = [f"L{l}" for l in REPORT_LAYERS]
    header = f"{'Layer':<6} {'trained_img_mean':<18} {'trained_last_pre':<18} {'untrained_img_mean':<20} {'untrained_last_pre':<18}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for ln in report_ln:
        t_im = all_results.get("trained_image_mean", {}).get(ln, -1)
        t_lp = all_results.get("trained_last_preaction", {}).get(ln, -1)
        u_im = all_results.get("untrained_image_mean", {}).get(ln, -1)
        u_lp = all_results.get("untrained_last_preaction", {}).get(ln, -1)
        print(f"{ln:<6} {t_im:<18.4f} {t_lp:<18.4f} {u_im:<20.4f} {u_lp:<18.4f}", flush=True)
    print(f"{'Chance':<6} {'0.1000':<18}", flush=True)


if __name__ == "__main__":
    main()
