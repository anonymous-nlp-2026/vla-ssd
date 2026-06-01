"""Classify cross-instruction features for full layer profile (L0-L31).

Label = instruction index (10 classes). Train/test split by task.
"""

import json
import os
import sys

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = "./results/cross_instruction_dense/"
TRAIN_TASKS = list(range(8))
TEST_TASKS = [8, 9]
PCA_COMPONENTS = 256
SEED = 42


def load_features_from_h5(h5_path, token_position="last_preaction"):
    """Load features. Returns per-layer arrays split by task membership.
    
    Returns: (train_X_per_layer, train_y_per_layer, test_X_per_layer, test_y_per_layer, layers)
    Each *_per_layer is a dict: {layer_idx: array}
    """
    f = h5py.File(h5_path, "r")
    layers = list(f.attrs['layers'])
    n_layers = len(layers)

    train_feats = {li: [] for li in range(n_layers)}
    train_labels = {li: [] for li in range(n_layers)}
    test_feats = {li: [] for li in range(n_layers)}
    test_labels = {li: [] for li in range(n_layers)}

    for key in sorted(f.keys()):
        if not key.startswith("task_"):
            continue
        task_idx = int(key.split("_")[1])
        task_grp = f[key]
        is_train = task_idx in TRAIN_TASKS

        for frame_key in sorted(task_grp.keys(), key=lambda x: int(x.split("_")[1])):
            frame_grp = task_grp[frame_key]
            for inst_key in sorted(frame_grp.keys(), key=lambda x: int(x.split("_")[1])):
                inst_idx = int(inst_key.split("_")[1])
                inst_grp = frame_grp[inst_key]
                feat = np.array(inst_grp[token_position], dtype=np.float32)

                for li in range(n_layers):
                    if is_train:
                        train_feats[li].append(feat[li])
                        train_labels[li].append(inst_idx)
                    else:
                        test_feats[li].append(feat[li])
                        test_labels[li].append(inst_idx)

    f.close()

    for li in range(n_layers):
        train_feats[li] = np.array(train_feats[li])
        train_labels[li] = np.array(train_labels[li])
        test_feats[li] = np.array(test_feats[li])
        test_labels[li] = np.array(test_labels[li])

    return train_feats, train_labels, test_feats, test_labels, layers


def classify_layer(train_X, train_y, test_X, test_y):
    pca = PCA(n_components=min(PCA_COMPONENTS, train_X.shape[1], train_X.shape[0]),
              random_state=SEED)
    train_X_pca = pca.fit_transform(train_X)
    test_X_pca = pca.transform(test_X)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(train_X_pca, train_y)
    acc = clf.score(test_X_pca, test_y)
    return round(acc, 4)


def main():
    condition = sys.argv[1] if len(sys.argv) > 1 else "llama2base"

    h5_dense = os.path.join(RESULTS_DIR, f"{condition}_cross_inst_dense.h5")
    h5_upper = os.path.join(RESULTS_DIR, f"{condition}_cross_inst_L9-L31.h5")

    results = {}

    # L0-L8
    print(f"Loading L0-L8 from: {h5_dense}")
    tr_f, tr_l, te_f, te_l, layers_08 = load_features_from_h5(h5_dense)
    print(f"  Layers: {layers_08}")
    print(f"  Train: {tr_f[0].shape}, Test: {te_f[0].shape}")
    print(f"  Train labels unique: {np.unique(tr_l[0])}")
    print(f"  Test labels unique: {np.unique(te_l[0])}")

    for li, layer_id in enumerate(layers_08):
        acc = classify_layer(tr_f[li], tr_l[li], te_f[li], te_l[li])
        results[f"L{layer_id}"] = acc
        print(f"  L{layer_id}: {acc}")

    del tr_f, tr_l, te_f, te_l

    # L9-L31
    print(f"\nLoading L9-L31 from: {h5_upper}")
    tr_f, tr_l, te_f, te_l, layers_931 = load_features_from_h5(h5_upper)
    print(f"  Layers: {layers_931}")
    print(f"  Train: {tr_f[0].shape}, Test: {te_f[0].shape}")

    for li, layer_id in enumerate(layers_931):
        acc = classify_layer(tr_f[li], tr_l[li], te_f[li], te_l[li])
        results[f"L{layer_id}"] = acc
        print(f"  L{layer_id}: {acc}")

    # Output
    output = {
        "condition": condition,
        "token_position": "last_preaction",
        "layers": results,
        "metadata": {
            "n_train_samples": int(tr_f[0].shape[0]),
            "n_test_samples": int(te_f[0].shape[0]),
            "n_classes": 10,
            "total_layers": len(layers_08) + len(layers_931),
            "train_tasks": TRAIN_TASKS,
            "test_tasks": TEST_TASKS,
            "pca_components": PCA_COMPONENTS,
            "classifier": "LogisticRegression(C=1.0, max_iter=2000)",
            "seed": SEED
        }
    }

    out_path = os.path.join(RESULTS_DIR, f"{condition}_full_layer_classification.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
