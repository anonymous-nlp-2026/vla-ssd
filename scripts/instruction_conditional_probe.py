"""
Instruction-Conditional Probe: Goal Classification with Linear Probe

10-class goal classification to test whether VLA representations encode
instruction-goal information. Compares last_preaction (instruction+image fused)
vs image_mean (vision only) across layers and training conditions.
"""
import argparse
import json
import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURE_ROOT = "./features"
RESULT_DIR = "./results/qa_probe"
RESULT_PATH = os.path.join(RESULT_DIR, "instruction_conditional_results.json")

TRAINED_DIR = os.path.join(FEATURE_ROOT, "trained_libero_goal")
UNTRAINED_DIR = os.path.join(FEATURE_ROOT, "untrained_libero_goal")
DINO_DIR = os.path.join(FEATURE_ROOT, "dinov2_libero_goal")

LAYER_INDICES = [0, 4, 8, 12, 16, 20, 24, 28, 32]
LAYER_NAMES = [f"L{i}" for i in LAYER_INDICES]

TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]

SEED = 42
PCA_DIM = 256


def load_vla_features(h5_path, layer_idx, agg_type="image_mean"):
    features_per_demo = []
    with h5py.File(h5_path, "r") as f:
        demo_keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
        for dk in demo_keys:
            feat = f[dk][agg_type][:, layer_idx, :]
            features_per_demo.append(feat)
    return features_per_demo


def load_dino_features(h5_path):
    features_per_demo = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
        for tk in traj_keys:
            feat = f[tk]["cls_token"][:]
            features_per_demo.append(feat)
    return features_per_demo


def run_goal_classification(all_features, all_labels):
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return round(acc, 4), round(f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="all",
                        help="Run condition: all (default), or dry-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 tasks x 1 layer x 1 condition for validation")
    args = parser.parse_args()

    dry_run = args.dry_run or args.condition == "dry-run"
    tasks_to_use = TASKS[:2] if dry_run else TASKS
    layers_idx = LAYER_INDICES[:1] if dry_run else LAYER_INDICES
    layers_nm = LAYER_NAMES[:1] if dry_run else LAYER_NAMES

    if dry_run:
        print(f"[DRY-RUN] 2 tasks, 1 layer ({layers_nm[0]}), trained only")

    print(f"PCA dimensionality reduction: {PCA_DIM}d")
    print(f"LogisticRegression max_iter: 2000")

    results = {
        "method": "Goal classification with Linear probe (PCA-256 + LogisticRegression)",
        "purpose": "Test instruction-visual fusion: last_preaction should outperform image_mean if instruction is utilized",
        "pca_dim": PCA_DIM,
        "conditions": {},
        "key_comparison": {},
    }

    conditions = [
        ("trained", TRAINED_DIR),
        ("untrained", UNTRAINED_DIR),
    ]
    if dry_run:
        conditions = conditions[:1]

    agg_types = ["image_mean", "last_preaction"]

    for cond_name, feat_dir in conditions:
        for agg in agg_types:
            cond_key = f"{cond_name}_{agg}"
            print(f"\n{'='*60}")
            print(f"Condition: {cond_key}")
            print(f"{'='*60}")

            layer_results = {}
            for li, ln in zip(layers_idx, layers_nm):
                all_features = []
                all_labels = []
                for task_idx, task_name in enumerate(tasks_to_use):
                    h5_path = os.path.join(feat_dir, f"{task_name}.h5")
                    feats = load_vla_features(h5_path, li, agg)
                    for demo_feat in feats:
                        all_features.append(demo_feat)
                        all_labels.append(
                            np.full(len(demo_feat), task_idx, dtype=int)
                        )

                acc, f1 = run_goal_classification(all_features, all_labels)
                layer_results[ln] = {"acc": acc, "f1": f1}
                print(f"  {ln}: acc={acc:.4f}, f1={f1:.4f}")

            results["conditions"][cond_key] = layer_results

    # DINOv2 baseline
    print(f"\n{'='*60}")
    print("Condition: dino_cls")
    print(f"{'='*60}")
    all_features = []
    all_labels = []
    for task_idx, task_name in enumerate(tasks_to_use):
        h5_path = os.path.join(DINO_DIR, f"{task_name}.h5")
        feats = load_dino_features(h5_path)
        for demo_feat in feats:
            all_features.append(demo_feat)
            all_labels.append(np.full(len(demo_feat), task_idx, dtype=int))
    acc, f1 = run_goal_classification(all_features, all_labels)
    results["conditions"]["dino_cls"] = {"acc": acc, "f1": f1}
    print(f"  dino_cls: acc={acc:.4f}, f1={f1:.4f}")

    # Key comparisons (full run only)
    if not dry_run:
        for cond_name in ["trained", "untrained"]:
            for agg in agg_types:
                cond_key = f"{cond_name}_{agg}"
                ld = results["conditions"][cond_key]
                best_layer = max(ld, key=lambda k: ld[k]["acc"])
                results["key_comparison"][f"{cond_key}_best"] = {
                    "layer": best_layer, **ld[best_layer]
                }

        for cond_name in ["trained", "untrained"]:
            lp = results["key_comparison"][f"{cond_name}_last_preaction_best"]["acc"]
            im = results["key_comparison"][f"{cond_name}_image_mean_best"]["acc"]
            results["key_comparison"][f"{cond_name}_delta_lastpre_minus_imgmean"] = round(lp - im, 4)

        results["key_comparison"]["dino"] = results["conditions"]["dino_cls"]

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
