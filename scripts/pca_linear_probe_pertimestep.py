import h5py
import numpy as np
import json
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FEATURES_BASE = "./features"
RESULTS_DIR = "./results"
PERLAYER_RESULT = os.path.join(RESULTS_DIR, "pca1024_linear_perlayer.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "pca1024_linear_pertimestep.json")
N_DEMOS = 50
TRAIN_DEMOS = 40
PCA_DIM = 1024
N_BINS = 10
MIN_SAMPLES = 50

MODELS = {
    "trained_vla": {
        "dir": os.path.join(FEATURES_BASE, "trained_libero_goal"),
        "default_layer": 1,
        "feat_key": "last_preaction",
        "demo_prefix": "demo",
        "has_layer_dim": True,
    },
    "untrained_vla": {
        "dir": os.path.join(FEATURES_BASE, "untrained_libero_goal"),
        "default_layer": 1,
        "feat_key": "last_preaction",
        "demo_prefix": "demo",
        "has_layer_dim": True,
    },
    "dino": {
        "dir": os.path.join(FEATURES_BASE, "dinov2_libero_goal"),
        "default_layer": 0,
        "feat_key": "cls_token",
        "demo_prefix": "traj",
        "has_layer_dim": False,
    },
}


def get_best_layers():
    """Try reading best layers from Exp-1 results, fall back to defaults."""
    layers = {}
    if os.path.exists(PERLAYER_RESULT):
        with open(PERLAYER_RESULT) as f:
            perlayer = json.load(f)
        for model_name in MODELS:
            if model_name in perlayer and "best_layer" in perlayer[model_name]:
                layers[model_name] = perlayer[model_name]["best_layer"]
                print(f"  {model_name}: layer {layers[model_name]} (from Exp-1)")
        if len(layers) == len(MODELS):
            return layers
    for model_name, cfg in MODELS.items():
        if model_name not in layers:
            layers[model_name] = cfg["default_layer"]
            print(f"  {model_name}: layer {layers[model_name]} (default)")
    return layers


def load_features(cfg, layer_idx):
    """Load features for a given model config and layer, returning per-split data with timestep info."""
    X_train, y_train, t_train = [], [], []
    X_val, y_val, t_val = [], [], []

    h5_files = sorted(Path(cfg["dir"]).glob("*.h5"))
    prefix = cfg["demo_prefix"]

    for task_idx, h5_path in enumerate(h5_files):
        with h5py.File(h5_path, "r") as f:
            demo_keys = sorted(
                [k for k in f.keys() if k.startswith(prefix)],
                key=lambda x: int(x.split("_")[-1]),
            )
            for dk in demo_keys:
                demo_idx = int(dk.split("_")[-1])
                is_train = demo_idx < TRAIN_DEMOS

                feat_data = f[dk][cfg["feat_key"]][:]
                if cfg["has_layer_dim"]:
                    feats = feat_data[:, layer_idx, :]  # (T, dim)
                else:
                    feats = feat_data  # (T, dim)

                feats = feats.astype(np.float32)
                T = len(feats)
                labels = [task_idx] * T
                t_norm = [t / T for t in range(T)]

                if is_train:
                    X_train.append(feats)
                    y_train.extend(labels)
                    t_train.extend(t_norm)
                else:
                    X_val.append(feats)
                    y_val.extend(labels)
                    t_val.extend(t_norm)

    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    return (
        X_train, np.array(y_train), np.array(t_train),
        X_val, np.array(y_val), np.array(t_val),
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Best Layer Selection ===")
    best_layers = get_best_layers()

    results = {}
    for model_name, cfg in MODELS.items():
        layer = best_layers[model_name]
        print(f"\n--- {model_name} (layer {layer}) ---")

        print("  Loading features...")
        X_train, y_train, t_train, X_val, y_val, t_val = load_features(cfg, layer)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        if X_train.shape[1] > PCA_DIM:
            print(f"  PCA {X_train.shape[1]} -> {PCA_DIM}...")
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            ev = pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {ev:.4f}")
        else:
            print(f"  Dim={X_train.shape[1]} <= {PCA_DIM}, skip PCA")

        print("  Training LogisticRegression...")
        clf = LogisticRegression(
            max_iter=2000, random_state=42, solver="lbfgs",
            C=1.0,
        )
        clf.fit(X_train, y_train)

        overall_acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  Overall val accuracy: {overall_acc:.4f}")

        bin_results = {}
        for b in range(N_BINS):
            lo = b * 0.1
            hi = (b + 1) * 0.1
            if b == N_BINS - 1:
                mask = (t_val >= lo) & (t_val <= 1.0)
            else:
                mask = (t_val >= lo) & (t_val < hi)
            n = int(mask.sum())
            if n < MIN_SAMPLES:
                bin_results[str(b)] = {"accuracy": None, "n_samples": n, "status": "insufficient"}
            else:
                acc = accuracy_score(y_val[mask], clf.predict(X_val[mask]))
                bin_results[str(b)] = {"accuracy": round(float(acc), 4), "n_samples": n}

        results[model_name] = {
            "best_layer": layer,
            "overall_accuracy": round(float(overall_acc), 4),
            "per_bin": bin_results,
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print comparison table
    print("\n=== PCA-1024 + Linear Probe Per-Timestep ===")
    print(f"{'Bin':>5} {'Range':>12} {'Trained':>10} {'Untrained':>10} {'DINO':>10} {'D(T-U)':>10} {'n_train':>8} {'n_untr':>8} {'n_dino':>8}")
    print("-" * 90)
    for b in range(N_BINS):
        bs = str(b)
        t_res = results["trained_vla"]["per_bin"][bs]
        u_res = results["untrained_vla"]["per_bin"][bs]
        d_res = results["dino"]["per_bin"][bs]
        t_acc = t_res["accuracy"]
        u_acc = u_res["accuracy"]
        d_acc = d_res["accuracy"]
        delta = (t_acc - u_acc) * 100 if t_acc is not None and u_acc is not None else None
        lo = b * 0.1
        hi = (b + 1) * 0.1
        rng = f"[{lo:.1f},{hi:.1f})"
        t_str = f"{t_acc:.4f}" if t_acc is not None else "N/A"
        u_str = f"{u_acc:.4f}" if u_acc is not None else "N/A"
        d_str = f"{d_acc:.4f}" if d_acc is not None else "N/A"
        delta_str = f"{delta:+.1f}pp" if delta is not None else "N/A"
        print(f"{b:>5} {rng:>12} {t_str:>10} {u_str:>10} {d_str:>10} {delta_str:>10} {t_res['n_samples']:>8} {u_res['n_samples']:>8} {d_res['n_samples']:>8}")

    # Gate check
    t0_t = results["trained_vla"]["per_bin"]["0"]["accuracy"]
    t0_u = results["untrained_vla"]["per_bin"]["0"]["accuracy"]
    if t0_t is not None and t0_u is not None:
        delta0 = (t0_t - t0_u) * 100
        gate = "PASS" if delta0 > 15 else "FAIL"
        print(f"\n=== Gate: t=0 bin D(trained-untrained) = {delta0:+.1f}pp {'>' if delta0 > 15 else '<='} 15pp => {gate} ===")
    else:
        print("\n=== Gate: insufficient samples at t=0 bin ===")

    # Overall summary
    print(f"\nOverall: trained={results['trained_vla']['overall_accuracy']:.4f}, "
          f"untrained={results['untrained_vla']['overall_accuracy']:.4f}, "
          f"dino={results['dino']['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
