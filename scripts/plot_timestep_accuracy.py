"""plot_timestep_accuracy.py — Visualize per-timestep goal classification accuracy.

Reads timestep_analysis.json from each model's results directory and plots
accuracy curves by relative timestep bin, with chance level and 25% gate
reference lines.

Usage:
  python scripts/plot_timestep_accuracy.py \
    --trained_dir ./results/trained_libero_goal/ \
    --untrained_dir ./results/untrained_libero_goal/ \
    --dino_dir ./results/dinov2_libero_goal/ \
    --output_dir ./results/figures/

  # Specific layer instead of best:
  python scripts/plot_timestep_accuracy.py \
    --trained_dir ... --dino_dir ... --layer 16 --output_dir ...
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


COLORS = {
    "trained": "#2196F3",
    "untrained": "#FF9800",
    "dinov2": "#4CAF50",
}
MARKERS = {"trained": "o", "untrained": "s", "dinov2": "^"}
LABELS = {"trained": "Trained VLA", "untrained": "Untrained VLA", "dinov2": "DINOv2"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-timestep goal classification accuracy")
    p.add_argument("--trained_dir", default=None,
                   help="Results dir for trained VLA")
    p.add_argument("--untrained_dir", default=None,
                   help="Results dir for untrained VLA")
    p.add_argument("--dino_dir", default=None,
                   help="Results dir for DINO-V2")
    p.add_argument("--output_dir",
                   default="./results/figures/",
                   help="Output directory for figures")
    p.add_argument("--layer", default="best",
                   help="Layer to plot: 'best' (per-model best) or integer")
    return p.parse_args()


def load_timestep_analysis(result_dir):
    path = os.path.join(result_dir, "goal_classification",
                        "timestep_analysis.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_best_layer(data):
    layers = data.get("layers", {})
    if not layers:
        return None
    return max(layers, key=lambda k: layers[k]["overall_accuracy"])


def plot_timestep_curves(model_data, layer_choice, output_dir):
    """Main plot: accuracy vs relative timestep bin."""
    fig, ax = plt.subplots(figsize=(8, 5))

    chance = None
    n_bins = None

    for model_name in ["trained", "untrained", "dinov2"]:
        data = model_data.get(model_name)
        if data is None:
            continue

        if layer_choice == "best":
            layer_key = get_best_layer(data)
        else:
            layer_key = str(layer_choice)

        layers = data.get("layers", {})
        if layer_key not in layers:
            print(f"WARNING: layer {layer_key} not found for {model_name}")
            continue

        lr = layers[layer_key]
        bin_edges = lr["bin_edges"]
        bin_accs = lr["bin_accuracies"]
        n_bins = len(bin_accs)
        chance = lr.get("chance_level", data.get("chance_level", 0.1))

        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2
                       for i in range(n_bins)]

        valid_x = [bin_centers[i] for i in range(n_bins)
                    if bin_accs[i] is not None]
        valid_y = [bin_accs[i] for i in range(n_bins)
                   if bin_accs[i] is not None]

        label_suffix = f" (L{layer_key})" if layer_choice == "best" else ""
        ax.plot(valid_x, valid_y,
                color=COLORS[model_name],
                marker=MARKERS[model_name],
                markersize=6,
                linewidth=2,
                label=f"{LABELS[model_name]}{label_suffix} "
                      f"(overall={lr['overall_accuracy']:.3f})")

    # Reference lines
    if chance is not None:
        ax.axhline(y=chance, color="gray", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Chance ({chance:.0%})")
    ax.axhline(y=0.25, color="red", linestyle=":", linewidth=1.5,
               alpha=0.7, label="25% gate")

    # Front 30% shading
    ax.axvspan(0, 0.3, alpha=0.08, color="green", zorder=0)
    ax.text(0.15, ax.get_ylim()[1] * 0.95, "front 30%",
            ha="center", va="top", fontsize=9, color="green", alpha=0.7)

    ax.set_xlabel("Relative timestep (t / T)", fontsize=12)
    ax.set_ylabel("Goal classification accuracy", fontsize=12)

    layer_desc = ("best per model" if layer_choice == "best"
                  else f"layer {layer_choice}")
    ax.set_title(f"Per-Timestep Goal Classification Accuracy ({layer_desc})",
                 fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    layer_tag = "best" if layer_choice == "best" else f"L{layer_choice}"
    base = f"timestep_goal_accuracy_{layer_tag}"
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{base}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, base)}.{{pdf,png}}")


def plot_front_vs_back(model_data, output_dir):
    """Bar chart: front-30% vs back-30% accuracy per model."""
    models = []
    front_accs = []
    back_accs = []

    for model_name in ["trained", "untrained", "dinov2"]:
        data = model_data.get(model_name)
        if data is None:
            continue
        layer_key = get_best_layer(data)
        if layer_key is None:
            continue
        lr = data["layers"][layer_key]
        fa = lr.get("front_30pct_accuracy")
        ba = lr.get("back_30pct_accuracy")
        if fa is None or ba is None:
            continue
        models.append(LABELS[model_name])
        front_accs.append(fa)
        back_accs.append(ba)

    if not models:
        print("No data for front-vs-back plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, front_accs, w, label="Front 30%", color="#66BB6A")
    ax.bar(x + w/2, back_accs, w, label="Back 30%", color="#EF5350")

    chance = model_data[list(model_data.keys())[0]].get("chance_level", 0.1)
    ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.7,
               label=f"Chance ({chance:.0%})")
    ax.axhline(y=0.25, color="red", linestyle=":", alpha=0.5,
               label="25% gate")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Goal classification accuracy")
    ax.set_title("Front 30% vs Back 30% Accuracy (best layer)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = "timestep_front_vs_back"
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"{base}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, base)}.{{pdf,png}}")


def main():
    args = parse_args()

    model_data = {}
    for name, path in [("trained", args.trained_dir),
                        ("untrained", args.untrained_dir),
                        ("dinov2", args.dino_dir)]:
        if path is None:
            continue
        data = load_timestep_analysis(path)
        if data is None:
            print(f"WARNING: No timestep_analysis.json in {path}")
            continue
        model_data[name] = data

    if not model_data:
        print("No timestep analysis data found. Run analyze_timestep_accuracy.py first.")
        sys.exit(1)

    print(f"Loaded data for: {', '.join(model_data.keys())}")

    layer = args.layer if args.layer == "best" else int(args.layer)
    plot_timestep_curves(model_data, layer, args.output_dir)
    plot_front_vs_back(model_data, args.output_dir)


if __name__ == "__main__":
    main()
