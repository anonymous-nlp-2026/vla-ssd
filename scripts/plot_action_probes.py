"""plot_action_probes.py — Visualize action prediction probe results.

Generates 4 figures comparing trained/untrained/DINO action probes.
Saves PDF + PNG to figures directory.

Usage:
  python scripts/plot_action_probes.py \
    --results_base ./results \
    --output_dir ./results/figures/
"""

import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


MODELS = ["trained", "untrained", "dinov2"]
PROBE_TYPES = ["action_prediction", "action_delta"]
ACTION_DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

COLORS = {
    "trained": "#2196F3",
    "untrained": "#FF9800",
    "dinov2": "#4CAF50",
}
MARKERS = {"trained": "o", "untrained": "s", "dinov2": "^"}
LABELS = {"trained": "Trained VLA", "untrained": "Untrained VLA", "dinov2": "DINOv2"}

COPY_BASELINE_R2 = 0.97
COPY_BASELINE_DELTA_R2 = 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_base", default="./results")
    p.add_argument("--output_dir",
                   default="./results/figures/")
    return p.parse_args()


def load_results(results_dir, probe_type):
    probe_dir = os.path.join(results_dir, probe_type)
    if not os.path.isdir(probe_dir):
        return None
    summary_path = os.path.join(probe_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    results = {}
    for p in sorted(glob.glob(os.path.join(probe_dir, "layer_*.json"))):
        with open(p) as f:
            m = json.load(f)
        key = str(m.get("layer", Path(p).stem.replace("layer_", "")))
        results[key] = m
    if results:
        return {"results": results}
    return None


def get_layers_and_metric(data, metric="overall_r2"):
    if data is None:
        return np.array([]), np.array([])
    results = data.get("results", {})
    items = []
    for k, v in results.items():
        try:
            layer = int(k)
        except ValueError:
            continue
        val = v.get(metric)
        if val is not None:
            items.append((layer, float(val)))
    items.sort()
    if not items:
        return np.array([]), np.array([])
    layers, vals = zip(*items)
    return np.array(layers), np.array(vals)


def get_best_layer_info(data):
    if data is None:
        return None, None
    results = data.get("results", {})
    best_layer, best_r2 = None, -np.inf
    for k, v in results.items():
        r2 = v.get("overall_r2", -np.inf)
        if r2 > best_r2:
            best_r2 = r2
            try:
                best_layer = int(k)
            except ValueError:
                best_layer = k
    return best_layer, best_r2


def get_per_dim_r2(data, layer=None):
    if data is None:
        return None
    results = data.get("results", {})
    if layer is not None:
        entry = results.get(str(layer))
        return entry.get("per_dim_r2") if entry else None
    bl, _ = get_best_layer_info(data)
    if bl is not None and str(bl) in results:
        return results[str(bl)].get("per_dim_r2")
    return None


def get_per_task_r2_all_layers(data):
    if data is None:
        return {}, []
    results = data.get("results", {})
    layers = sorted([int(k) for k in results.keys() if k.isdigit()])
    all_tasks = set()
    for v in results.values():
        all_tasks.update((v.get("per_task_r2") or {}).keys())
    all_tasks = sorted(all_tasks)
    return {
        layer: results[str(layer)].get("per_task_r2", {})
        for layer in layers
    }, layers, all_tasks


def short_task_name(full_name):
    parts = full_name.split("_", 2)
    if len(parts) > 2:
        desc = parts[2].replace("_", " ")
        if len(desc) > 35:
            desc = desc[:32] + "..."
        return desc
    return full_name


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved {name}.pdf/.png")
    plt.close(fig)


def plot_layer_r2_action(all_data, output_dir):
    """Fig 1: per-layer R²(action) curves — trained vs untrained vs DINO."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in ["trained", "untrained"]:
        if model not in all_data:
            continue
        layers, vals = get_layers_and_metric(all_data[model], "overall_r2")
        if len(layers) == 0:
            continue
        ax.plot(layers, vals, marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model], linewidth=2, markersize=4, zorder=3)

    if "dinov2" in all_data:
        _, dino_r2 = get_best_layer_info(all_data["dinov2"])
        if dino_r2 is not None:
            ax.axhline(y=dino_r2, color=COLORS["dinov2"], linestyle="--",
                        linewidth=1.5, label=f"{LABELS['dinov2']} (R²={dino_r2:.3f})",
                        zorder=2)

    ax.axhline(y=COPY_BASELINE_R2, color="red", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Copy baseline (R²={COPY_BASELINE_R2})", zorder=1)

    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (action prediction)")
    ax.set_title("Action Prediction Probe: Per-Layer R²")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    save_fig(fig, output_dir, "fig1_action_prediction_layer_r2")


def plot_layer_r2_delta(all_data_delta, output_dir):
    """Fig 2: per-layer R²(action-delta) curves."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in ["trained", "untrained"]:
        if model not in all_data_delta:
            continue
        layers, vals = get_layers_and_metric(all_data_delta[model], "overall_r2")
        if len(layers) == 0:
            continue
        ax.plot(layers, vals, marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model], linewidth=2, markersize=4, zorder=3)

    ax.axhline(y=COPY_BASELINE_DELTA_R2, color="grey", linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Copy baseline (R²={COPY_BASELINE_DELTA_R2})", zorder=1)

    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (action delta)")
    ax.set_title("Action Delta Probe: Per-Layer R²")
    ax.legend(frameon=False)

    save_fig(fig, output_dir, "fig2_action_delta_layer_r2")


def plot_per_dim_r2(all_data, output_dir):
    """Fig 3: per-dim R² bar chart at best layer."""
    dim_data = {}
    best_layers = {}
    for model in MODELS:
        if model in all_data:
            bl, _ = get_best_layer_info(all_data[model])
            best_layers[model] = bl
            dim_data[model] = get_per_dim_r2(all_data[model])

    available = [m for m in MODELS if m in dim_data and dim_data[m]]
    if not available:
        print("Skipping Fig 3: no per-dim data")
        return

    x = np.arange(len(ACTION_DIMS))
    n = len(available)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(available):
        vals = [dim_data[model].get(d, 0) for d in ACTION_DIMS]
        bl = best_layers[model]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=f"{LABELS[model]} (L{bl})",
               color=COLORS[model], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_DIMS)
    ax.set_ylabel("R²")
    ax.set_title("Per-Dimension R² at Best Layer")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    save_fig(fig, output_dir, "fig3_per_dim_r2")


def plot_per_task_heatmap(all_data, output_dir):
    """Fig 4: per-task R² heatmap, layers × tasks, trained + untrained side by side."""
    models_to_plot = [m for m in ["trained", "untrained"] if m in all_data]
    if not models_to_plot:
        print("Skipping Fig 4: no trained/untrained data")
        return

    ncols = len(models_to_plot)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)

    all_vals = []
    for model in models_to_plot:
        task_by_layer, layers, tasks = get_per_task_r2_all_layers(all_data[model])
        for layer in layers:
            for t in tasks:
                v = task_by_layer.get(layer, {}).get(t)
                if v is not None:
                    all_vals.append(v)

    if not all_vals:
        print("Skipping Fig 4: no per-task data")
        plt.close(fig)
        return

    vmin, vmax = min(all_vals), max(all_vals)

    for idx, model in enumerate(models_to_plot):
        ax = axes[0, idx]
        task_by_layer, layers, tasks = get_per_task_r2_all_layers(all_data[model])

        if not layers or not tasks:
            ax.set_title(f"{LABELS[model]} — no data")
            continue

        matrix = np.full((len(tasks), len(layers)), np.nan)
        for j, layer in enumerate(layers):
            for i, task in enumerate(tasks):
                v = task_by_layer.get(layer, {}).get(task)
                if v is not None:
                    matrix[i, j] = v

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels([short_task_name(t) for t in tasks], fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(LABELS[model])

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="R²")
    fig.suptitle("Per-Task R² Heatmap (Action Prediction)", fontsize=14, y=1.02)

    save_fig(fig, output_dir, "fig4_per_task_heatmap")


def main():
    args = parse_args()
    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    all_data = {}
    for model in MODELS:
        d = load_results(os.path.join(args.results_base, model), "action_prediction")
        if d is not None:
            all_data[model] = d
            print(f"Loaded {model}/action_prediction: "
                  f"{len(d.get('results', {}))} layers")

    all_data_delta = {}
    for model in MODELS:
        d = load_results(os.path.join(args.results_base, model), "action_delta")
        if d is not None:
            all_data_delta[model] = d
            print(f"Loaded {model}/action_delta: "
                  f"{len(d.get('results', {}))} layers")

    if all_data:
        plot_layer_r2_action(all_data, args.output_dir)
        plot_per_dim_r2(all_data, args.output_dir)
        plot_per_task_heatmap(all_data, args.output_dir)
    else:
        print("No action_prediction data found. Skipping Figs 1, 3, 4.")

    if all_data_delta:
        plot_layer_r2_delta(all_data_delta, args.output_dir)
    else:
        print("No action_delta data found. Skipping Fig 2.")

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
