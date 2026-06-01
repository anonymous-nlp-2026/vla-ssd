"""analyze_results.py — Compare probe results across models and visualize.

Input:  Multiple result directories (from train_probes.py)
Output: Comparison tables, plots, PASS/FAIL judgment

Usage:
  python analyze_results.py \
    --results_dirs trained:results/trained/,untrained:results/untrained/,dinov2:results/dinov2/ \
    --output_dir results/analysis/
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


def parse_args():
    p = argparse.ArgumentParser(description="Analyze and compare probe results")
    p.add_argument("--results_dirs", required=True,
                   help="Comma-separated name:path pairs, e.g. trained:results/trained/")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--probe_type", default="temporal_distance",
                   choices=["temporal_distance", "subtask_predicate"])
    return p.parse_args()


def parse_results_dirs(s):
    d = {}
    for entry in s.split(","):
        name, path = entry.strip().split(":")
        d[name.strip()] = path.strip()
    return d


def load_results(results_dir, probe_type):
    """Load probe results from a directory."""
    probe_dir = os.path.join(results_dir, probe_type)
    summary_path = os.path.join(probe_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    results = {}
    for p in sorted(glob.glob(os.path.join(probe_dir, "layer_*.json"))):
        with open(p) as f:
            m = json.load(f)
        key = m.get("layer", Path(p).stem.replace("layer_", ""))
        results[str(key)] = m
    return {"results": results}


def load_oracle(results_dirs):
    """Auto-detect oracle results from any provided directory."""
    for path in results_dirs.values():
        oracle_dir = os.path.join(path, "oracle")
        summary = os.path.join(oracle_dir, "summary.json")
        if os.path.exists(summary):
            with open(summary) as f:
                data = json.load(f)
            r = data.get("results", {})
            if "oracle" in r:
                return r["oracle"].get("spearman_rho")
    return None


def _sort_key(x):
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


def extract_layer_metrics(data, metric="spearman_rho"):
    results = data.get("results", {})
    layers, values = [], []
    for k, v in sorted(results.items(), key=lambda x: _sort_key(x[0])):
        if k == "oracle":
            continue
        val = v.get(metric)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            layers.append(int(k))
            values.append(float(val))
    return np.array(layers), np.array(values)

def extract_per_task_mean_rho(data):
    results = data.get("results", {})
    layers, mean_rhos = [], []
    for k, v in sorted(results.items(), key=lambda x: _sort_key(x[0])):
        if k == "oracle":
            continue
        per_task = v.get("per_task_rho", {})
        if per_task:
            valid = [r for r in per_task.values()
                     if r is not None and not np.isnan(r)]
            if valid:
                layers.append(int(k))
                mean_rhos.append(float(np.mean(valid)))
    return np.array(layers), np.array(mean_rhos)



# ============================================================
# Plots
# ============================================================

COLORS = {
    "trained": "#2196F3", "untrained": "#F44336",
    "dinov2": "#4CAF50", "oracle": "#FF9800",
}
MARKERS = {"trained": "o", "untrained": "s", "dinov2": "^", "oracle": "D"}

# D004: DINO gate threshold (per-task mean ρ)
DINO_GATE_THRESHOLD = 0.901  # D006: rerun with updated pipeline (per_task_n_pairs=5000)


def plot_layer_rho(all_data, oracle_rho, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in all_data.items():
        layers, rhos = extract_layer_metrics(data, "spearman_rho")
        if len(layers) == 0:
            continue
        ax.plot(layers, rhos,
                marker=MARKERS.get(name, "o"),
                color=COLORS.get(name, "#999"),
                label=name, linewidth=2, markersize=5)

    if oracle_rho is not None:
        ax.axhline(y=oracle_rho, color=COLORS["oracle"],
                   linestyle="--", label=f"oracle (rho={oracle_rho:.3f})", alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Temporal Distance Probe: Layer-wise Spearman rho")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "layer_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_delta_rho(all_data, output_dir):
    if "trained" not in all_data or "untrained" not in all_data:
        print("Skipping delta-rho plot (need trained + untrained)")
        return

    l_tr, rho_tr = extract_layer_metrics(all_data["trained"], "spearman_rho")
    l_un, rho_un = extract_layer_metrics(all_data["untrained"], "spearman_rho")

    tr_map = dict(zip(l_tr.tolist(), rho_tr.tolist()))
    un_map = dict(zip(l_un.tolist(), rho_un.tolist()))
    common = sorted(set(l_tr.tolist()) & set(l_un.tolist()))
    if not common:
        return

    layers = np.array(common)
    deltas = np.array([tr_map[l] - un_map[l] for l in common])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3" if d >= 0 else "#F44336" for d in deltas]
    ax.bar(layers, deltas, color=colors, width=0.8)
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="threshold=0.1")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta-rho (trained - untrained)")
    ax.set_title("Temporal Distance Probe: Delta-rho per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "delta_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_subtask_auc_heatmap(data, output_dir, name="trained"):
    results = data.get("results", {})
    layers = sorted([int(k) for k in results if k != "oracle"])
    if not layers:
        return

    pred_names = None
    for v in results.values():
        if "predicate_names" in v:
            pred_names = v["predicate_names"]
            break
    if pred_names is None:
        return

    n_pred = len(pred_names)
    heatmap = np.full((len(layers), n_pred), np.nan)

    for li, layer in enumerate(layers):
        auc = results[str(layer)].get("auc_per_predicate", {})
        for k in range(n_pred):
            val = auc.get(str(k), np.nan)
            heatmap[li, k] = val

    fig, ax = plt.subplots(figsize=(max(8, n_pred), max(6, len(layers) * 0.3)))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n_pred))
    ax.set_xticklabels(pred_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Predicate")
    ax.set_ylabel("Layer")
    ax.set_title(f"Sub-task Predicate AUC ({name})")
    fig.colorbar(im, ax=ax, label="AUC-ROC")

    path = os.path.join(output_dir, f"subtask_auc_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_layer_rho_profile(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in ["trained", "untrained"]:
        if name not in all_data:
            continue
        layers, rhos = extract_layer_metrics(all_data[name], "spearman_rho")
        if len(layers) == 0:
            continue
        ax.plot(layers, rhos,
                marker=MARKERS.get(name, "o"),
                color=COLORS.get(name, "#999"),
                label=name, linewidth=2, markersize=5)

    if "dinov2" in all_data:
        _, rho_dino = extract_layer_metrics(all_data["dinov2"], "spearman_rho")
        if len(rho_dino) > 0:
            dino_val = float(rho_dino.max())
            ax.axhline(y=dino_val, color=COLORS.get("dinov2", "#4CAF50"),
                       linestyle="--", linewidth=2,
                       label=f"DINO-V2 (rho={dino_val:.3f})")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Per-layer rho Profile: VLA (trained/untrained) vs DINO-V2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "per_layer_rho_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Judgment
# ============================================================

def bootstrap_delta_rho(trained_results, untrained_results, best_layer, n_bootstrap=1000, seed=42):
    """Task-level bootstrap for delta-rho at best layer."""
    rng = np.random.RandomState(seed)

    trained_per_task = trained_results[best_layer].get("per_task_rho", {})
    untrained_per_task = untrained_results[best_layer].get("per_task_rho", {})

    task_ids = sorted(set(trained_per_task.keys()) & set(untrained_per_task.keys()))
    task_ids = [t for t in task_ids
                if not (np.isnan(trained_per_task[t]) or np.isnan(untrained_per_task[t]))]
    if len(task_ids) < 2:
        return 0.0, 0.0, 0.0, 1.0

    trained_rhos = np.array([trained_per_task[t] for t in task_ids])
    untrained_rhos = np.array([untrained_per_task[t] for t in task_ids])
    n_tasks = len(task_ids)
    observed_delta = float(np.mean(trained_rhos) - np.mean(untrained_rhos))

    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_tasks, n_tasks, replace=True)
        delta = np.mean(trained_rhos[idx]) - np.mean(untrained_rhos[idx])
        boot_deltas.append(delta)
    boot_deltas = sorted(boot_deltas)
    ci_lower = boot_deltas[int(0.025 * n_bootstrap)]
    ci_upper = boot_deltas[int(0.975 * n_bootstrap)]

    n_perm = 10000
    perm_deltas = []
    combined = np.concatenate([trained_rhos, untrained_rhos])
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_delta = np.mean(combined[:n_tasks]) - np.mean(combined[n_tasks:])
        perm_deltas.append(perm_delta)
    p_value = float(np.mean(np.array(perm_deltas) >= observed_delta))

    return observed_delta, ci_lower, ci_upper, p_value


def compute_judgment(all_data):
    """D002/D003/D004-compliant judgment: pure temporal distance, fixed DINO gate."""
    criteria = {}

    # 1) per-task mean rho at best layer > 0.5 (D002)
    # D006: detect collapsed layers (per-task mean rho < 0.1 in either model)
    collapsed_layers = set()
    for model_name in ["trained", "untrained"]:
        if model_name in all_data:
            c_layers, c_rhos = extract_per_task_mean_rho(all_data[model_name])
            for l, r in zip(c_layers.tolist(), c_rhos.tolist()):
                if r < 0.1:
                    collapsed_layers.add(int(l))

    if "trained" in all_data:
        layers_pt, rhos_pt = extract_per_task_mean_rho(all_data["trained"])
        if len(layers_pt) > 0:
            healthy_mask = np.array([int(l) not in collapsed_layers for l in layers_pt])
            healthy_layers = layers_pt[healthy_mask]
            healthy_rhos = rhos_pt[healthy_mask]
        else:
            healthy_layers, healthy_rhos = layers_pt, rhos_pt

        if len(healthy_rhos) > 0:
            best_rho = float(healthy_rhos.max())
            best_layer = int(healthy_layers[np.argmax(healthy_rhos)])
        elif len(layers_pt) > 0:
            best_rho = 0.0
            best_layer = -1
        else:
            layers_pooled, rhos_pooled = extract_layer_metrics(all_data["trained"], "spearman_rho")
            best_rho = float(rhos_pooled.max()) if len(rhos_pooled) > 0 else 0.0
            best_layer = int(layers_pooled[np.argmax(rhos_pooled)]) if len(rhos_pooled) > 0 else -1
        layers_pooled, rhos_pooled = extract_layer_metrics(all_data["trained"], "spearman_rho")
        best_pooled_rho = float(rhos_pooled.max()) if len(rhos_pooled) > 0 else 0.0
    else:
        best_rho, best_layer, best_pooled_rho = 0.0, -1, 0.0

    criteria["best_rho"] = best_rho
    criteria["best_pooled_rho"] = best_pooled_rho
    criteria["best_layer"] = best_layer
    criteria["collapsed_layers_excluded"] = sorted(collapsed_layers)
    criteria["rho_pass"] = best_rho > 0.5

    # 2) delta-rho at best layer with task-level bootstrap CI + permutation test
    delta_at_best = 0.0
    delta_ci_lower = 0.0
    delta_ci_upper = 0.0
    ci_excludes_zero = False
    p_value_perm = 1.0
    if "trained" in all_data and "untrained" in all_data:
        tr_results = all_data["trained"].get("results", {})
        un_results = all_data["untrained"].get("results", {})
        best_key = str(best_layer)
        has_per_task = (best_key in tr_results
                        and "per_task_rho" in tr_results.get(best_key, {})
                        and best_key in un_results
                        and "per_task_rho" in un_results.get(best_key, {}))
        if has_per_task:
            delta_at_best, delta_ci_lower, delta_ci_upper, p_value_perm = (
                bootstrap_delta_rho(tr_results, un_results, best_key))
            ci_excludes_zero = delta_ci_lower > 0
        else:
            l_tr, rho_tr = extract_layer_metrics(all_data["trained"], "spearman_rho")
            l_un, rho_un = extract_layer_metrics(all_data["untrained"], "spearman_rho")
            tr_map = dict(zip(l_tr.tolist(), rho_tr.tolist()))
            un_map = dict(zip(l_un.tolist(), rho_un.tolist()))
            if best_layer in tr_map and best_layer in un_map:
                delta_at_best = tr_map[best_layer] - un_map[best_layer]

    criteria["delta_at_best"] = delta_at_best
    criteria["delta_ci_lower"] = delta_ci_lower
    criteria["delta_ci_upper"] = delta_ci_upper
    criteria["ci_excludes_zero"] = ci_excludes_zero
    criteria["p_value_perm"] = p_value_perm
    # D003: delta_pass requires CI excludes 0 AND permutation p < 0.05
    criteria["delta_pass"] = delta_at_best > 0.1 and ci_excludes_zero and p_value_perm < 0.05

    # 3) VLA per-task mean rho >= DINO gate threshold (D004: 0.554)
    vla_mean_rho = 0.0
    dino_mean_rho = 0.0
    if "trained" in all_data:
        _, rho_vla_pt = extract_per_task_mean_rho(all_data["trained"])
        if len(rho_vla_pt) > 0:
            vla_mean_rho = float(rho_vla_pt.max())
    if "dinov2" in all_data:
        _, rho_dino_pt = extract_per_task_mean_rho(all_data["dinov2"])
        if len(rho_dino_pt) > 0:
            dino_mean_rho = float(rho_dino_pt.max())
    vla_ge_threshold = vla_mean_rho >= DINO_GATE_THRESHOLD

    # D003: subtask AUC is supplementary only — computed but not used in gate
    vla_best_auc = 0.0
    dino_best_auc = 0.0
    if "trained" in all_data:
        for v in all_data["trained"].get("results", {}).values():
            if isinstance(v.get("macro_auc"), (int, float)):
                vla_best_auc = max(vla_best_auc, v["macro_auc"])
    if "dinov2" in all_data:
        for v in all_data["dinov2"].get("results", {}).values():
            if isinstance(v.get("macro_auc"), (int, float)):
                dino_best_auc = max(dino_best_auc, v["macro_auc"])

    criteria["vla_mean_rho"] = vla_mean_rho
    criteria["dino_mean_rho"] = dino_mean_rho
    criteria["dino_gate_threshold"] = DINO_GATE_THRESHOLD
    criteria["vla_ge_threshold"] = vla_ge_threshold
    criteria["vla_best_auc"] = vla_best_auc
    criteria["dino_best_auc"] = dino_best_auc

    # D003: judgment based on temporal distance only (no subtask AUC gate)
    if len(collapsed_layers) > 0 and best_layer == -1:
        judgment = "FAIL"
    elif not criteria["rho_pass"] or not criteria["delta_pass"]:
        judgment = "FAIL"
    elif vla_ge_threshold:
        judgment = "PASS"
    else:
        judgment = "CONDITIONAL"

    # Effect size assessment (D004): flag weak VLA contribution even on PASS
    if judgment == "PASS":
        delta_vla_dino = vla_mean_rho - dino_mean_rho
        criteria["delta_vla_dino"] = delta_vla_dino
        criteria["contribution_warning"] = delta_vla_dino < 0.05

    reasons = []
    if len(collapsed_layers) > 0 and best_layer == -1:
        reasons.append(f"All layers collapsed (excluded: {sorted(collapsed_layers)}")
    if not criteria["rho_pass"]:
        reasons.append(f"best per-task mean rho = {best_rho:.4f} <= 0.5")
    if not criteria["delta_pass"]:
        if delta_at_best <= 0.1:
            reasons.append(f"delta-rho = {delta_at_best:.4f} <= 0.1")
        if not ci_excludes_zero:
            reasons.append(f"delta-rho CI [{delta_ci_lower:.4f}, {delta_ci_upper:.4f}] includes 0")
        if p_value_perm >= 0.05:
            reasons.append(f"permutation p={p_value_perm:.4f} >= 0.05")
    if not vla_ge_threshold:
        reasons.append(f"VLA per-task mean rho ({vla_mean_rho:.4f}) < DINO gate ({DINO_GATE_THRESHOLD})")
    if judgment == "PASS" and criteria.get("contribution_warning"):
        reasons.append(f"WARNING: VLA-DINO delta ({criteria['delta_vla_dino']:.4f}) < 0.05 — weak contribution")

    return dict(judgment=judgment, criteria=criteria,
                reasons=reasons if reasons else ["All criteria met"])


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dirs = parse_results_dirs(args.results_dirs)
    print(f"Loading results: {list(dirs.keys())}")

    all_data = {}
    for name, path in dirs.items():
        try:
            data = load_results(path, args.probe_type)
            all_data[name] = data
            n = len(data.get("results", {}))
            print(f"  {name}: {n} entries")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not all_data:
        print("No data loaded.")
        return

    oracle_rho = load_oracle(dirs)
    if oracle_rho is not None:
        print(f"Oracle rho: {oracle_rho:.4f}")

    # --- Comparison table ---
    print("\n=== Layer-wise Spearman rho ===")
    names = list(all_data.keys())
    header = ["Layer"] + names
    print(" | ".join(f"{h:>12}" for h in header))
    print("-" * (14 * len(header)))

    all_layers = set()
    for data in all_data.values():
        layers, _ = extract_layer_metrics(data, "spearman_rho")
        all_layers.update(layers.tolist())

    for layer in sorted(all_layers):
        row = [f"{layer:>12}"]
        for name in names:
            layers, rhos = extract_layer_metrics(all_data[name], "spearman_rho")
            lmap = dict(zip(layers.tolist(), rhos.tolist()))
            val = lmap.get(layer, float("nan"))
            row.append(f"{val:>12.4f}")
        print(" | ".join(row))

    print("\n=== Best Layer ===")
    for name, data in all_data.items():
        layers, rhos = extract_layer_metrics(data, "spearman_rho")
        if len(rhos) > 0:
            bi = int(np.argmax(rhos))
            print(f"  {name}: layer {layers[bi]} (rho={rhos[bi]:.4f})")

    # --- Plots ---
    if args.probe_type == "temporal_distance":
        plot_layer_rho(all_data, oracle_rho, args.output_dir)
        plot_delta_rho(all_data, args.output_dir)
        plot_per_layer_rho_profile(all_data, args.output_dir)

    if args.probe_type == "subtask_predicate":
        for name, data in all_data.items():
            plot_subtask_auc_heatmap(data, args.output_dir, name)

    # --- Judgment ---
    judgment = compute_judgment(all_data)
    print(f"\n=== Judgment: {judgment['judgment']} ===")
    for k, v in judgment["criteria"].items():
        print(f"  {k}: {v}")
    for r in judgment["reasons"]:
        print(f"  >> {r}")

    # --- Save ---
    summary = dict(
        models=names,
        probe_type=args.probe_type,
        judgment=judgment,
    )
    for name, data in all_data.items():
        layers, rhos = extract_layer_metrics(data, "spearman_rho")
        entry = {}
        if len(rhos) > 0:
            bi = int(np.argmax(rhos))
            entry["best_layer_pooled"] = int(layers[bi])
            entry["best_pooled_rho"] = float(rhos[bi])
        layers_pt, rhos_pt = extract_per_task_mean_rho(data)
        if len(rhos_pt) > 0:
            bi_pt = int(np.argmax(rhos_pt))
            entry["best_layer"] = int(layers_pt[bi_pt])
            entry["best_rho"] = float(rhos_pt[bi_pt])
        if entry:
            summary[name] = entry

    if oracle_rho is not None:
        summary["oracle_rho"] = oracle_rho

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
