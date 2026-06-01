"""analyze_action_probes.py — Analyze action prediction probe results.

Reads trained/untrained/dinov2 action_prediction (and optionally action_delta)
results, prints comparison tables, per-dim/per-task breakdowns, and gate judgments.

Supports partial results (not all 33 layers need to be present).

Usage:
  python scripts/analyze_action_probes.py \
    --results_base ./results \
    --output_dir ./results/analysis/
"""

import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np


MODELS = ["trained", "untrained", "dinov2"]
PROBE_TYPES = ["action_prediction", "action_delta"]
ACTION_DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

GATE_THRESHOLDS = {
    "G1_best_r2": 0.3,
    "G1_delta_peak_r2": 0.05,
    "G2_delta_margin": 0.03,
    "G2_delta_min_tasks": 5,
    "G3_dino_margin": 0.15,
}

COPY_BASELINE_R2 = 0.97
COPY_BASELINE_DELTA_R2 = 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_base", default="./results")
    p.add_argument("--output_dir", default="./results/analysis/")
    p.add_argument("--probe_type", default="action_prediction",
                   choices=PROBE_TYPES)
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


def get_per_dim_r2(data, layer=None):
    if data is None:
        return None
    results = data.get("results", {})
    if layer is not None:
        entry = results.get(str(layer))
        if entry:
            return entry.get("per_dim_r2")
        return None
    best_layer = None
    best_r2 = -np.inf
    for k, v in results.items():
        r2 = v.get("overall_r2", -np.inf)
        if r2 > best_r2:
            best_r2 = r2
            best_layer = k
    if best_layer and best_layer in results:
        return results[best_layer].get("per_dim_r2")
    return None


def get_per_task_r2(data, layer=None):
    if data is None:
        return None
    results = data.get("results", {})
    if layer is not None:
        entry = results.get(str(layer))
        if entry:
            return entry.get("per_task_r2")
        return None
    best_layer = None
    best_r2 = -np.inf
    for k, v in results.items():
        r2 = v.get("overall_r2", -np.inf)
        if r2 > best_r2:
            best_r2 = r2
            best_layer = k
    if best_layer and best_layer in results:
        return results[best_layer].get("per_task_r2")
    return None


def get_best_layer_info(data):
    if data is None:
        return None, None
    results = data.get("results", {})
    best_layer = None
    best_r2 = -np.inf
    for k, v in results.items():
        r2 = v.get("overall_r2", -np.inf)
        if r2 > best_r2:
            best_r2 = r2
            try:
                best_layer = int(k)
            except ValueError:
                best_layer = k
    return best_layer, best_r2


def short_task_name(full_name):
    parts = full_name.split("_", 2)
    scene = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
    desc = parts[2] if len(parts) > 2 else ""
    if len(desc) > 40:
        desc = desc[:37] + "..."
    return f"{scene}: {desc}"


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze(args):
    probe_type = args.probe_type
    all_data = {}
    for model in MODELS:
        d = load_results(os.path.join(args.results_base, model), probe_type)
        if d is not None:
            all_data[model] = d
            n_layers = len(d.get("results", {}))
            print(f"Loaded {model}/{probe_type}: {n_layers} layers")
        else:
            print(f"[SKIP] {model}/{probe_type}: not found")

    if not all_data:
        print("No results found. Exiting.")
        return

    # --- Per-layer R² table ---
    print_header(f"Per-Layer R² — {probe_type}")
    all_layers = set()
    layer_data = {}
    for model, data in all_data.items():
        layers, vals = get_layers_and_metric(data, "overall_r2")
        layer_data[model] = dict(zip(layers.tolist(), vals.tolist()))
        all_layers.update(layers.tolist())

    all_layers = sorted(all_layers)
    header = f"{'Layer':>6}"
    for m in MODELS:
        if m in all_data:
            header += f"  {m:>12}"
    print(header)
    print("-" * len(header))

    for layer in all_layers:
        row = f"{int(layer):>6}"
        for m in MODELS:
            if m in layer_data:
                val = layer_data[m].get(layer)
                row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12}"
        print(row)

    # --- Best layer summary ---
    print_header("Best Layer Summary")
    for model in MODELS:
        if model in all_data:
            bl, br = get_best_layer_info(all_data[model])
            print(f"  {model:>12}: best_layer={bl}, best_R²={br:.4f}")

    # --- Per-dim R² at best layer ---
    print_header(f"Per-Dim R² at Best Layer — {probe_type}")
    dim_header = f"{'Dim':>10}"
    best_layers = {}
    for m in MODELS:
        if m in all_data:
            bl, _ = get_best_layer_info(all_data[m])
            best_layers[m] = bl
            dim_header += f"  {m}(L{bl}):>16"[-18:]
    # reformat header properly
    dim_header = f"{'Dim':>10}"
    for m in MODELS:
        if m in all_data:
            bl = best_layers[m]
            dim_header += f"  {m}(L{bl})".rjust(16)
    print(dim_header)
    print("-" * len(dim_header))

    per_dim_data = {}
    for m in MODELS:
        if m in all_data:
            per_dim_data[m] = get_per_dim_r2(all_data[m])

    for dim in ACTION_DIMS:
        row = f"{dim:>10}"
        for m in MODELS:
            if m in per_dim_data and per_dim_data[m]:
                val = per_dim_data[m].get(dim)
                row += f"  {val:>14.4f}" if val is not None else f"  {'—':>14}"
        print(row)

    # --- Per-task R² at best layer ---
    print_header(f"Per-Task R² at Best Layer — {probe_type}")
    per_task_data = {}
    all_tasks = set()
    for m in MODELS:
        if m in all_data:
            pt = get_per_task_r2(all_data[m])
            per_task_data[m] = pt or {}
            all_tasks.update((pt or {}).keys())

    all_tasks = sorted(all_tasks)
    task_header = f"{'Task':>55}"
    for m in MODELS:
        if m in all_data:
            task_header += f"  {m:>10}"
    print(task_header)
    print("-" * len(task_header))

    consistency_count = 0
    total_tasks = 0
    for task in all_tasks:
        row = f"{short_task_name(task):>55}"
        vals = {}
        for m in MODELS:
            if m in per_task_data:
                val = per_task_data[m].get(task)
                vals[m] = val
                row += f"  {val:>10.4f}" if val is not None else f"  {'—':>10}"
        print(row)
        if "trained" in vals and "untrained" in vals:
            if vals["trained"] is not None and vals["untrained"] is not None:
                total_tasks += 1
                if vals["trained"] > vals["untrained"]:
                    consistency_count += 1

    if total_tasks > 0:
        print(f"\nPer-task consistency (trained > untrained): "
              f"{consistency_count}/{total_tasks} tasks")

    # --- Gate Judgments ---
    print_header("Gate Judgments")

    # G1: best-layer R² > 0.3
    trained_best_layer, trained_best_r2 = get_best_layer_info(
        all_data.get("trained"))
    g1_pass = trained_best_r2 is not None and trained_best_r2 > GATE_THRESHOLDS["G1_best_r2"]
    status = "PASS" if g1_pass else ("FAIL" if trained_best_r2 is not None else "N/A")
    val_str = f"{trained_best_r2:.4f}" if trained_best_r2 is not None else "—"
    print(f"  G1 (best-layer R² > {GATE_THRESHOLDS['G1_best_r2']}): "
          f"{status} (trained best R²={val_str})")

    # G1-delta: trained action-delta peak R² > 0.05
    if probe_type == "action_delta":
        _, delta_best = get_best_layer_info(all_data.get("trained"))
        g1d_pass = delta_best is not None and delta_best > GATE_THRESHOLDS["G1_delta_peak_r2"]
        status = "PASS" if g1d_pass else ("FAIL" if delta_best is not None else "N/A")
        val_str = f"{delta_best:.4f}" if delta_best is not None else "—"
        print(f"  G1-delta (action-delta peak R² > "
              f"{GATE_THRESHOLDS['G1_delta_peak_r2']}): "
              f"{status} (R²={val_str})")
    else:
        delta_data = load_results(
            os.path.join(args.results_base, "trained"), "action_delta")
        if delta_data:
            _, delta_best = get_best_layer_info(delta_data)
            g1d_pass = delta_best is not None and delta_best > GATE_THRESHOLDS["G1_delta_peak_r2"]
            status = "PASS" if g1d_pass else "FAIL"
            print(f"  G1-delta (action-delta peak R² > "
                  f"{GATE_THRESHOLDS['G1_delta_peak_r2']}): "
                  f"{status} (R²={delta_best:.4f})")
        else:
            print(f"  G1-delta: N/A (action_delta results not available)")

    # G2-delta: Δ(trained-untrained) action-delta > 0.03, ≥5/10 tasks consistent
    delta_trained = load_results(
        os.path.join(args.results_base, "trained"), "action_delta")
    delta_untrained = load_results(
        os.path.join(args.results_base, "untrained"), "action_delta")
    if delta_trained and delta_untrained:
        tr_bl, tr_best = get_best_layer_info(delta_trained)
        un_bl, un_best = get_best_layer_info(delta_untrained)
        if tr_best is not None and un_best is not None:
            margin = tr_best - un_best
            tr_tasks = get_per_task_r2(delta_trained)
            un_tasks = get_per_task_r2(delta_untrained)
            n_consistent = 0
            n_total = 0
            if tr_tasks and un_tasks:
                for t in tr_tasks:
                    if t in un_tasks:
                        n_total += 1
                        if tr_tasks[t] > un_tasks[t]:
                            n_consistent += 1
            g2_pass = (margin > GATE_THRESHOLDS["G2_delta_margin"] and
                       n_consistent >= GATE_THRESHOLDS["G2_delta_min_tasks"])
            print(f"  G2-delta (Δ > {GATE_THRESHOLDS['G2_delta_margin']} "
                  f"& ≥{GATE_THRESHOLDS['G2_delta_min_tasks']}/10 tasks): "
                  f"{'PASS' if g2_pass else 'FAIL'} "
                  f"(Δ={margin:.4f}, consistency={n_consistent}/{n_total})")
    else:
        print(f"  G2-delta: N/A (need both trained & untrained action_delta)")

    # G3: Δ(trained - DINO) > 0.15
    dino_best_layer, dino_best_r2 = get_best_layer_info(
        all_data.get("dinov2"))
    if trained_best_r2 is not None and dino_best_r2 is not None:
        g3_margin = trained_best_r2 - dino_best_r2
        g3_pass = g3_margin > GATE_THRESHOLDS["G3_dino_margin"]
        print(f"  G3 (trained - DINO > {GATE_THRESHOLDS['G3_dino_margin']}): "
              f"{'PASS' if g3_pass else 'FAIL'} "
              f"(Δ={g3_margin:.4f}, trained={trained_best_r2:.4f}, "
              f"DINO={dino_best_r2:.4f})")
    else:
        print(f"  G3: N/A (need both trained & dinov2)")

    # --- Save summary JSON ---
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "probe_type": probe_type,
        "models_loaded": list(all_data.keys()),
        "best_layers": {},
        "gate_results": {},
    }
    for m in MODELS:
        if m in all_data:
            bl, br = get_best_layer_info(all_data[m])
            summary["best_layers"][m] = {
                "layer": bl, "overall_r2": br,
                "per_dim_r2": get_per_dim_r2(all_data[m]),
                "per_task_r2": get_per_task_r2(all_data[m]),
            }

    summary["gate_results"]["G1"] = {
        "threshold": GATE_THRESHOLDS["G1_best_r2"],
        "value": trained_best_r2,
        "pass": g1_pass if trained_best_r2 is not None else None,
    }

    out_path = os.path.join(args.output_dir, f"action_probe_analysis_{probe_type}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    analyze(args)

    if args.probe_type == "action_prediction":
        print("\n\n" + "="*70)
        print("  Also checking action_delta results...")
        print("="*70)
        args2 = argparse.Namespace(**vars(args))
        args2.probe_type = "action_delta"
        analyze(args2)
