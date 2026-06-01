"""rsa_instruction_control.py — RSA comparison across instruction control modes.

For plan_008: compares RSA (representation-action correlation) across instruction
conditions to verify whether instruction content affects representation structure.

Conditions:
  normal:     Real task instructions          (trained_libero_goal/)
  empty:      Empty instruction string        (trained_libero_goal_no_inst/)
  random_s0:  Random nonsense seed=0          (trained_libero_goal_random_s0/)
  random_s1:  Random nonsense seed=1          (trained_libero_goal_random_s1/)
  random_s2:  Random nonsense seed=2          (trained_libero_goal_random_s2/)
  shuffled:   Shuffled word order             (trained_libero_goal_shuffled/)
  wrong_task: Cyclic-shifted task instruction (trained_libero_goal_wrong_task/)

Input:
  Feature HDF5: ./features/{condition_dir}/*.h5
    Keys: demo_X/{last_preaction,image_mean}, shape (T, 33, 4096)
  Actions:      ./data/libero/libero_goal/*_demo.hdf5
    Keys: data/demo_X/actions, shape (T, 7)

Output:
  ./results/rsa/rsa_instruction_control.json

Usage:
  python scripts/rsa_instruction_control.py
  python scripts/rsa_instruction_control.py --token_type image_mean
  python scripts/rsa_instruction_control.py --layers 0,8,16,24,32  # subset for speed
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

FEAT_ROOT = Path("./features")
DATA_ROOT = Path("./data/libero/libero_goal")
OUT_DIR   = Path("./results/rsa")

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

CONDITIONS = {
    "normal":     "trained_libero_goal",
    "empty":      "trained_libero_goal_no_inst",
    "random_s0":  "trained_libero_goal_random_s0",
    "random_s1":  "trained_libero_goal_random_s1",
    "random_s2":  "trained_libero_goal_random_s2",
    "shuffled":   "trained_libero_goal_shuffled",
    "wrong_task": "trained_libero_goal_wrong_task",
}

TEMPORAL_BINS = {
    "early": (0.0, 0.33),
    "mid":   (0.33, 0.67),
    "late":  (0.67, 1.0),
}

N_SAMPLE_FRAMES = 20


# ── data loading ──────────────────────────────────────────────────────

def load_vla_features(feat_dir, task, demo_ids, layer, token_type="last_preaction"):
    """Load VLA features for one task. Returns {demo_id: (T, D) array} or None."""
    path = FEAT_ROOT / feat_dir / f"{task}.h5"
    if not path.exists():
        return None
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"demo_{did}/{token_type}"
            if key in f:
                result[did] = f[key][:, layer, :].astype(np.float32)
    return result if result else None


def load_actions(task, demo_ids):
    path = DATA_ROOT / f"{task}_demo.hdf5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"data/demo_{did}/actions"
            if key in f:
                result[did] = f[key][:].astype(np.float32)
    return result


def get_demo_ids(task, max_demos=50):
    """Get available demo IDs for a task."""
    path = DATA_ROOT / f"{task}_demo.hdf5"
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        ids = [int(k.split("_")[1]) for k in keys[:max_demos]]
    return ids


# ── dissimilarity computation ─────────────────────────────────────────

def get_phase_slice(arr, phase):
    T = len(arr)
    lo, hi = TEMPORAL_BINS[phase]
    t_norm = np.linspace(0, 1, T, endpoint=False)
    if phase == "late":
        mask = (t_norm >= lo) & (t_norm <= 1.0)
    else:
        mask = (t_norm >= lo) & (t_norm < hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.array([0])
    seg = arr[idx]
    if len(seg) > N_SAMPLE_FRAMES:
        sample_idx = np.linspace(0, len(seg) - 1, N_SAMPLE_FRAMES, dtype=int)
        seg = seg[sample_idx]
    return seg


def build_phase_tensor(data, keys, phase):
    slices = [get_phase_slice(data[k], phase) for k in keys]
    max_len = max(len(s) for s in slices)
    D = slices[0].shape[1]
    N = len(slices)
    tensor = np.zeros((N, max_len, D), dtype=np.float32)
    lengths = np.zeros(N, dtype=int)
    for i, s in enumerate(slices):
        tensor[i, :len(s)] = s
        lengths[i] = len(s)
    return tensor, lengths


def pairwise_cosine_dm(tensor, lengths):
    """Pairwise mean cosine distance matrix. tensor: (N, T, D), lengths: (N,)."""
    N, T, D = tensor.shape
    norms = np.linalg.norm(tensor, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = tensor / norms
    for i in range(N):
        if lengths[i] < T:
            normed[i, lengths[i]:] = 0.0
    sim = np.einsum('itd,jtd->ijt', normed, normed)
    min_lens = np.minimum(lengths[:, None], lengths[None, :])
    frame_idx = np.arange(T)[None, None, :]
    valid = frame_idx < min_lens[:, :, None]
    dist = (1.0 - sim) * valid
    count = np.maximum(valid.sum(axis=2).astype(np.float32), 1.0)
    return (dist.sum(axis=2) / count).astype(np.float32)


def compute_rsa(rdm, adm):
    """Spearman correlation between upper triangles of two distance matrices."""
    idx = np.triu_indices_from(rdm, k=1)
    r_vec = rdm[idx]
    a_vec = adm[idx]
    if np.std(r_vec) < 1e-10 or np.std(a_vec) < 1e-10:
        return 0.0
    rho, _ = spearmanr(r_vec, a_vec)
    return float(rho) if not np.isnan(rho) else 0.0


# ── per-task RSA ──────────────────────────────────────────────────────

def compute_task_rsa(feat_data, action_data, common_keys):
    """Compute RSA for one task, averaged across temporal phases."""
    phase_rsa = {}
    for phase in TEMPORAL_BINS:
        feat_tensor, feat_lens = build_phase_tensor(feat_data, common_keys, phase)
        act_tensor, act_lens = build_phase_tensor(action_data, common_keys, phase)
        rdm = pairwise_cosine_dm(feat_tensor, feat_lens)
        adm = pairwise_cosine_dm(act_tensor, act_lens)
        phase_rsa[phase] = compute_rsa(rdm, adm)
    return float(np.mean(list(phase_rsa.values()))), phase_rsa


# ── main ──────────────────────────────────────────────────────────────

def parse_layers(spec, max_layer=32):
    if spec is None:
        return list(range(max_layer + 1))
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="RSA comparison across instruction control modes (plan_008)")
    p.add_argument("--token_type", default="last_preaction",
                   choices=["last_preaction", "image_mean"],
                   help="Which token position to analyze")
    p.add_argument("--layers", default=None,
                   help="Layers: '0-32' or '0,8,16,24,32'. Default: all 33")
    p.add_argument("--max_demos", type=int, default=50)
    p.add_argument("--conditions", default=None,
                   help="Comma-separated condition names. Default: all available")
    return p.parse_args()


def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    token_type = args.token_type

    # Determine which conditions have feature data
    if args.conditions:
        requested = [c.strip() for c in args.conditions.split(",")]
        cond_map = {k: CONDITIONS[k] for k in requested if k in CONDITIONS}
    else:
        cond_map = dict(CONDITIONS)

    available = {}
    for name, feat_dir in cond_map.items():
        feat_path = FEAT_ROOT / feat_dir
        if feat_path.exists() and any(feat_path.glob("*.h5")):
            available[name] = feat_dir

    if len(available) < 2:
        print(f"ERROR: Need at least 2 conditions with data. Found: {list(available.keys())}")
        print(f"Missing feature dirs:")
        for name, feat_dir in cond_map.items():
            if name not in available:
                print(f"  {name}: {FEAT_ROOT / feat_dir}")
        return

    print(f"=== plan_008: Instruction Control RSA Comparison ===")
    print(f"Token type: {token_type}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Conditions: {list(available.keys())}")

    results = {
        "config": {
            "token_type": token_type,
            "layers": layers,
            "conditions": list(available.keys()),
            "n_tasks": len(TASKS),
            "max_demos": args.max_demos,
        },
        "per_layer": {},
        "summary": {},
    }

    # Pre-load demo IDs (shared across conditions)
    task_demo_ids = {task: get_demo_ids(task, args.max_demos) for task in TASKS}

    t_total = time.time()
    for layer in layers:
        t0 = time.time()
        layer_key = f"layer_{layer}"
        layer_results = {}

        for cond_name, feat_dir in available.items():
            task_rsa_vals = []

            for task in TASKS:
                demo_ids = task_demo_ids[task]
                feat_data = load_vla_features(feat_dir, task, demo_ids, layer, token_type)
                if feat_data is None:
                    continue
                action_data = load_actions(task, demo_ids)
                common = sorted(set(feat_data.keys()) & set(action_data.keys()))
                if len(common) < 5:
                    continue
                rsa_val, _ = compute_task_rsa(feat_data, action_data, common)
                task_rsa_vals.append(rsa_val)

            if task_rsa_vals:
                layer_results[cond_name] = round(float(np.mean(task_rsa_vals)), 4)
            else:
                layer_results[cond_name] = None

        # Aggregate random seeds
        random_vals = [layer_results.get(f"random_s{s}")
                       for s in range(3)
                       if layer_results.get(f"random_s{s}") is not None]
        if random_vals:
            layer_results["random_mean"] = round(float(np.mean(random_vals)), 4)
            layer_results["random_std"] = round(float(np.std(random_vals)), 4)

        results["per_layer"][layer_key] = layer_results

        elapsed = time.time() - t0
        parts = [f"{k}={v:.4f}" for k, v in layer_results.items()
                 if v is not None and not k.startswith("random_s")]
        print(f"  L{layer:2d}: {', '.join(parts)}  [{elapsed:.1f}s]")

    # ── summary: peak RSA per condition ──
    for cond_name in list(available.keys()) + (["random_mean"] if any(
            results["per_layer"][f"layer_{l}"].get("random_mean") is not None for l in layers) else []):
        vals = [(l, results["per_layer"][f"layer_{l}"].get(cond_name))
                for l in layers]
        vals = [(l, v) for l, v in vals if v is not None]
        if not vals:
            continue
        peak_layer, peak_val = max(vals, key=lambda x: x[1])
        mean_val = float(np.mean([v for _, v in vals]))
        results["summary"][cond_name] = {
            "peak_rsa": round(peak_val, 4),
            "peak_layer": int(peak_layer),
            "mean_rsa": round(mean_val, 4),
        }

    # ── pairwise deltas at peak ──
    ref_cond = "normal" if "normal" in results["summary"] else list(results["summary"].keys())[0]
    ref_peak = results["summary"][ref_cond]["peak_rsa"]
    deltas = {}
    for cond_name, s in results["summary"].items():
        if cond_name == ref_cond:
            continue
        deltas[cond_name] = round(s["peak_rsa"] - ref_peak, 4)
    results["deltas_vs_normal"] = deltas

    max_delta = max(abs(d) for d in deltas.values()) if deltas else 0.0
    results["conclusion"] = {
        "max_absolute_delta": round(max_delta, 4),
        "instruction_invariant": max_delta < 0.02,
        "note": ("All conditions within 0.02 of normal — instruction content "
                 "does not affect RSA" if max_delta < 0.02
                 else f"Max delta {max_delta:.4f} exceeds 0.02 threshold"),
    }

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal: {total_min:.1f} min")

    # ── save ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"rsa_instruction_control_{token_type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # ── summary table ──
    print("\n" + "=" * 90)
    print("INSTRUCTION CONTROL RSA SUMMARY")
    print("=" * 90)

    cond_names = [c for c in available.keys() if not c.startswith("random_s")]
    if any(results["per_layer"][f"layer_{l}"].get("random_mean") is not None for l in layers):
        cond_names_display = []
        for c in cond_names:
            cond_names_display.append(c)
            if c == "empty" and "random_mean" not in cond_names:
                cond_names_display.append("random_mean")
        cond_names = cond_names_display

    header = f"{'Layer':>6}"
    for c in cond_names:
        header += f" | {c:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for l in layers:
        lk = f"layer_{l}"
        row = f"{l:>6d}"
        for c in cond_names:
            v = results["per_layer"][lk].get(c)
            if v is not None:
                row += f" | {v:>12.4f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    print(f"\n{'Condition':>12} | {'Peak RSA':>10} | {'Peak Layer':>10} | {'Mean RSA':>10} | {'Delta':>8}")
    print("-" * 62)
    for cond_name in cond_names:
        s = results["summary"].get(cond_name)
        if s is None:
            continue
        delta = results["deltas_vs_normal"].get(cond_name, 0.0)
        print(f"{cond_name:>12} | {s['peak_rsa']:>10.4f} | {s['peak_layer']:>10d} | "
              f"{s['mean_rsa']:>10.4f} | {delta:>+8.4f}")

    print(f"\nConclusion: {results['conclusion']['note']}")


if __name__ == "__main__":
    main()
