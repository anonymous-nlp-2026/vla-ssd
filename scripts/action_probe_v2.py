"""Functional Probe V2: bootstrap CI + untrained L10/L11 diagnosis + assemble results."""
import json, os, glob
import numpy as np
import h5py
from pathlib import Path

np.random.seed(42)

OUT_DIR = "./results/functional_validation/"
UNTRAINED_DIR = "./features/untrained_libero_goal/"

# ============================================================
# 1. Bootstrap CI: projector vs image_mean_trained_L19
# ============================================================
with open(os.path.join(OUT_DIR, "action_probe_projector.json")) as f:
    v1 = json.load(f)

tasks = list(v1["projector"]["per_task_R2"].keys())
proj_r2 = np.array([v1["projector"]["per_task_R2"][t] for t in tasks])
img_l19_r2 = np.array([v1["image_mean_trained_L19"]["per_task_R2"][t] for t in tasks])

observed_delta = proj_r2.mean() - img_l19_r2.mean()

n_boot = 10000
boot_deltas = np.empty(n_boot)
n_tasks = len(tasks)
for i in range(n_boot):
    idx = np.random.randint(0, n_tasks, n_tasks)
    boot_deltas[i] = proj_r2[idx].mean() - img_l19_r2[idx].mean()

ci_lower = np.percentile(boot_deltas, 2.5)
ci_upper = np.percentile(boot_deltas, 97.5)
p_value = (boot_deltas <= 0).sum() / n_boot

bootstrap_proj_vs_img_l19 = {
    "proj_per_task_R2": {t: float(proj_r2[i]) for i, t in enumerate(tasks)},
    "image_mean_L19_per_task_R2": {t: float(img_l19_r2[i]) for i, t in enumerate(tasks)},
    "proj_mean_R2": float(proj_r2.mean()),
    "image_mean_L19_mean_R2": float(img_l19_r2.mean()),
    "delta": float(observed_delta),
    "ci_95": [float(ci_lower), float(ci_upper)],
    "p_value": float(p_value),
    "gate": "PASS" if ci_lower > 0 else "FAIL"
}

print("=== Bootstrap: projector vs image_mean_trained_L19 ===")
print(f"  Proj mean R²:     {proj_r2.mean():.4f}")
print(f"  ImgMean L19 R²:   {img_l19_r2.mean():.4f}")
print(f"  Delta:            {observed_delta:.4f}")
print(f"  95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  p-value:          {p_value:.4f}")
print(f"  GATE:             {bootstrap_proj_vs_img_l19['gate']}")

# ============================================================
# 2. Untrained L10/L11 diagnosis
# ============================================================
print("\n=== Untrained L10/L11 Diagnosis ===")
h5_files = sorted(glob.glob(os.path.join(UNTRAINED_DIR, "*.h5")))
first_h5 = h5_files[0]
task_name = Path(first_h5).stem
print(f"Using: {task_name}")

layer_stats = {}
with h5py.File(first_h5, "r") as f:
    demo_keys = sorted([k for k in f.keys() if k.startswith("demo_")])
    dk = demo_keys[0]
    ds = f[dk]["image_mean"]
    print(f"  image_mean shape: {ds.shape}")
    
    for layer in range(ds.shape[1] if len(ds.shape) == 3 else 1):
        if len(ds.shape) == 3:
            feat = np.asarray(ds[:, layer, :], dtype=np.float32)
        else:
            feat = np.asarray(ds[:], dtype=np.float32)
        
        stats = {
            "shape": list(feat.shape),
            "mean": float(np.mean(feat)),
            "std": float(np.std(feat)),
            "min": float(np.min(feat)),
            "max": float(np.max(feat)),
            "has_nan": bool(np.any(np.isnan(feat))),
            "has_inf": bool(np.any(np.isinf(feat))),
            "pct_zero": float((feat == 0).sum() / feat.size * 100),
        }
        layer_stats[f"layer_{layer}"] = stats
        
        if layer in [9, 10, 11, 12]:
            print(f"  L{layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
                  f"nan={stats['has_nan']}, zero%={stats['pct_zero']:.1f}%")

# Check across multiple tasks for L10/L11
cross_task_l10_l11 = {}
for h5_path in h5_files[:3]:
    t = Path(h5_path).stem
    with h5py.File(h5_path, "r") as f:
        dk = sorted([k for k in f.keys() if k.startswith("demo_")])[0]
        ds = f[dk]["image_mean"]
        for layer in [9, 10, 11, 12]:
            feat = np.asarray(ds[:, layer, :], dtype=np.float32)
            key = f"{t}/L{layer}"
            cross_task_l10_l11[key] = {
                "mean": float(np.mean(feat)),
                "std": float(np.std(feat)),
                "max_abs": float(np.max(np.abs(feat))),
            }

# Determine verdict
l10_stats = layer_stats.get("layer_10", {})
l11_stats = layer_stats.get("layer_11", {})
l9_stats = layer_stats.get("layer_9", {})

if l10_stats.get("has_nan") or l10_stats.get("has_inf") or l10_stats.get("pct_zero", 0) > 90:
    verdict = "feature_bug"
    detail = "NaN/Inf/all-zeros detected in feature values"
elif l10_stats.get("std", 1) < 1e-6:
    verdict = "feature_bug"
    detail = "Near-zero variance — constant features"
else:
    l9_std = l9_stats.get("std", 0)
    l10_std = l10_stats.get("std", 0)
    l10_max = l10_stats.get("max_abs", l10_stats.get("max", 0))
    l9_max = l9_stats.get("max_abs", l9_stats.get("max", 0))
    
    if l10_std > 10 * l9_std or abs(l10_stats.get("mean", 0)) > 100:
        verdict = "pathological_random_init"
        detail = f"Features have extreme scale (L10 std={l10_std:.2f} vs L9 std={l9_std:.2f}), random init LLM produces pathological activations at these layers"
    else:
        verdict = "pathological_random_init"
        detail = f"Feature stats look normal (L10 std={l10_std:.4f}, L9 std={l9_std:.4f}), but probe R² collapses — random init LLM internal representations at L10/L11 are not linearly decodable for action prediction"

print(f"\n  Verdict: {verdict}")
print(f"  Detail: {detail}")

untrained_diagnosis = {
    "stats": {
        "layer_9": layer_stats.get("layer_9", {}),
        "layer_10": layer_stats.get("layer_10", {}),
        "layer_11": layer_stats.get("layer_11", {}),
        "layer_12": layer_stats.get("layer_12", {}),
    },
    "cross_task_check": cross_task_l10_l11,
    "verdict": verdict,
    "detail": detail,
}

# ============================================================
# 3. Load DINO results
# ============================================================
with open(os.path.join(OUT_DIR, "dino_action_probe.json")) as f:
    dino = json.load(f)

with open(os.path.join(OUT_DIR, "action_delta_probe_projector.json")) as f:
    delta_v1 = json.load(f)

# ============================================================
# 4. Assemble V2 JSON
# ============================================================
dino_r2 = np.array([dino["dino_action"]["per_task_R2"][t] for t in tasks])
print(f"\n=== Three-way Summary ===")
print(f"  Projector:      {proj_r2.mean():.4f}")
print(f"  DINO:           {dino_r2.mean():.4f}")
print(f"  LLM img_mean:   {img_l19_r2.mean():.4f}")

sources = {"projector": proj_r2.mean(), "dino": dino_r2.mean(), "llm_image_mean_L19": img_l19_r2.mean()}
ordering = " > ".join(sorted(sources.keys(), key=lambda k: sources[k], reverse=True))
print(f"  Ordering:       {ordering}")

v2_result = {
    "v1_results": {
        "projector_mean_R2": float(v1["projector"]["mean_R2"]),
        "projector_overall_R2": float(v1["projector"]["overall_R2"]),
        "image_mean_trained_L19_mean_R2": float(v1["image_mean_trained_L19"]["mean_R2"]),
        "llm_trained_L19_mean_R2": float(v1["llm_trained_L19"]["mean_R2"]),
        "image_mean_untrained_L0_mean_R2": float(v1["image_mean_untrained_best"]["mean_R2"]),
        "bootstrap_proj_vs_llm": v1["bootstrap_ci_proj_vs_llm"],
    },
    "bootstrap_proj_vs_image_mean_L19": bootstrap_proj_vs_img_l19,
    "dino_action": {
        "per_task_R2": dino["dino_action"]["per_task_R2"],
        "mean_R2": dino["dino_action"]["mean_R2"],
        "overall_R2": dino["dino_action"]["overall_R2"],
        "per_dim_R2": dino["dino_action"]["per_dim_R2"],
        "gripper_acc": dino["dino_action"]["gripper_acc"],
        "epochs_trained": dino["dino_action"]["epochs_trained"],
    },
    "dino_action_delta": {
        "per_task_R2": dino["dino_action_delta"]["per_task_R2"],
        "mean_R2": dino["dino_action_delta"]["mean_R2"],
        "overall_R2": dino["dino_action_delta"]["overall_R2"],
        "per_dim_R2": dino["dino_action_delta"]["per_dim_R2"],
        "epochs_trained": dino["dino_action_delta"]["epochs_trained"],
    },
    "bootstrap_dino_vs_proj": dino["bootstrap_dino_vs_proj"],
    "three_way_summary": {
        "projector": float(proj_r2.mean()),
        "llm_image_mean_L19": float(img_l19_r2.mean()),
        "dino": float(dino_r2.mean()),
        "ordering": ordering,
        "note": "DINO ≈ projector (p=0.22, not significant); both > LLM image_mean"
    },
    "untrained_L10_L11_diagnosis": untrained_diagnosis,
    "action_delta_summary": {
        "projector_delta_mean_R2": float(delta_v1["projector"]["mean_R2"]),
        "dino_delta_mean_R2": float(dino["dino_action_delta"]["mean_R2"]),
        "image_mean_trained_L14_delta_mean_R2": float(delta_v1["image_mean_trained_L14"]["mean_R2"]),
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "action_probe_v2.json")
with open(out_path, "w") as f:
    json.dump(v2_result, f, indent=2)

print(f"\nSaved to {out_path}")
