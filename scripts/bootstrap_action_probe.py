import json
import numpy as np

RESULTS_PATH = "./results/l1_head_ablation/action_probe_results.json"
OUTPUT_PATH = "./results/l1_head_ablation/bootstrap_ci_results.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

detailed = data["detailed_results"]
conditions = ["baseline", "ablate_all24", "ablate_all32"]
layers = ["L1", "L8", "L13"]

# Extract per-task R² as aligned arrays (same task order)
task_names = sorted(detailed["baseline"]["L1"]["per_task_R2"].keys())

def get_per_task_array(condition, layer):
    d = detailed[condition][layer]["per_task_R2"]
    return np.array([d[t] for t in task_names])

def paired_bootstrap(vals_a, vals_b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(vals_a)
    obs_delta = np.mean(vals_b) - np.mean(vals_a)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[i] = np.mean(vals_b[idx]) - np.mean(vals_a[idx])
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    # one-sided p: H1 is delta > 0 for L1, two-sided for others
    p_one = np.mean(boot_deltas <= 0)
    p_two = 2 * min(np.mean(boot_deltas <= 0), np.mean(boot_deltas >= 0))
    return obs_delta, (ci_lo, ci_hi), p_one, p_two, boot_deltas

def permutation_test(vals_a, vals_b, n_perm=10000, seed=42):
    rng = np.random.RandomState(seed)
    obs = np.mean(vals_b) - np.mean(vals_a)
    count = 0
    for _ in range(n_perm):
        swap = rng.binomial(1, 0.5, size=len(vals_a)).astype(bool)
        pa = np.where(swap, vals_b, vals_a)
        pb = np.where(swap, vals_a, vals_b)
        if np.mean(pb) - np.mean(pa) >= obs:
            count += 1
    return count / n_perm

comparisons = [
    ("baseline", "ablate_all24"),
    ("baseline", "ablate_all32"),
]

results = {}

print("=" * 60)
print("Bootstrap CI Results (10,000 resamples, paired by task)")
print("=" * 60)

for cond_a, cond_b in comparisons:
    key = f"{cond_a}_vs_{cond_b}"
    results[key] = {}
    print(f"\n{cond_a} vs {cond_b}:")
    for layer in layers:
        va = get_per_task_array(cond_a, layer)
        vb = get_per_task_array(cond_b, layer)
        delta, (ci_lo, ci_hi), p_one, p_two, _ = paired_bootstrap(va, vb)
        p_perm = permutation_test(va, vb)
        sig = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else "ns"
        print(f"  {layer}:  Δ = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
              f"p(one-sided) = {p_one:.4f}, p(two-sided) = {p_two:.4f}, "
              f"p(perm) = {p_perm:.4f}  {sig}")
        results[key][layer] = {
            "observed_delta": round(delta, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "p_value_one_sided": round(p_one, 4),
            "p_value_two_sided": round(p_two, 4),
            "p_value_permutation": round(p_perm, 4),
            "significant_at_005": p_one < 0.05,
            "baseline_mean_R2": round(float(np.mean(va)), 4),
            "ablated_mean_R2": round(float(np.mean(vb)), 4),
            "per_task_deltas": {t: round(float(vb[i] - va[i]), 4) for i, t in enumerate(task_names)},
        }

print("\n" + "=" * 60)
print("Per-task Δ (ablate_all24 - baseline, L1):")
print("=" * 60)
va = get_per_task_array("baseline", "L1")
vb = get_per_task_array("ablate_all24", "L1")
for i, t in enumerate(task_names):
    print(f"  {t:50s}  {va[i]:.4f} -> {vb[i]:.4f}  Δ={vb[i]-va[i]:+.4f}")

print("\nPer-task Δ (ablate_all32 - baseline, L1):")
vc = get_per_task_array("ablate_all32", "L1")
for i, t in enumerate(task_names):
    print(f"  {t:50s}  {va[i]:.4f} -> {vc[i]:.4f}  Δ={vc[i]-va[i]:+.4f}")

output = {
    "bootstrap_ci": results,
    "metadata": {
        "n_bootstrap": 10000,
        "n_permutation": 10000,
        "seed": 42,
        "n_tasks": len(task_names),
        "task_names": task_names,
        "method": "paired bootstrap (resample task indices with replacement)",
        "ci_level": 0.95,
        "p_one_sided": "H1: ablated > baseline",
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
