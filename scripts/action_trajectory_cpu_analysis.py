"""Action Trajectory CPU Analysis

Two analyses using existing ablation data (no GPU, no model loading):
1. Temporal coherence: demo-level autocorrelation + per-dim CV proxy
2. Bootstrap CI on per-head effect size (B=1000)
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = "./results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "action_trajectory_cpu_analysis.json")

def load_all_data():
    with open(os.path.join(RESULTS_DIR, "action_trajectory_analysis.json")) as f:
        inst = json.load(f)
    with open(os.path.join(RESULTS_DIR, "action_trajectory_control.json")) as f:
        ctrl = json.load(f)
    with open(os.path.join(RESULTS_DIR, "trajectory_level_cohens_d.json")) as f:
        traj = json.load(f)
    return inst, ctrl, traj


# ─── Analysis 1: Temporal Coherence ───────────────────────────────────

def temporal_coherence_analysis(inst, ctrl, traj):
    """
    Raw per-timestep actions not saved, so we use two proxy approaches:
    
    A) Demo-level MSE autocorrelation: within each task's 10 demos, compute
       lag-1 autocorrelation of demo MSEs. Higher autocorrelation = more
       structured/temporally-coherent ablation effect.
       
    B) Per-dim CV: lower CV = more consistent perturbation magnitude across
       samples, suggesting systematic rather than random disruption.
    """
    
    # --- A) Demo-level autocorrelation from trajectory_level_cohens_d ---
    # head_ablation_top24 = instruction ablation (50 demos = 5 tasks x 10)
    cond_map = {r["condition"]: r for r in traj["results"]}
    
    inst_cond = cond_map["head_ablation_top24"]
    inst_demo_mses = np.array(inst_cond["demo_mses"])  # 50 demos
    n_tasks_inst = 5
    demos_per_task_inst = len(inst_demo_mses) // n_tasks_inst  # 10
    
    # Compute lag-1 autocorrelation within each task's demo sequence
    def lag1_autocorr(series):
        if len(series) < 3:
            return np.nan
        x = np.array(series)
        n = len(x)
        mean = np.mean(x)
        var = np.var(x, ddof=0)
        if var < 1e-15:
            return np.nan
        autocov = np.sum((x[:-1] - mean) * (x[1:] - mean)) / n
        return autocov / var
    
    inst_autocorrs = []
    for t in range(n_tasks_inst):
        start = t * demos_per_task_inst
        end = start + demos_per_task_inst
        task_mses = inst_demo_mses[start:end]
        inst_autocorrs.append(lag1_autocorr(task_mses))
    
    # For control: use specificity_bottom24 (non-instruction 24 heads, 30 demos)
    # and also counterfactual conditions as additional control references
    ctrl_conditions = {}
    for cname in ["specificity_bottom24", "dose_response_top4", "dose_response_top8"]:
        if cname in cond_map:
            c = cond_map[cname]
            mses = np.array(c["demo_mses"])
            n_demos = c["n_demos"]
            # Assume 5 tasks (or however many fit)
            dpt = n_demos // 5 if n_demos >= 5 else n_demos
            autocorrs = []
            for t in range(min(5, n_demos // max(dpt, 1))):
                s, e = t * dpt, (t + 1) * dpt
                if e <= len(mses):
                    autocorrs.append(lag1_autocorr(mses[s:e]))
            ctrl_conditions[cname] = {
                "autocorrs": autocorrs,
                "mean_autocorr": float(np.nanmean(autocorrs)) if autocorrs else None,
            }
    
    inst_mean_autocorr = float(np.nanmean(inst_autocorrs))
    
    # Specificity_bottom24 as primary control comparison
    spec_autocorrs = ctrl_conditions.get("specificity_bottom24", {}).get("autocorrs", [])
    
    # Test: are instruction autocorrelations different from control?
    if len(spec_autocorrs) >= 3 and len(inst_autocorrs) >= 3:
        inst_arr = np.array([x for x in inst_autocorrs if not np.isnan(x)])
        spec_arr = np.array([x for x in spec_autocorrs if not np.isnan(x)])
        if len(inst_arr) >= 2 and len(spec_arr) >= 2:
            # Mann-Whitney U (small sample, non-parametric)
            u_stat, u_p = stats.mannwhitneyu(inst_arr, spec_arr, alternative='two-sided')
            autocorr_test = {"U_statistic": float(u_stat), "p_value": float(u_p), "test": "Mann-Whitney U"}
        else:
            autocorr_test = {"note": "insufficient non-NaN values for test"}
    else:
        autocorr_test = {"note": "insufficient data for test"}
    
    # --- B) Per-dim CV proxy ---
    inst_means = np.array(inst["main_metrics"]["per_dim_action_shift"]["means"])
    inst_stds = np.array(inst["main_metrics"]["per_dim_action_shift"]["stds"])
    ctrl_means = np.array(ctrl["main_metrics"]["per_dim_action_shift"]["means"])
    ctrl_stds = np.array(ctrl["main_metrics"]["per_dim_action_shift"]["stds"])
    
    inst_cv = np.where(inst_means > 1e-12, inst_stds / inst_means, np.inf)
    ctrl_cv = np.where(ctrl_means > 1e-12, ctrl_stds / ctrl_means, np.inf)
    
    # Use first 6 dims (continuous), exclude gripper
    inst_cv_6d = inst_cv[:6]
    ctrl_cv_6d = ctrl_cv[:6]
    
    # Paired Wilcoxon signed-rank test on CVs (6 paired observations)
    cv_stat, cv_p = stats.wilcoxon(inst_cv_6d, ctrl_cv_6d, alternative='less')
    
    # Compute overall autocorrelation for baseline (no ablation) using
    # the variance structure: baseline would have autocorr ≈ 0 if noise is iid
    baseline_autocorr_note = (
        "No baseline (unablated) trajectory data available. "
        "Baseline autocorrelation estimated as ~0 under iid assumption."
    )
    
    return {
        "method": "demo-level lag-1 autocorrelation + per-dim CV proxy",
        "data_limitation": "Raw per-timestep actions not saved; using demo-level MSE sequences as proxy",
        "demo_level_autocorrelation": {
            "instruction_ablated": {
                "per_task_autocorr": [float(x) for x in inst_autocorrs],
                "mean_autocorr": inst_mean_autocorr,
                "n_demos": int(len(inst_demo_mses)),
                "demos_per_task": demos_per_task_inst,
            },
            "control_specificity_bottom24": {
                "per_task_autocorr": [float(x) for x in spec_autocorrs] if spec_autocorrs else None,
                "mean_autocorr": float(np.nanmean(spec_autocorrs)) if spec_autocorrs else None,
            },
            "test": autocorr_test,
        },
        "baseline_autocorr": 0.0,
        "baseline_note": baseline_autocorr_note,
        "per_dim_cv_proxy": {
            "instruction_ablation_mean_cv_6d": float(np.mean(inst_cv_6d)),
            "control_ablation_mean_cv_6d": float(np.mean(ctrl_cv_6d)),
            "wilcoxon_stat": float(cv_stat),
            "wilcoxon_p": float(cv_p),
            "interpretation": (
                "Instruction ablation CV significantly lower than control "
                f"({np.mean(inst_cv_6d):.3f} vs {np.mean(ctrl_cv_6d):.3f}, "
                f"p={cv_p:.4f}), indicating more structured/coherent disruption"
            ) if cv_p < 0.05 else (
                f"No significant CV difference (p={cv_p:.4f})"
            ),
        },
    }


# ─── Analysis 2: Bootstrap CI on Per-Head Effect Size ─────────────────

def bootstrap_per_head_effect_size(inst, ctrl, traj, B=1000, seed=42):
    """
    Bootstrap CI for per-head Cohen's d.
    
    Two approaches:
    A) Task-level bootstrap: resample 5 task d-values with replacement
    B) Demo-level bootstrap: resample demo MSEs from trajectory_level_cohens_d
    """
    rng = np.random.RandomState(seed)
    
    # Per-task d values
    task_names = list(inst["per_task"].keys())
    inst_task_d = np.array([inst["per_task"][t]["paired_mse"]["cohens_d"] for t in task_names])
    ctrl_task_d = np.array([ctrl["per_task"][t]["paired_mse"]["cohens_d"] for t in task_names])
    
    n_inst_heads = len(inst["config"]["ablated_heads"])  # 24
    n_ctrl_heads = len(ctrl["config"]["ablated_heads"])  # 8
    
    inst_per_head_d = inst_task_d / n_inst_heads
    ctrl_per_head_d = ctrl_task_d / n_ctrl_heads
    
    observed_inst_mean = float(np.mean(inst_per_head_d))
    observed_ctrl_mean = float(np.mean(ctrl_per_head_d))
    observed_diff = observed_ctrl_mean - observed_inst_mean
    
    # --- A) Task-level bootstrap (n=5 tasks) ---
    boot_inst_means = np.zeros(B)
    boot_ctrl_means = np.zeros(B)
    boot_diffs = np.zeros(B)
    
    n_tasks = len(task_names)
    for b in range(B):
        idx = rng.choice(n_tasks, size=n_tasks, replace=True)
        boot_inst_means[b] = np.mean(inst_per_head_d[idx])
        boot_ctrl_means[b] = np.mean(ctrl_per_head_d[idx])
        boot_diffs[b] = boot_ctrl_means[b] - boot_inst_means[b]
    
    task_level = {
        "n_tasks": n_tasks,
        "n_bootstrap": B,
        "instruction_heads": {
            "n_heads": n_inst_heads,
            "mean_per_head_d": observed_inst_mean,
            "ci_95": [float(np.percentile(boot_inst_means, 2.5)),
                      float(np.percentile(boot_inst_means, 97.5))],
        },
        "control_heads": {
            "n_heads": n_ctrl_heads,
            "mean_per_head_d": observed_ctrl_mean,
            "ci_95": [float(np.percentile(boot_ctrl_means, 2.5)),
                      float(np.percentile(boot_ctrl_means, 97.5))],
        },
        "difference_ctrl_minus_inst": {
            "observed": observed_diff,
            "ci_95": [float(np.percentile(boot_diffs, 2.5)),
                      float(np.percentile(boot_diffs, 97.5))],
            "excludes_zero": bool(np.percentile(boot_diffs, 2.5) > 0),
            "boot_p_value": float(np.mean(boot_diffs <= 0)),
        },
    }
    
    # --- B) Demo-level bootstrap ---
    cond_map = {r["condition"]: r for r in traj["results"]}
    
    inst_cond = cond_map["head_ablation_top24"]
    inst_demo_mses = np.array(inst_cond["demo_mses"])
    
    # For control at demo level: use specificity_bottom24 (24 non-inst heads)
    ctrl_demo_cond = cond_map.get("specificity_bottom24")
    
    if ctrl_demo_cond:
        ctrl_demo_mses = np.array(ctrl_demo_cond["demo_mses"])
        n_ctrl_demo_heads = 24  # specificity_bottom24 ablates 24 non-instruction heads
        
        # Bootstrap demo-level Cohen's d
        boot_inst_d = np.zeros(B)
        boot_ctrl_d = np.zeros(B)
        
        for b in range(B):
            # Resample instruction demos
            idx_i = rng.choice(len(inst_demo_mses), size=len(inst_demo_mses), replace=True)
            samp_i = inst_demo_mses[idx_i]
            d_i = np.mean(samp_i) / np.std(samp_i, ddof=1) if np.std(samp_i, ddof=1) > 0 else 0
            boot_inst_d[b] = d_i / n_inst_heads
            
            # Resample control demos
            idx_c = rng.choice(len(ctrl_demo_mses), size=len(ctrl_demo_mses), replace=True)
            samp_c = ctrl_demo_mses[idx_c]
            d_c = np.mean(samp_c) / np.std(samp_c, ddof=1) if np.std(samp_c, ddof=1) > 0 else 0
            boot_ctrl_d[b] = d_c / n_ctrl_demo_heads
        
        demo_level = {
            "note": "specificity_bottom24 used as control (24 non-instruction heads, comparable to 24 instruction heads)",
            "instruction_24h": {
                "mean_per_head_d": float(np.mean(boot_inst_d)),
                "ci_95": [float(np.percentile(boot_inst_d, 2.5)),
                          float(np.percentile(boot_inst_d, 97.5))],
                "n_demos": len(inst_demo_mses),
            },
            "non_instruction_24h": {
                "mean_per_head_d": float(np.mean(boot_ctrl_d)),
                "ci_95": [float(np.percentile(boot_ctrl_d, 2.5)),
                          float(np.percentile(boot_ctrl_d, 97.5))],
                "n_demos": len(ctrl_demo_mses),
            },
            "ci_overlap": bool(
                np.percentile(boot_inst_d, 97.5) >= np.percentile(boot_ctrl_d, 2.5) and
                np.percentile(boot_ctrl_d, 97.5) >= np.percentile(boot_inst_d, 2.5)
            ),
        }
    else:
        demo_level = {"note": "specificity_bottom24 not found, skipped demo-level bootstrap"}
    
    # CI overlap check for the main comparison (instruction 24h vs control 8h)
    inst_ci = task_level["instruction_heads"]["ci_95"]
    ctrl_ci = task_level["control_heads"]["ci_95"]
    ci_overlap = inst_ci[1] >= ctrl_ci[0] and ctrl_ci[1] >= inst_ci[0]
    
    return {
        "task_level_bootstrap": task_level,
        "demo_level_bootstrap": demo_level,
        "instruction_heads_mean_d": observed_inst_mean,
        "instruction_heads_ci95": task_level["instruction_heads"]["ci_95"],
        "control_heads_mean_d": observed_ctrl_mean,
        "control_heads_ci95": task_level["control_heads"]["ci_95"],
        "ci_overlap": ci_overlap,
    }


def main():
    print("Loading data...")
    inst, ctrl, traj = load_all_data()
    
    print("Analysis 1: Temporal coherence...")
    tc = temporal_coherence_analysis(inst, ctrl, traj)
    
    print("Analysis 2: Bootstrap CI on per-head effect size (B=1000)...")
    bs = bootstrap_per_head_effect_size(inst, ctrl, traj, B=1000, seed=42)
    
    results = {
        "temporal_coherence": {
            "baseline_autocorr": tc["baseline_autocorr"],
            "instruction_ablated_autocorr": tc["demo_level_autocorrelation"]["instruction_ablated"]["mean_autocorr"],
            "control_ablated_autocorr": tc["demo_level_autocorrelation"]["control_specificity_bottom24"]["mean_autocorr"],
            "test_statistic": tc["demo_level_autocorrelation"]["test"].get("U_statistic", None),
            "p_value": tc["demo_level_autocorrelation"]["test"].get("p_value", None),
            "per_dim_cv": tc["per_dim_cv_proxy"],
            "detail": tc,
        },
        "per_head_effect_size": {
            "instruction_heads_mean_d": bs["instruction_heads_mean_d"],
            "instruction_heads_ci95": bs["instruction_heads_ci95"],
            "control_heads_mean_d": bs["control_heads_mean_d"],
            "control_heads_ci95": bs["control_heads_ci95"],
            "ci_overlap": bs["ci_overlap"],
            "detail": bs,
        },
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_PATH}")
    print("\n=== Summary ===")
    print(f"\n[Temporal Coherence]")
    print(f"  Instruction ablated autocorr: {tc['demo_level_autocorrelation']['instruction_ablated']['mean_autocorr']:.4f}")
    ctrl_ac = tc['demo_level_autocorrelation']['control_specificity_bottom24']['mean_autocorr']
    print(f"  Control ablated autocorr: {ctrl_ac:.4f}" if ctrl_ac else "  Control: N/A")
    print(f"  Per-dim CV (inst vs ctrl): {tc['per_dim_cv_proxy']['instruction_ablation_mean_cv_6d']:.3f} vs {tc['per_dim_cv_proxy']['control_ablation_mean_cv_6d']:.3f}")
    print(f"  Wilcoxon p={tc['per_dim_cv_proxy']['wilcoxon_p']:.4f}")
    
    print(f"\n[Per-Head Effect Size]")
    print(f"  Instruction (24h): d={bs['instruction_heads_mean_d']:.4f}, 95% CI {bs['instruction_heads_ci95']}")
    print(f"  Control (8h):      d={bs['control_heads_mean_d']:.4f}, 95% CI {bs['control_heads_ci95']}")
    print(f"  CI overlap: {bs['ci_overlap']}")
    diff = bs['task_level_bootstrap']['difference_ctrl_minus_inst']
    print(f"  Difference CI: {diff['ci_95']}, excludes_zero={diff['excludes_zero']}")


if __name__ == "__main__":
    main()
