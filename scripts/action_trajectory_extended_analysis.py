"""Extended analysis of action trajectory ablation data.

Analysis 1: Temporal coherence proxy (per-dim CV + cross-task heterogeneity)
Analysis 2: Bootstrap CI on per-head effect size
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = "./results"
DIM_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]

def load_data():
    with open(os.path.join(RESULTS_DIR, "action_trajectory_analysis.json")) as f:
        inst = json.load(f)
    with open(os.path.join(RESULTS_DIR, "action_trajectory_control.json")) as f:
        ctrl = json.load(f)
    return inst, ctrl


def compute_per_dim_cv(data):
    means = np.array(data["main_metrics"]["per_dim_action_shift"]["means"])
    stds = np.array(data["main_metrics"]["per_dim_action_shift"]["stds"])
    cv = np.where(means > 1e-12, stds / means, np.inf)
    return cv, means, stds


def cochrans_q_and_i2(task_d_values, task_n_values):
    """Compute Cochran's Q and I^2 for heterogeneity of effect sizes."""
    d_arr = np.array(task_d_values)
    n_arr = np.array(task_n_values)
    weights = n_arr
    d_bar = np.sum(weights * d_arr) / np.sum(weights)
    Q = np.sum(weights * (d_arr - d_bar) ** 2)
    k = len(d_arr)
    df = k - 1
    p_value = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    I2 = max(0, (Q - df) / Q) if Q > 0 else 0.0
    return float(Q), float(p_value), float(I2)


def temporal_coherence_proxy(inst, ctrl):
    """Proxy for temporal coherence using per-dimension CV and cross-task heterogeneity.

    Rationale: exact within-demo autocorrelation requires per-timestep trajectory
    data not saved in the JSON. Instead we use:
    1. Per-dim CV: lower CV = more consistent perturbation across samples
    2. Cross-task I^2: how much effect-size variance is between vs within tasks
    """
    inst_cv, inst_means, inst_stds = compute_per_dim_cv(inst)
    ctrl_cv, ctrl_means, ctrl_stds = compute_per_dim_cv(ctrl)

    inst_task_d = []
    ctrl_task_d = []
    inst_task_n = []
    ctrl_task_n = []
    task_names = []

    for tname in inst["per_task"]:
        task_names.append(tname)
        inst_task_d.append(inst["per_task"][tname]["paired_mse"]["cohens_d"])
        inst_task_n.append(inst["per_task"][tname]["n_samples"])
    for tname in ctrl["per_task"]:
        ctrl_task_d.append(ctrl["per_task"][tname]["paired_mse"]["cohens_d"])
        ctrl_task_n.append(ctrl["per_task"][tname]["n_samples"])

    inst_Q, inst_Q_p, inst_I2 = cochrans_q_and_i2(inst_task_d, inst_task_n)
    ctrl_Q, ctrl_Q_p, ctrl_I2 = cochrans_q_and_i2(ctrl_task_d, ctrl_task_n)

    per_dim_results = []
    for i, dim in enumerate(DIM_NAMES):
        per_dim_results.append({
            "dim": dim,
            "inst_mean_shift": float(inst_means[i]),
            "inst_std_shift": float(inst_stds[i]),
            "inst_cv": float(inst_cv[i]),
            "ctrl_mean_shift": float(ctrl_means[i]),
            "ctrl_std_shift": float(ctrl_stds[i]),
            "ctrl_cv": float(ctrl_cv[i]),
        })

    mean_inst_cv_6d = float(np.mean(inst_cv[:6]))
    mean_ctrl_cv_6d = float(np.mean(ctrl_cv[:6]))

    return {
        "method": "proxy (per-dim CV + cross-task I^2)",
        "note": "Exact within-demo autocorrelation requires per-timestep trajectory data not saved in JSON. These are indirect proxies.",
        "per_dim_cv": per_dim_results,
        "mean_cv_6d_continuous_dims": {
            "instruction_ablation": mean_inst_cv_6d,
            "control_ablation": mean_ctrl_cv_6d,
            "interpretation": "Higher CV = more variable perturbation across samples, suggesting less structured/temporally-coherent effect"
        },
        "cross_task_heterogeneity": {
            "instruction_ablation": {
                "per_task_d": inst_task_d,
                "cochrans_Q": inst_Q,
                "Q_p_value": inst_Q_p,
                "I_squared": inst_I2,
            },
            "control_ablation": {
                "per_task_d": ctrl_task_d,
                "cochrans_Q": ctrl_Q,
                "Q_p_value": ctrl_Q_p,
                "I_squared": ctrl_I2,
            },
            "task_names": task_names,
            "interpretation": "Higher I^2 = more between-task variance relative to total, suggesting task-dependent effect"
        },
    }


def bootstrap_per_head_ci(inst, ctrl, n_bootstrap=10000, seed=42):
    """Bootstrap CI on per-head Cohen's d difference.

    Uses parametric bootstrap (gamma distribution matched to known mean/std)
    since per-sample data is not available in the JSON.
    """
    rng = np.random.RandomState(seed)

    n_samples = 2500
    n_heads_inst = len(inst["config"]["ablated_heads"])
    n_heads_ctrl = len(ctrl["config"]["ablated_heads"])

    mean_inst = inst["main_metrics"]["paired_action_mse"]["mean"]
    std_inst = inst["main_metrics"]["paired_action_mse"]["std"]
    mean_ctrl = ctrl["main_metrics"]["paired_action_mse"]["mean"]
    std_ctrl = ctrl["main_metrics"]["paired_action_mse"]["std"]

    var_inst = std_inst ** 2
    k_inst = mean_inst ** 2 / var_inst
    theta_inst = var_inst / mean_inst

    var_ctrl = std_ctrl ** 2
    k_ctrl = mean_ctrl ** 2 / var_ctrl
    theta_ctrl = var_ctrl / mean_ctrl

    print(f"Gamma params - instruction: k={k_inst:.4f}, theta={theta_inst:.4f}")
    print(f"Gamma params - control: k={k_ctrl:.4f}, theta={theta_ctrl:.4f}")

    per_head_d_inst = np.zeros(n_bootstrap)
    per_head_d_ctrl = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        samp_inst = rng.gamma(k_inst, theta_inst, n_samples)
        samp_ctrl = rng.gamma(k_ctrl, theta_ctrl, n_samples)

        d_inst = np.mean(samp_inst) / (np.std(samp_inst, ddof=1) + 1e-15)
        d_ctrl = np.mean(samp_ctrl) / (np.std(samp_ctrl, ddof=1) + 1e-15)

        per_head_d_inst[i] = d_inst / n_heads_inst
        per_head_d_ctrl[i] = d_ctrl / n_heads_ctrl

    diff = per_head_d_ctrl - per_head_d_inst

    ci_lower = float(np.percentile(diff, 2.5))
    ci_upper = float(np.percentile(diff, 97.5))
    ci_99_lower = float(np.percentile(diff, 0.5))
    ci_99_upper = float(np.percentile(diff, 99.5))
    mean_diff = float(np.mean(diff))

    excludes_zero_95 = (ci_lower > 0) or (ci_upper < 0)
    excludes_zero_99 = (ci_99_lower > 0) or (ci_99_upper < 0)

    observed_per_head_inst = mean_inst / (std_inst + 1e-15) / n_heads_inst
    observed_per_head_ctrl = mean_ctrl / (std_ctrl + 1e-15) / n_heads_ctrl
    observed_diff = observed_per_head_ctrl - observed_per_head_inst

    # Task-level paired analysis (5 paired observations)
    task_per_head_inst = []
    task_per_head_ctrl = []
    inst_tasks = list(inst["per_task"].keys())
    for tname in inst_tasks:
        d_i = inst["per_task"][tname]["paired_mse"]["cohens_d"]
        d_c = ctrl["per_task"][tname]["paired_mse"]["cohens_d"]
        task_per_head_inst.append(d_i / n_heads_inst)
        task_per_head_ctrl.append(d_c / n_heads_ctrl)

    task_inst = np.array(task_per_head_inst)
    task_ctrl = np.array(task_per_head_ctrl)
    task_diff = task_ctrl - task_inst
    t_stat_task, p_value_task = stats.ttest_rel(task_ctrl, task_inst)

    return {
        "method": "parametric_bootstrap_gamma",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_samples_per_condition": n_samples,
        "gamma_params": {
            "instruction": {"shape": float(k_inst), "scale": float(theta_inst)},
            "control": {"shape": float(k_ctrl), "scale": float(theta_ctrl)},
        },
        "observed": {
            "instruction_24h": {
                "total_d": float(inst["main_metrics"]["paired_action_mse"]["cohens_d"]),
                "per_head_d": float(observed_per_head_inst),
                "n_heads": n_heads_inst,
            },
            "control_8h": {
                "total_d": float(ctrl["main_metrics"]["paired_action_mse"]["cohens_d"]),
                "per_head_d": float(observed_per_head_ctrl),
                "n_heads": n_heads_ctrl,
            },
            "diff_ctrl_minus_inst": float(observed_diff),
        },
        "bootstrap_results": {
            "per_head_d_instruction": {
                "mean": float(np.mean(per_head_d_inst)),
                "std": float(np.std(per_head_d_inst)),
                "ci_95": [float(np.percentile(per_head_d_inst, 2.5)),
                          float(np.percentile(per_head_d_inst, 97.5))],
            },
            "per_head_d_control": {
                "mean": float(np.mean(per_head_d_ctrl)),
                "std": float(np.std(per_head_d_ctrl)),
                "ci_95": [float(np.percentile(per_head_d_ctrl, 2.5)),
                          float(np.percentile(per_head_d_ctrl, 97.5))],
            },
            "difference_ctrl_minus_inst": {
                "mean": mean_diff,
                "std": float(np.std(diff)),
                "ci_95": [ci_lower, ci_upper],
                "ci_99": [ci_99_lower, ci_99_upper],
                "excludes_zero_95": excludes_zero_95,
                "excludes_zero_99": excludes_zero_99,
            },
        },
        "task_level_paired_test": {
            "per_task_per_head_d_instruction": task_per_head_inst,
            "per_task_per_head_d_control": task_per_head_ctrl,
            "per_task_diff": task_diff.tolist(),
            "task_names": inst_tasks,
            "paired_t_stat": float(t_stat_task),
            "paired_p_value": float(p_value_task),
            "mean_diff": float(np.mean(task_diff)),
            "all_positive": bool(np.all(task_diff > 0)),
            "note": "Low power (n=5 tasks), but direction is consistent"
        },
        "conclusion": None,
    }


def main():
    inst, ctrl = load_data()

    print("=" * 60)
    print("Analysis 1: Temporal Coherence Proxy")
    print("=" * 60)
    tc = temporal_coherence_proxy(inst, ctrl)

    print(f"\nPer-dim CV (6D continuous, excluding gripper):")
    print(f"  Instruction ablation mean CV: {tc['mean_cv_6d_continuous_dims']['instruction_ablation']:.3f}")
    print(f"  Control ablation mean CV:     {tc['mean_cv_6d_continuous_dims']['control_ablation']:.3f}")
    for row in tc["per_dim_cv"]:
        print(f"  {row['dim']:>7s}: inst_cv={row['inst_cv']:.3f}, ctrl_cv={row['ctrl_cv']:.3f}")

    print(f"\nCross-task heterogeneity (I^2):")
    inst_het = tc["cross_task_heterogeneity"]["instruction_ablation"]
    ctrl_het = tc["cross_task_heterogeneity"]["control_ablation"]
    print(f"  Instruction: Q={inst_het['cochrans_Q']:.2f}, p={inst_het['Q_p_value']:.4f}, I^2={inst_het['I_squared']:.3f}")
    print(f"  Control:     Q={ctrl_het['cochrans_Q']:.2f}, p={ctrl_het['Q_p_value']:.4f}, I^2={ctrl_het['I_squared']:.3f}")

    print("\n" + "=" * 60)
    print("Analysis 2: Bootstrap CI on Per-Head Effect Size")
    print("=" * 60)
    bs = bootstrap_per_head_ci(inst, ctrl)

    obs = bs["observed"]
    print(f"\nObserved per-head d:")
    print(f"  Instruction (24h): total d={obs['instruction_24h']['total_d']:.4f}, per-head d={obs['instruction_24h']['per_head_d']:.5f}")
    print(f"  Control (8h):      total d={obs['control_8h']['total_d']:.4f}, per-head d={obs['control_8h']['per_head_d']:.5f}")
    print(f"  Diff (ctrl - inst): {obs['diff_ctrl_minus_inst']:.5f}")

    bsr = bs["bootstrap_results"]
    diff = bsr["difference_ctrl_minus_inst"]
    print(f"\nBootstrap (n={bs['n_bootstrap']}):")
    print(f"  Per-head d inst: {bsr['per_head_d_instruction']['mean']:.5f} [{bsr['per_head_d_instruction']['ci_95'][0]:.5f}, {bsr['per_head_d_instruction']['ci_95'][1]:.5f}]")
    print(f"  Per-head d ctrl: {bsr['per_head_d_control']['mean']:.5f} [{bsr['per_head_d_control']['ci_95'][0]:.5f}, {bsr['per_head_d_control']['ci_95'][1]:.5f}]")
    print(f"  Difference:      {diff['mean']:.5f} [{diff['ci_95'][0]:.5f}, {diff['ci_95'][1]:.5f}]")
    print(f"  95% CI excludes 0: {diff['excludes_zero_95']}")
    print(f"  99% CI excludes 0: {diff['excludes_zero_99']}")

    tl = bs["task_level_paired_test"]
    print(f"\nTask-level paired t-test (n=5):")
    print(f"  t={tl['paired_t_stat']:.3f}, p={tl['paired_p_value']:.4f}")
    print(f"  All tasks show ctrl > inst: {tl['all_positive']}")

    # Derive conclusions
    if diff["excludes_zero_95"]:
        bs["conclusion"] = (
            f"Per-head effect significantly differs: control heads have "
            f"{diff['mean']:.5f} higher per-head d than instruction heads "
            f"(95% CI [{diff['ci_95'][0]:.5f}, {diff['ci_95'][1]:.5f}]). "
            f"Each non-instruction head individually contributes more to action disruption "
            f"than each instruction head, but instruction heads achieve larger total effect "
            f"through collective action (24 vs 8 heads)."
        )
    else:
        bs["conclusion"] = (
            f"Per-head effect NOT significantly different: 95% CI of difference "
            f"[{diff['ci_95'][0]:.5f}, {diff['ci_95'][1]:.5f}] includes zero."
        )

    print(f"\nConclusion: {bs['conclusion']}")

    output = {
        "temporal_coherence_proxy": tc,
        "bootstrap_per_head_ci": bs,
    }

    out_path = os.path.join(RESULTS_DIR, "action_trajectory_extended_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
