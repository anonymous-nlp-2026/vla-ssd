import json
import os

RESULTS_DIR = "./results/rsa"

# Load all data sources
with open(os.path.join(RESULTS_DIR, "rsa_siglip_only.json")) as f:
    siglip = json.load(f)

with open(os.path.join(RESULTS_DIR, "rsa_results.json")) as f:
    with_inst = json.load(f)

with open(os.path.join(RESULTS_DIR, "rsa_results_no_inst.json")) as f:
    no_inst = json.load(f)

siglip_rsa = siglip["siglip_only_peak_rsa"]  # 0.6153

# Extract per-layer RSA for all 4 conditions
layers = list(range(33))  # 0-32
layer_keys = [f"layer_{i}" for i in layers]

wi_trained = [with_inst["per_layer"][k]["trained_rsa"] for k in layer_keys]
wi_untrained = [with_inst["per_layer"][k]["untrained_rsa"] for k in layer_keys]
ni_trained = [no_inst["per_layer"][k]["trained_rsa"] for k in layer_keys]
ni_untrained = [no_inst["per_layer"][k]["untrained_rsa"] for k in layer_keys]

# Build degradation curves (projector -> L0 -> L1 -> ... -> L32)
def build_curve(per_layer_values):
    return [siglip_rsa] + per_layer_values

def find_max_drop(curve, labels):
    max_delta = 0
    from_idx, to_idx = 0, 1
    for i in range(len(curve) - 1):
        delta = curve[i] - curve[i+1]
        if delta > max_delta:
            max_delta = delta
            from_idx, to_idx = i, i+1
    return {
        "from_layer": labels[from_idx],
        "to_layer": labels[to_idx],
        "delta": round(max_delta, 4)
    }

def curve_pattern(curve):
    increases = sum(1 for i in range(1, len(curve)) if curve[i] > curve[i-1])
    decreases = sum(1 for i in range(1, len(curve)) if curve[i] < curve[i-1])
    if decreases > increases * 2:
        return "predominantly_decreasing"
    elif increases > decreases * 2:
        return "predominantly_increasing"
    else:
        return "non_monotonic"

labels = ["projector"] + layers
curves = {
    "with_inst_trained": build_curve(wi_trained),
    "with_inst_untrained": build_curve(wi_untrained),
    "no_inst_trained": build_curve(ni_trained),
    "no_inst_untrained": build_curve(ni_untrained),
}

# Analysis
analysis = {}
for name, curve in curves.items():
    # Skip layer 0 anomaly (0.0) for trained models when finding meaningful drops
    proj_to_L1 = round(curve[0] - curve[2], 4)  # projector -> L1 (skip L0=0.0)
    L1_to_L32 = round(curve[2] - curve[-1], 4)
    peak_val = max(curve[1:])  # peak among LLM layers (exclude projector)
    peak_idx = curve[1:].index(peak_val) + 1
    
    analysis[name] = {
        "projector_rsa": curve[0],
        "peak_llm_rsa": round(peak_val, 4),
        "peak_llm_layer": labels[peak_idx],
        "total_degradation": round(curve[0] - peak_val, 4),
        "proj_to_L1_drop": proj_to_L1,
        "L1_to_L32_change": L1_to_L32,
        "max_single_step_drop": find_max_drop(curve, labels),
        "pattern_after_L1": curve_pattern(curve[2:]),  # pattern from L1 onward
    }

# Trained vs untrained divergence analysis
def find_divergence(t_curve, u_curve, labels):
    diffs = [round(t_curve[i] - u_curve[i], 4) for i in range(len(t_curve))]
    max_diff_idx = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
    return {
        "per_layer_delta": {labels[i]: diffs[i] for i in range(len(diffs))},
        "max_divergence_layer": labels[max_diff_idx],
        "max_divergence_value": diffs[max_diff_idx],
    }

divergence_with_inst = find_divergence(
    curves["with_inst_trained"], curves["with_inst_untrained"], labels
)
divergence_no_inst = find_divergence(
    curves["no_inst_trained"], curves["no_inst_untrained"], labels
)

# Output
output = {
    "layers": labels,
    "curves": {
        "with_inst_trained": [round(v, 4) for v in curves["with_inst_trained"]],
        "with_inst_untrained": [round(v, 4) for v in curves["with_inst_untrained"]],
        "no_inst_trained": [round(v, 4) for v in curves["no_inst_trained"]],
        "no_inst_untrained": [round(v, 4) for v in curves["no_inst_untrained"]],
    },
    "reference_lines": {
        "siglip_projector": siglip_rsa,
        "dino": 0.388,
    },
    "per_condition_analysis": analysis,
    "trained_vs_untrained_divergence": {
        "with_inst": divergence_with_inst,
        "no_inst": divergence_no_inst,
    },
    "summary": {},
}

# Generate summary
# Key finding: projector->L0 is the biggest drop for all conditions
all_proj_drops = {k: v["total_degradation"] for k, v in analysis.items()}
output["summary"] = {
    "main_finding": (
        f"SigLIP+projector RSA ({siglip_rsa}) drops sharply at LLM entry. "
        f"Best LLM-layer RSA across conditions: "
        f"with-inst trained peak={analysis['with_inst_trained']['peak_llm_rsa']} (layer {analysis['with_inst_trained']['peak_llm_layer']}), "
        f"no-inst trained peak={analysis['no_inst_trained']['peak_llm_rsa']} (layer {analysis['no_inst_trained']['peak_llm_layer']}). "
        f"Degradation range: {min(all_proj_drops.values()):.4f} to {max(all_proj_drops.values()):.4f}."
    ),
    "layer_0_anomaly": (
        "Layer 0 (embedding) shows RSA=0.0 for trained models in both conditions, "
        "likely due to token embedding not preserving continuous action structure. "
        "Untrained no-inst L0=0.2025; all other L0=0.0."
    ),
}

out_path = os.path.join(RESULTS_DIR, "perlayer_degradation_curve.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved to {out_path}")
print(f"\n=== Quick Summary ===")
for cond, a in analysis.items():
    print(f"\n{cond}:")
    print(f"  Projector RSA: {a['projector_rsa']}")
    print(f"  Peak LLM RSA: {a['peak_llm_rsa']} @ layer {a['peak_llm_layer']}")
    print(f"  Total degradation: {a['total_degradation']}")
    print(f"  Proj->L1 drop: {a['proj_to_L1_drop']}")
    print(f"  Pattern after L1: {a['pattern_after_L1']}")
