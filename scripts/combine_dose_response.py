"""Combine all 8 dose-response JSON files into one."""
import json
import os

results_dir = './results'
conditions = [4, 8, 12, 16, 20, 24, 28, 32]
combined = {"conditions": [], "summary": {}}

for k in conditions:
    path = f'{results_dir}/dose_response_top{k}.json'
    with open(path) as f:
        data = json.load(f)
    combined["conditions"].append(data)

combined["summary"] = {
    "n_heads_ablated": conditions,
    "cohens_d": [c["paired_mse"]["cohens_d"] for c in combined["conditions"]],
    "ci_95_lower": [c["paired_mse"]["ci_95"][0] for c in combined["conditions"]],
    "ci_95_upper": [c["paired_mse"]["ci_95"][1] for c in combined["conditions"]],
    "token_divergence_rate": [c["token_divergence_rate"] for c in combined["conditions"]],
    "kl_divergence_mean": [c["kl_divergence"]["mean"] for c in combined["conditions"]],
    "paired_mse_mean": [c["paired_mse"]["mean"] for c in combined["conditions"]],
}

out_path = f'{results_dir}/dose_response_curve_combined.json'
with open(out_path, 'w') as f:
    json.dump(combined, f, indent=2)
print(f"Combined data saved to {out_path}")
