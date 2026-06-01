"""Plot OOD diagnostics: 3-panel figure."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_JSON = "./results/ood_diagnostics.json"
OUT_DIR = "./results/figures/"

with open(IN_JSON) as f:
    data = json.load(f)

layers = list(range(33))
norm_l2b = [data["per_layer"][f"layer_{l}"]["activation_norm_llama2base"] for l in layers]
norm_tr = [data["per_layer"][f"layer_{l}"]["activation_norm_trained"] for l in layers]
cos_mean = [data["per_layer"][f"layer_{l}"]["cosine_sim_mean"] for l in layers]
cos_std = [data["per_layer"][f"layer_{l}"]["cosine_sim_std"] for l in layers]
var_l2b = [data["per_layer"][f"layer_{l}"]["variance_explained_top10_llama2base"] for l in layers]
var_tr = [data["per_layer"][f"layer_{l}"]["variance_explained_top10_trained"] for l in layers]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(layers, norm_l2b, "o-", color="#2196F3", markersize=4, linewidth=1.5, label="Llama-2 base")
ax.plot(layers, norm_tr, "s-", color="#FF5722", markersize=4, linewidth=1.5, label="Trained (OpenVLA)")
ax.set_xlabel("Layer", fontsize=10)
ax.set_ylabel("Mean L2 Norm", fontsize=10)
ax.set_title("(A) Activation Magnitude", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)
ax.set_xlim(-0.5, 32.5)

ax = axes[1]
cos_mean_arr = np.array(cos_mean)
cos_std_arr = np.array(cos_std)
ax.plot(layers, cos_mean, "o-", color="#4CAF50", markersize=4, linewidth=1.5)
ax.fill_between(layers, cos_mean_arr - cos_std_arr, cos_mean_arr + cos_std_arr,
                alpha=0.2, color="#4CAF50")
ax.set_xlabel("Layer", fontsize=10)
ax.set_ylabel("Cosine Similarity", fontsize=10)
ax.set_title("(B) Llama-2 Base vs Trained", fontsize=11)
ax.tick_params(labelsize=8)
ax.set_xlim(-0.5, 32.5)
ax.set_ylim(0, 1.05)

ax = axes[2]
ax.plot(layers, var_l2b, "o-", color="#2196F3", markersize=4, linewidth=1.5, label="Llama-2 base")
ax.plot(layers, var_tr, "s-", color="#FF5722", markersize=4, linewidth=1.5, label="Trained (OpenVLA)")
ax.set_xlabel("Layer", fontsize=10)
ax.set_ylabel("Var. Explained (top 10 PCs)", fontsize=10)
ax.set_title("(C) Effective Dimensionality", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)
ax.set_xlim(-0.5, 32.5)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "fig_ood_diagnostics.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_ood_diagnostics.png"), dpi=200, bbox_inches="tight")
print("Saved to", OUT_DIR)
