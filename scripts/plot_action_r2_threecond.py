import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data_path = './results/llama2base_action_probe/three_condition_comparison.json'
with open(data_path) as f:
    data = json.load(f)

layers = []
untrained_r2 = []
llama2base_r2 = []
trained_r2 = []

for i in range(33):
    key = f'L{i}'
    entry = data['per_layer_comparison'][key]
    layers.append(i)
    untrained_r2.append(entry['untrained_R2'])
    llama2base_r2.append(entry['llama2base_R2'])
    trained_r2.append(entry['trained_R2'])

layers = np.array(layers)
untrained_r2 = np.array(untrained_r2)
llama2base_r2 = np.array(llama2base_r2)
trained_r2 = np.array(trained_r2)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(layers, untrained_r2, color='#1f77b4', marker='o', markersize=4.5,
        linewidth=1.8, label='Reference VLA', zorder=3)
ax.plot(layers, llama2base_r2, color='#ff7f0e', marker='^', markersize=4.5,
        linewidth=1.8, label='Llama-2-7B Base', zorder=3)
ax.plot(layers, trained_r2, color='#2ca02c', marker='s', markersize=4.5,
        linewidth=1.8, label='Trained VLA (OpenVLA)', zorder=3)

# L0 annotation
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.annotate('Shared visual\nprojector output', xy=(0, 0.64), xytext=(2.5, -0.25),
            fontsize=7.5, color='gray', ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Trained peak at L8
ax.annotate('Peak (L8)', xy=(8, trained_r2[8]), xytext=(11, 0.68),
            fontsize=7.5, color='#2ca02c', ha='left',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=0.8))

# Collapsed layers shading (L10-L11)
ax.axvspan(9.5, 11.5, alpha=0.12, color='red', zorder=1)
ax.text(10.5, -0.38, 'Collapsed\nlayers', fontsize=7, color='#c0392b',
        ha='center', va='bottom')

ax.set_xlabel('Layer Index', fontsize=10)
ax.set_ylabel('Action R² (image_mean)', fontsize=10)
ax.set_xlim(-0.5, 32.5)
ax.set_ylim(-0.45, 0.75)
ax.set_xticks(np.arange(0, 33, 4))
ax.tick_params(labelsize=8.5)
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

plt.tight_layout()

out_dir = './results/figures'
fig.savefig(f'{out_dir}/fig_action_r2_threecond.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig_action_r2_threecond.png', dpi=300, bbox_inches='tight')
print('Saved PDF and PNG to:', out_dir)
plt.close()
