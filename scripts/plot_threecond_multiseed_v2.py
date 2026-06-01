import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

with open('./results/threecond_multiseed_v2.json') as f:
    data = json.load(f)

results = data['results']
num_layers = 33
layers = np.arange(num_layers)

conditions = [
    ('untrained', 'Reference', '#1f77b4'),
    ('llama2base', 'LLaMA-2 Base', '#ff7f0e'),
    ('trained', 'Trained (OpenVLA)', '#2ca02c'),
]

fig, ax = plt.subplots(figsize=(8, 4.5))

for cond_key, label, color in conditions:
    means = []
    stds = []
    for i in range(num_layers):
        layer_data = results[cond_key][f'L{i}']
        means.append(layer_data['mean'])
        stds.append(layer_data['std'])
    means = np.array(means)
    stds = np.array(stds)
    ax.plot(layers, means, color=color, linewidth=2.0, label=label)
    ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.2)

ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel(r'Action R$^2$ (7-dim)', fontsize=11)
ax.set_xlim(0, 32)
ax.tick_params(labelsize=9)
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()

out_dir = './results/figures'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig_threecond_multiseed_v2.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_threecond_multiseed_v2.png'), dpi=300, bbox_inches='tight')
print('Saved PDF and PNG.')
