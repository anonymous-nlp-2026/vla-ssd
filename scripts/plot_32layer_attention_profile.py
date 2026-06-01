import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

with open('./results/attention/full_32layer_attention_summary.json') as f:
    data = json.load(f)

layers = list(range(32))
trained = [data['trained'][f'L{i}']['n_instruction_dominant'] for i in layers]
llama2base = [data['llama2base'][f'L{i}']['n_instruction_dominant'] for i in layers]
untrained = [data['untrained'][f'L{i}']['n_instruction_dominant'] for i in layers]

fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(layers, trained, color='#4C72B0', linestyle='-', linewidth=1.8, label='VLA Trained')
ax.plot(layers, llama2base, color='#55A868', linestyle='--', linewidth=1.8, label='Llama-2 Base')
ax.plot(layers, untrained, color='#888888', linestyle=':', linewidth=1.5, label='Reference VLA')

# Highlight bands
ax.axvspan(0, 1.5, alpha=0.1, color='#4C72B0', zorder=0)
ax.axvspan(9, 16, alpha=0.08, color='#55A868', zorder=0)

# Annotations
ax.annotate('entry\nrouting', xy=(0.75, 22.5), fontsize=8, color='#4C72B0',
            ha='center', va='bottom')
ax.annotate('exit\nrouting', xy=(31, 18), fontsize=8, color='#4C72B0',
            ha='right', va='bottom', xytext=(29.5, 20),
            arrowprops=dict(arrowstyle='-', color='#4C72B0', lw=0.5))
ax.annotate('mid-layer\nprocessing', xy=(12.5, 23), fontsize=8, color='#55A868',
            ha='center', va='bottom')

# 16/32 reference line
ax.axhline(y=16, color='#AAAAAA', linestyle='--', linewidth=0.7, alpha=0.3, zorder=0)
ax.text(31.3, 16, '16', fontsize=8, color='#AAAAAA', va='center')

# Grid
ax.yaxis.grid(True, alpha=0.2, color='#CCCCCC')
ax.set_axisbelow(True)

# Axes
ax.set_xlim(-0.5, 31.5)
ax.set_ylim(0, 32)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
ax.set_yticks([0, 8, 16, 24, 32])
ax.set_xlabel('Layer')
ax.set_ylabel('Instruction-dominant heads')

# Legend
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./scripts/fig_32layer_attention_profile.pdf', bbox_inches='tight', dpi=300)
plt.savefig('./scripts/fig_32layer_attention_profile.png', bbox_inches='tight', dpi=150)
print('Saved.')
