import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

with open('./results/attention/full_32layer_attention_summary.json') as f:
    data = json.load(f)

layers = list(range(32))

trained_heads = [data['trained'][f'L{i}']['n_instruction_dominant'] for i in layers]
llama_heads = [data['llama2base'][f'L{i}']['n_instruction_dominant'] for i in layers]
untrained_heads = [data['untrained'][f'L{i}']['n_instruction_dominant'] for i in layers]

trained_attn = [data['trained'][f'L{i}']['mean_text_attn'] for i in layers]
llama_attn = [data['llama2base'][f'L{i}']['mean_text_attn'] for i in layers]
untrained_attn = [data['untrained'][f'L{i}']['mean_text_attn'] for i in layers]

colors = {'trained': '#4C72B0', 'llama': '#55A868', 'untrained': '#C44E52'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

# Panel (a): Instruction-dominant heads
ax1.plot(layers, trained_heads, color=colors['trained'], linestyle='-', linewidth=2, label='VLA Trained')
ax1.plot(layers, llama_heads, color=colors['llama'], linestyle='--', linewidth=2, label='Llama-2 Base')
ax1.plot(layers, untrained_heads, color=colors['untrained'], linestyle=':', linewidth=2, label='Random Init')
ax1.axhline(y=16, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax1.set_xlabel('Layer', fontsize=9)
ax1.set_ylabel('Instruction-dominant heads', fontsize=9)
ax1.set_xlim(-0.5, 31.5)
ax1.set_ylim(-1, 33)
ax1.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax1.tick_params(labelsize=8)
ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
ax1.grid(axis='y', color='lightgray', linewidth=0.5, alpha=0.7)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

# Annotations for panel (a)
ax1.annotate('entry\nrouting', xy=(1, 24), xytext=(5, 29),
             fontsize=7, color=colors['trained'], alpha=0.85,
             arrowprops=dict(arrowstyle='->', color=colors['trained'], alpha=0.5, lw=0.8),
             ha='center', va='center')

ax1.annotate('exit\nrouting', xy=(31, 18), xytext=(27, 27),
             fontsize=7, color=colors['trained'], alpha=0.85,
             arrowprops=dict(arrowstyle='->', color=colors['trained'], alpha=0.5, lw=0.8),
             ha='center', va='center')

peak_llama_idx = 10
ax1.annotate('mid-layer\nprocessing', xy=(peak_llama_idx, 22), xytext=(15, 29),
             fontsize=7, color=colors['llama'], alpha=0.85,
             arrowprops=dict(arrowstyle='->', color=colors['llama'], alpha=0.5, lw=0.8),
             ha='center', va='center')

# Panel (b): Mean text attention fraction
ax2.plot(layers, trained_attn, color=colors['trained'], linestyle='-', linewidth=2, label='VLA Trained')
ax2.plot(layers, llama_attn, color=colors['llama'], linestyle='--', linewidth=2, label='Llama-2 Base')
ax2.plot(layers, untrained_attn, color=colors['untrained'], linestyle=':', linewidth=2, label='Random Init')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax2.set_xlabel('Layer', fontsize=9)
ax2.set_ylabel('Mean instruction attention fraction', fontsize=9)
ax2.set_xlim(-0.5, 31.5)
ax2.set_ylim(0.0, 0.7)
ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax2.tick_params(labelsize=8)
ax2.grid(axis='y', color='lightgray', linewidth=0.5, alpha=0.7)
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

# Unified legend at bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
           frameon=True, edgecolor='lightgray', fancybox=False,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out_dir = './results/attention'
fig.savefig(f'{out_dir}/fig_full_32layer_attention.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig_full_32layer_attention.png', dpi=300, bbox_inches='tight')
print('Saved PDF and PNG')
