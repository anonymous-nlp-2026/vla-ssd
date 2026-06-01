import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BASE = './results'

def load_r2(path, n_layers, readout='last_preaction'):
    with open(path) as f:
        data = json.load(f)
    r2 = []
    for i in range(n_layers):
        key = f'L{i}_{readout}'
        r2.append(data[key]['mean'])
    return np.array(r2)

# --- Load data ---
ovla_trained   = load_r2(f'{BASE}/libero_goal/action_r2_probing.json', 32)
ovla_untrained = load_r2(f'{BASE}/libero_goal_untrained/action_r2_probing.json', 32)

trac_trained   = load_r2(f'{BASE}/tracvla_goal/trained/action_r2_probing.json', 32)
trac_untrained = load_r2(f'{BASE}/tracvla_goal/untrained/action_r2_probing.json', 32)

nora_trained   = load_r2(f'{BASE}/nora_goal/trained/action_r2_probing.json', 36)
nora_untrained = load_r2(f'{BASE}/nora_goal/untrained/action_r2_probing.json', 36)

NLP_BASELINE = 0.155

# --- Colors ---
C_TRAINED   = '#2563eb'
C_UNTRAINED = '#dc2626'
C_NLP       = '#6b7280'

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

configs = [
    ('OpenVLA (Llama-2-7B)',    ovla_trained,  ovla_untrained,  32, True),
    ('TracVLA (Phi-3-mini)',    trac_trained,  trac_untrained,  32, False),
    ('NORA (Qwen2.5-VL-3B)',   nora_trained,  nora_untrained,  36, False),
]

for ax, (title, trained, untrained, n_layers, show_nlp) in zip(axes, configs):
    layers = np.arange(n_layers)

    ax.plot(layers, trained, color=C_TRAINED, linewidth=2.0, marker='o',
            markersize=3, label='Trained', zorder=4)
    ax.plot(layers, untrained, color=C_UNTRAINED, linewidth=1.5, linestyle='--',
            marker='s', markersize=2.5, label='Untrained', zorder=3)

    if show_nlp:
        ax.axhline(y=NLP_BASELINE, color=C_NLP, linestyle=':', linewidth=1.5,
                   alpha=0.7, label='Llama-2-chat (NLP)', zorder=2)

    # Peak annotation for trained
    peak_idx = np.argmax(trained)
    peak_val = trained[peak_idx]
    exit_val = trained[-1]
    delta = peak_val - exit_val

    ax.annotate(f'peak={peak_val:.3f} (L{peak_idx})',
                xy=(peak_idx, peak_val),
                xytext=(peak_idx + n_layers*0.08, peak_val + 0.03),
                fontsize=8, color=C_TRAINED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_TRAINED, lw=0.8))

    ax.annotate(f'$\\Delta$={delta:.3f}',
                xy=(n_layers-1, exit_val),
                xytext=(n_layers*0.75, exit_val - 0.08),
                fontsize=8, color=C_TRAINED,
                arrowprops=dict(arrowstyle='->', color=C_TRAINED, lw=0.8))

    ax.set_title(title, fontweight='semibold')
    ax.set_xlabel('Layer Index')
    if ax == axes[0]:
        ax.set_ylabel('$R^2$ (Action Prediction)')
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.05, 0.95)
    ax.grid(True, linestyle='--', alpha=0.25, color='gray')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

plt.tight_layout()

out_pdf = '/tmp/fig_r2_collapse.pdf'
out_png = '/tmp/fig_r2_collapse.png'
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f'Saved: {out_pdf} ({__import__("os").path.getsize(out_pdf)} bytes)')
print(f'Saved: {out_png} ({__import__("os").path.getsize(out_png)} bytes)')
plt.close()
