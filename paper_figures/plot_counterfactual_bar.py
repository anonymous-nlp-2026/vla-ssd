import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

colors = {
    'empty':       '#E69F00',
    'wrong':       '#56B4E9',
    'shuffled':    '#009E73',
    'paraphrased': '#CC79A7',
}
conds = ['empty', 'wrong', 'shuffled', 'paraphrased']

# ── Panel A: Action-space Cohen's d (order: empty, wrong, shuffled, paraphrased) ──
pa = {
    'OpenVLA\nGoal':    [1.170, 1.182, 1.096, 0.996],
    'OpenVLA\nSpatial': [1.173, 0.885, 0.979, 0.807],
    'OpenVLA\nObject':  [1.212, 1.188, 1.122, 0.896],
    'OpenVLA\nLong':    [0.505, 0.545, 0.565, 0.449],
    'TracVLA\nGoal':    [1.524, 1.405, 1.336, 1.213],
    'TracVLA\nSpatial': [1.327, 1.172, 1.295, 1.183],
    'TracVLA\nObject':  [1.212, 1.060, 1.164, 0.971],
    'TracVLA\nLong':    [1.473, 1.418, 1.391, 1.161],
}

# ── Panel B: Representation-space cosine distance ──
pb = {
    'NORA-Goal\n(trained)':     [1.0,      1.0,      1.0,      1.0],
    'NORA-Spat.\n(trained)':    [0.238098, 0.274791, 0.114071, 0.023267],
    'Phi3V\nGoal':              [0.153032, 0.015982, 0.083874, 0.004254],
    'Phi3V\nSpatial':           [0.180470, 0.004608, 0.139520, 0.004437],
    'Phi3V\nObject':            [0.185045, 0.004410, 0.097740, 0.005246],
    'Phi3V\nLong':              [0.173376, 0.022108, 0.120393, 0.007031],
    'Qwen2.5VL\nGoal':         [0.081358, 0.014346, 0.033180, 0.004082],
    'Qwen2.5VL\nSpatial':      [0.080699, 0.001990, 0.031287, 0.004723],
}

nc = len(conds)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                gridspec_kw={'height_ratios': [1.5, 1]})
fig.subplots_adjust(hspace=0.45)

# ═══════════════════════ Panel A ═══════════════════════
ga = list(pa.keys())
na = len(ga)
bw = 0.18
xa = np.arange(na)

for ci, c in enumerate(conds):
    vals = [pa[g][ci] for g in ga]
    off = (ci - (nc - 1) / 2) * bw
    ax1.bar(xa + off, vals, bw, color=colors[c], edgecolor='white', linewidth=0.5)

ax1.axvline(x=3.5, color='#aaaaaa', linestyle='--', linewidth=0.8)

ymax_a = 1.72
ax1.set_ylim(0, ymax_a)
ax1.text(1.5, ymax_a * 0.97, 'OpenVLA  (VLA→VLA)', ha='center', fontsize=9,
         fontstyle='italic', color='#666666')
ax1.text(5.5, ymax_a * 0.97, 'TracVLA  (VLM→VLA)', ha='center', fontsize=9,
         fontstyle='italic', color='#666666')

for gi, g in enumerate(ga):
    vals = pa[g]
    mi = int(np.argmax(vals))
    if mi == 0:
        off = (0 - (nc - 1) / 2) * bw
        ax1.text(xa[gi] + off, vals[0] + 0.025, '★', ha='center', va='bottom',
                 fontsize=7, color=colors['empty'])

ax1.set_xticks(xa)
ax1.set_xticklabels(ga, fontsize=8)
ax1.set_ylabel("Cohen's $d$")
ax1.set_title('(A)  Action-space counterfactual sensitivity',
              fontsize=11, fontweight='bold', loc='left', pad=6)

handles = [mpatches.Patch(facecolor=colors[c], label=c) for c in conds]
ax1.legend(handles=handles, ncol=4, fontsize=8.5, loc='upper right',
           framealpha=0.85, edgecolor='#cccccc')

# ═══════════════════════ Panel B ═══════════════════════
gb = list(pb.keys())
nb = len(gb)
xb = np.arange(nb)
bw_b = 0.18
ylim_b = 0.33

for ci, c in enumerate(conds):
    vals = [pb[g][ci] for g in gb]
    off = (ci - (nc - 1) / 2) * bw_b
    disp = [min(v, ylim_b * 0.95) for v in vals]
    ax2.bar(xb + off, disp, bw_b, color=colors[c], edgecolor='white', linewidth=0.5)

# break marks for NORA-Goal (all = 1.0)
bar_top = ylim_b * 0.95
for ci in range(nc):
    off = (ci - (nc - 1) / 2) * bw_b
    cx = xb[0] + off
    hw = bw_b * 0.35
    dy = ylim_b * 0.012
    yb = bar_top - 3 * dy
    ax2.plot([cx - hw, cx + hw], [yb, yb + 2*dy], color='white', linewidth=1.8, solid_capstyle='butt')
    ax2.plot([cx - hw, cx + hw], [yb - 1.5*dy, yb + 0.5*dy], color='white', linewidth=1.8, solid_capstyle='butt')

ax2.annotate('all = 1.0', xy=(xb[0], bar_top + ylim_b*0.02), fontsize=7,
             fontweight='bold', ha='center', va='bottom', color='#444444')

# vertical separators
ax2.axvline(x=1.5, color='#aaaaaa', linestyle='--', linewidth=0.8)
ax2.axvline(x=5.5, color='#aaaaaa', linestyle='--', linewidth=0.8)

# region labels
ax2.text(0.5, ylim_b * 0.97, 'NORA (trained)', ha='center', fontsize=8,
         fontstyle='italic', color='#666666')
ax2.text(3.5, ylim_b * 0.97, 'Phi-3-Vision (untrained ref)', ha='center', fontsize=8,
         fontstyle='italic', color='#666666')
ax2.text(6.5, ylim_b * 0.97, 'Qwen2.5-VL (untrained ref)', ha='center', fontsize=8,
         fontstyle='italic', color='#666666')

ax2.set_ylim(0, ylim_b)
ax2.set_xticks(xb)
ax2.set_xticklabels(gb, fontsize=7.5)
ax2.set_ylabel('Cosine distance')
ax2.set_title('(B)  Representation-space counterfactual sensitivity',
              fontsize=11, fontweight='bold', loc='left', pad=6)
ax2.annotate('Note: not directly comparable to Panel A',
             xy=(0.99, 0.04), xycoords='axes fraction',
             fontsize=7, ha='right', fontstyle='italic', color='#999999')

out = './paper_figures/fig_counterfactual_bar'
plt.savefig(out + '.pdf')
plt.savefig(out + '.png')
print(f'Saved: {out}.pdf and {out}.png')
