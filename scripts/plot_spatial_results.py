import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ── Paths ──
RSA_SPATIAL = './results/rsa/rsa_libero_spatial.json'
RSA_GOAL   = './results/rsa/unified_rsa_results.json'
R2_SPATIAL  = './results/functional_validation/action_probe_libero_spatial.json'
R2_GOAL     = './results/functional_validation/fulllayer_action_probe.json'
OUT_DIR     = './results/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──
with open(RSA_SPATIAL) as f: rsa_sp = json.load(f)
with open(RSA_GOAL) as f:   rsa_gl = json.load(f)
with open(R2_SPATIAL) as f:  r2_sp  = json.load(f)
with open(R2_GOAL) as f:    r2_gl  = json.load(f)

N_LAYERS = 33
layers = np.arange(N_LAYERS)

def extract_rsa(data, condition, token_type):
    cond = data['conditions'][condition][token_type]
    return np.array([cond[f'layer_{i}']['mean_rsa'] for i in range(N_LAYERS)])

def extract_rsa_ci(data, condition, token_type):
    cond = data['conditions'][condition][token_type]
    lo = np.array([cond[f'layer_{i}']['ci95'][0] for i in range(N_LAYERS)])
    hi = np.array([cond[f'layer_{i}']['ci95'][1] for i in range(N_LAYERS)])
    return lo, hi

def extract_r2(data, condition, token_type):
    cond = data[condition][token_type]
    return np.array([cond[f'layer_{i}']['mean_R2'] for i in range(N_LAYERS)])

# ── Extract all curves ──
# Spatial RSA
sp_rsa_t_im  = extract_rsa(rsa_sp, 'trained',   'image_mean')
sp_rsa_t_lp  = extract_rsa(rsa_sp, 'trained',   'last_preaction')
sp_rsa_u_im  = extract_rsa(rsa_sp, 'untrained', 'image_mean')
sp_rsa_u_lp  = extract_rsa(rsa_sp, 'untrained', 'last_preaction')
sp_rsa_t_im_ci = extract_rsa_ci(rsa_sp, 'trained',   'image_mean')
sp_rsa_t_lp_ci = extract_rsa_ci(rsa_sp, 'trained',   'last_preaction')
sp_rsa_u_im_ci = extract_rsa_ci(rsa_sp, 'untrained', 'image_mean')
sp_rsa_u_lp_ci = extract_rsa_ci(rsa_sp, 'untrained', 'last_preaction')

# Goal RSA
gl_rsa_t_im  = extract_rsa(rsa_gl, 'trained_with_inst',   'image_mean')
gl_rsa_t_lp  = extract_rsa(rsa_gl, 'trained_with_inst',   'last_preaction')
gl_rsa_u_im  = extract_rsa(rsa_gl, 'untrained_with_inst', 'image_mean')
gl_rsa_u_lp  = extract_rsa(rsa_gl, 'untrained_with_inst', 'last_preaction')
gl_rsa_t_im_ci = extract_rsa_ci(rsa_gl, 'trained_with_inst',   'image_mean')
gl_rsa_t_lp_ci = extract_rsa_ci(rsa_gl, 'trained_with_inst',   'last_preaction')
gl_rsa_u_im_ci = extract_rsa_ci(rsa_gl, 'untrained_with_inst', 'image_mean')
gl_rsa_u_lp_ci = extract_rsa_ci(rsa_gl, 'untrained_with_inst', 'last_preaction')

# Spatial R2
sp_r2_t_im = extract_r2(r2_sp, 'trained',   'image_mean')
sp_r2_t_lp = extract_r2(r2_sp, 'trained',   'last_preaction')
sp_r2_u_im = extract_r2(r2_sp, 'untrained', 'image_mean')
sp_r2_u_lp = extract_r2(r2_sp, 'untrained', 'last_preaction')

# Goal R2
gl_r2_t_im = extract_r2(r2_gl, 'trained',   'image_mean')
gl_r2_t_lp = extract_r2(r2_gl, 'trained',   'last_preaction')
gl_r2_u_im = extract_r2(r2_gl, 'untrained', 'image_mean')
gl_r2_u_lp = extract_r2(r2_gl, 'untrained', 'last_preaction')

# ── Style constants ──
C_TRAIN_IM  = '#1f77b4'  # blue
C_TRAIN_LP  = '#0b3d91'  # dark blue
C_UTRAIN_IM = '#d62728'  # red
C_UTRAIN_LP = '#e67300'  # orange
ALPHA_CI = 0.15
LW = 1.8
LW_THIN = 1.2

def style_ax(ax, xlabel, ylabel, title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    if title:
        ax.set_title(title, fontsize=13, pad=8)

def annotate_peak(ax, x, y, color, offset_y=0.01):
    peak_idx = np.argmax(y)
    ax.annotate(f'L{peak_idx}', xy=(peak_idx, y[peak_idx]),
                xytext=(peak_idx + 1.5, y[peak_idx] + offset_y),
                fontsize=8, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

def plot_4curves_rsa(ax, t_im, t_lp, u_im, u_lp,
                     t_im_ci, t_lp_ci, u_im_ci, u_lp_ci, title=None):
    ax.plot(layers, t_im, '-',  color=C_TRAIN_IM,  lw=LW, label='Trained, image_mean')
    ax.plot(layers, t_lp, '--', color=C_TRAIN_LP,  lw=LW, label='Trained, last_preaction')
    ax.plot(layers, u_im, '-',  color=C_UTRAIN_IM, lw=LW, label='Untrained, image_mean')
    ax.plot(layers, u_lp, '--', color=C_UTRAIN_LP, lw=LW, label='Untrained, last_preaction')
    ax.fill_between(layers, t_im_ci[0], t_im_ci[1], color=C_TRAIN_IM,  alpha=ALPHA_CI)
    ax.fill_between(layers, t_lp_ci[0], t_lp_ci[1], color=C_TRAIN_LP,  alpha=ALPHA_CI)
    ax.fill_between(layers, u_im_ci[0], u_im_ci[1], color=C_UTRAIN_IM, alpha=ALPHA_CI)
    ax.fill_between(layers, u_lp_ci[0], u_lp_ci[1], color=C_UTRAIN_LP, alpha=ALPHA_CI)
    annotate_peak(ax, layers, t_im, C_TRAIN_IM)
    annotate_peak(ax, layers, t_lp, C_TRAIN_LP)
    annotate_peak(ax, layers, u_im, C_UTRAIN_IM, offset_y=-0.015)
    annotate_peak(ax, layers, u_lp, C_UTRAIN_LP, offset_y=-0.015)
    style_ax(ax, 'Layer', r'RSA (Spearman $\rho$)', title)

def plot_4curves_r2(ax, t_im, t_lp, u_im, u_lp, title=None):
    ax.plot(layers, t_im, '-',  color=C_TRAIN_IM,  lw=LW, label='Trained, image_mean')
    ax.plot(layers, t_lp, '--', color=C_TRAIN_LP,  lw=LW, label='Trained, last_preaction')
    ax.plot(layers, u_im, '-',  color=C_UTRAIN_IM, lw=LW, label='Untrained, image_mean')
    ax.plot(layers, u_lp, '--', color=C_UTRAIN_LP, lw=LW, label='Untrained, last_preaction')
    annotate_peak(ax, layers, t_im, C_TRAIN_IM)
    annotate_peak(ax, layers, t_lp, C_TRAIN_LP)
    annotate_peak(ax, layers, u_im, C_UTRAIN_IM, offset_y=-0.01)
    annotate_peak(ax, layers, u_lp, C_UTRAIN_LP, offset_y=-0.01)
    style_ax(ax, 'Layer', r'Mean $R^2$', title)

# ════════════════════════════════════════════
# Figure A: LIBERO-Spatial Per-Layer RSA
# ════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
plot_4curves_rsa(ax, sp_rsa_t_im, sp_rsa_t_lp, sp_rsa_u_im, sp_rsa_u_lp,
                 sp_rsa_t_im_ci, sp_rsa_t_lp_ci, sp_rsa_u_im_ci, sp_rsa_u_lp_ci,
                 'LIBERO-Spatial: Per-Layer RSA (Method B)')
ax.legend(fontsize=9, loc='best', framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'libero_spatial_rsa_perlayer.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'libero_spatial_rsa_perlayer.png'), bbox_inches='tight')
plt.close(fig)
print('Figure A saved.')

# ════════════════════════════════════════════
# Figure B: LIBERO-Spatial Per-Layer R²
# ════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
plot_4curves_r2(ax, sp_r2_t_im, sp_r2_t_lp, sp_r2_u_im, sp_r2_u_lp,
                'LIBERO-Spatial: Per-Layer Action Probe $R^2$')
ax.legend(fontsize=9, loc='best', framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'libero_spatial_r2_perlayer.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'libero_spatial_r2_perlayer.png'), bbox_inches='tight')
plt.close(fig)
print('Figure B saved.')

# ════════════════════════════════════════════
# Figure C: Cross-Benchmark Comparison (1×2)
# ════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=300)

# Left panel: RSA comparison
# Goal = solid thin, Spatial = solid thick (both same colors for trained/untrained)
# Distinguish benchmark by linewidth + marker
for arr, ci, color, ls, label in [
    (gl_rsa_t_im, gl_rsa_t_im_ci, C_TRAIN_IM, '-',  'Goal, Trained, img_mean'),
    (gl_rsa_t_lp, gl_rsa_t_lp_ci, C_TRAIN_LP, '--', 'Goal, Trained, last_pre'),
    (gl_rsa_u_im, gl_rsa_u_im_ci, C_UTRAIN_IM, '-',  'Goal, Untrained, img_mean'),
    (gl_rsa_u_lp, gl_rsa_u_lp_ci, C_UTRAIN_LP, '--', 'Goal, Untrained, last_pre'),
]:
    ax1.plot(layers, arr, ls, color=color, lw=LW_THIN, alpha=0.5, label=label)
    ax1.fill_between(layers, ci[0], ci[1], color=color, alpha=0.06)

for arr, ci, color, ls, label in [
    (sp_rsa_t_im, sp_rsa_t_im_ci, C_TRAIN_IM, '-',  'Spatial, Trained, img_mean'),
    (sp_rsa_t_lp, sp_rsa_t_lp_ci, C_TRAIN_LP, '--', 'Spatial, Trained, last_pre'),
    (sp_rsa_u_im, sp_rsa_u_im_ci, C_UTRAIN_IM, '-',  'Spatial, Untrained, img_mean'),
    (sp_rsa_u_lp, sp_rsa_u_lp_ci, C_UTRAIN_LP, '--', 'Spatial, Untrained, last_pre'),
]:
    ax1.plot(layers, arr, ls, color=color, lw=LW, label=label)
    ax1.fill_between(layers, ci[0], ci[1], color=color, alpha=0.08)

style_ax(ax1, 'Layer', r'RSA (Spearman $\rho$)', 'Per-Layer RSA: Goal (thin) vs Spatial (thick)')
# Custom legend: 2 columns
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=7.5, loc='lower left', ncol=2, framealpha=0.9)

# Right panel: R² comparison
for arr, color, ls, label in [
    (gl_r2_t_im, C_TRAIN_IM, '-',  'Goal, Trained, img_mean'),
    (gl_r2_t_lp, C_TRAIN_LP, '--', 'Goal, Trained, last_pre'),
    (gl_r2_u_im, C_UTRAIN_IM, '-',  'Goal, Untrained, img_mean'),
    (gl_r2_u_lp, C_UTRAIN_LP, '--', 'Goal, Untrained, last_pre'),
]:
    ax2.plot(layers, arr, ls, color=color, lw=LW_THIN, alpha=0.5, label=label)

for arr, color, ls, label in [
    (sp_r2_t_im, C_TRAIN_IM, '-',  'Spatial, Trained, img_mean'),
    (sp_r2_t_lp, C_TRAIN_LP, '--', 'Spatial, Trained, last_pre'),
    (sp_r2_u_im, C_UTRAIN_IM, '-',  'Spatial, Untrained, img_mean'),
    (sp_r2_u_lp, C_UTRAIN_LP, '--', 'Spatial, Untrained, last_pre'),
]:
    ax2.plot(layers, arr, ls, color=color, lw=LW, label=label)

style_ax(ax2, 'Layer', r'Mean $R^2$', 'Per-Layer $R^2$: Goal (thin) vs Spatial (thick)')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=7.5, loc='lower left', ncol=2, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'cross_benchmark_comparison.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'cross_benchmark_comparison.png'), bbox_inches='tight')
plt.close(fig)
print('Figure C saved.')

print('\nAll figures saved to:', OUT_DIR)
