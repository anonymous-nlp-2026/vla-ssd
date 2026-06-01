"""Generate publication-quality L1 attention three-condition comparison figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("./results/attention/")
DATA_DIR = OUTPUT_DIR

# Colors
C_TEXT = '#4C72B0'
C_IMAGE = '#DD8452'
C_BOS = '#999999'

def load_trained():
    with open(DATA_DIR / "l1_attention_trained_summary.json") as f:
        data = json.load(f)
    # Average across 3 tasks
    text = np.mean([d['per_head_attn_to_text'] for d in data], axis=0)
    image = np.mean([d['per_head_attn_to_image'] for d in data], axis=0)
    bos = np.mean([d['per_head_attn_to_bos'] for d in data], axis=0)
    return text, image, bos

def load_untrained():
    with open(DATA_DIR / "l1_attention_untrained_summary.json") as f:
        data = json.load(f)
    text = np.array([data['per_head'][f'head_{i}']['mean_instr_attn'] for i in range(32)])
    image = np.array([data['per_head'][f'head_{i}']['mean_image_attn'] for i in range(32)])
    bos = 1.0 - text - image  # residual (near zero)
    bos = np.clip(bos, 0, 1)
    return text, image, bos

def load_llama2base():
    with open(DATA_DIR / "l1_attention_llama2base_summary.json") as f:
        data = json.load(f)
    text = np.mean([d['per_head_attn_to_text'] for d in data], axis=0)
    image = np.mean([d['per_head_attn_to_image'] for d in data], axis=0)
    bos = np.mean([d['per_head_attn_to_bos'] for d in data], axis=0)
    return text, image, bos

def count_dominant(text_attn, threshold=0.5):
    return int(np.sum(text_attn > threshold))

def plot_panel(ax, text, image, bos, subtitle, n_dominant):
    # Sort by text attention descending
    order = np.argsort(-text)
    text_s = text[order]
    image_s = image[order]
    bos_s = bos[order]
    
    x = np.arange(32)
    
    # Stacked bars: bottom = BOS, middle = image, top = text
    ax.bar(x, bos_s, color=C_BOS, width=0.85, label='BOS')
    ax.bar(x, image_s, bottom=bos_s, color=C_IMAGE, width=0.85, label='Image')
    ax.bar(x, text_s, bottom=bos_s + image_s, color=C_TEXT, width=0.85, label='Instruction')
    
    # 50% threshold
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # Annotation
    ax.text(0.98, 0.95, f'{n_dominant}/32 instruction-dominant',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 31.6)
    ax.set_xlabel('Head index (sorted)', fontsize=9)
    ax.set_title(subtitle, fontsize=10, pad=4)
    ax.set_xticks([0, 7, 15, 23, 31])
    ax.tick_params(labelsize=8)

def main():
    text_u, image_u, bos_u = load_untrained()
    text_l, image_l, bos_l = load_llama2base()
    text_t, image_t, bos_t = load_trained()
    
    n_u = count_dominant(text_u)
    n_l = count_dominant(text_l)
    n_t = count_dominant(text_t)
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    
    plot_panel(axes[0], text_u, image_u, bos_u, '(a) Untrained VLA', n_u)
    plot_panel(axes[1], text_l, image_l, bos_l, '(b) Llama-2-7B Base', n_l)
    plot_panel(axes[2], text_t, image_t, bos_t, '(c) VLA Trained', n_t)
    
    axes[0].set_ylabel('Attention Fraction', fontsize=9)
    
    # Legend on last panel
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save
    pdf_path = OUTPUT_DIR / "fig_l1_attention_three_conditions.pdf"
    png_path = OUTPUT_DIR / "fig_l1_attention_three_conditions.png"
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Instruction-dominant heads: Untrained={n_u}, Llama2Base={n_l}, Trained={n_t}")

if __name__ == "__main__":
    main()
