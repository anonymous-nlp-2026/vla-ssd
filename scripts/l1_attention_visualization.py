"""l1_attention_visualization.py — Extract and visualize Layer 1 attention patterns.

Extracts attention weights at Layer 1 for the last_preaction token position,
showing how it attends to instruction tokens vs image tokens.

Token layout (Prismatic/OpenVLA): [BOS] [256 image tokens] [text tokens...]
"""

import argparse
import gc
import glob
import json
import os
import re
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image

NUM_IMAGE_TOKENS = 256
TARGET_LAYER = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--output_dir", default="./results/attention/")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--condition", choices=["trained", "untrained", "llama2base"], default="trained")
    return p.parse_args()


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def load_model(model_path, condition, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    if condition == "untrained":
        print("Reinitializing LLM backbone...")
        from transformers import AutoModelForCausalLM
        llm_cfg = model.language_model.config
        random_llm = AutoModelForCausalLM.from_config(llm_cfg, torch_dtype=torch.bfloat16)
        model.language_model.load_state_dict(random_llm.state_dict())
        del random_llm
        gc.collect()

    elif condition == "llama2base":
        print("Replacing LLM backbone with Llama-2-7B base weights...")
        from transformers import AutoModelForCausalLM
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        llama2_path = "<HF_CACHE>/models--NousResearch--Llama-2-7b-hf/snapshots/"
        snap_dirs = [d for d in os.listdir(llama2_path) if not d.startswith(".")]
        assert len(snap_dirs) >= 1, f"No Llama-2 snapshot found in {llama2_path}"
        llama2_full_path = os.path.join(llama2_path, snap_dirs[0])
        base_llm = AutoModelForCausalLM.from_pretrained(
            llama2_full_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        base_sd = base_llm.state_dict()
        for name, param in model.language_model.named_parameters():
            if name in base_sd:
                if param.shape == base_sd[name].shape:
                    param.data.copy_(base_sd[name].to(param.dtype))
                elif param.shape[0] > base_sd[name].shape[0]:
                    param.data[:base_sd[name].shape[0]].copy_(base_sd[name].to(param.dtype))
        del base_llm, base_sd
        gc.collect()
        print("LLM backbone replaced with Llama-2-7B base.")

    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model, processor


@torch.no_grad()
def extract_attention(model, processor, image, instruction, device):
    """Forward pass returning Layer 1 attention weights (n_heads, seq_len, seq_len)."""
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    attn_storage = {}

    def hook_fn(module, input, output):
        # LlamaAttention.forward returns (attn_output, attn_weights, past_kv) when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_storage["weights"] = output[1].detach().cpu().float()

    # Enable output_attentions
    model.language_model.config.output_attentions = True

    hook = model.language_model.model.layers[TARGET_LAYER].self_attn.register_forward_hook(hook_fn)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=False, output_attentions=True)

    hook.remove()
    model.language_model.config.output_attentions = False

    if "weights" not in attn_storage:
        # Fallback: try outputs.attentions
        if hasattr(out, "attentions") and out.attentions is not None:
            attn_storage["weights"] = out.attentions[TARGET_LAYER].detach().cpu().float()
        else:
            raise RuntimeError("Could not capture attention weights. Check model architecture.")

    attn = attn_storage["weights"].squeeze(0)  # (n_heads, seq_len, seq_len)
    seq_len = attn.shape[-1]

    # Token regions
    # BOS: position 0
    # Image: positions 1 to 256 (NUM_IMAGE_TOKENS)
    # Text/instruction: positions 257 to end
    input_ids = inputs.get("input_ids", None)
    if input_ids is not None:
        total_len = input_ids.shape[1]
    else:
        total_len = seq_len

    token_info = {
        "bos_range": (0, 1),
        "image_range": (1, 1 + NUM_IMAGE_TOKENS),
        "text_range": (1 + NUM_IMAGE_TOKENS, seq_len),
        "seq_len": seq_len,
        "last_preaction_idx": seq_len - 1,
    }

    del out, inputs
    return attn.numpy(), token_info


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model(args.model_path, args.condition, device)

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} tasks")

    # Select 5 representative tasks (indices 0, 2, 4, 6, 8)
    task_indices = [0, 2, 4, 6, 8]
    demo_idx = 0
    frame_idx = 25

    results = []

    for task_idx in task_indices:
        if task_idx >= len(hdf5_files):
            print(f"Task index {task_idx} out of range, skipping")
            continue

        hdf5_path = hdf5_files[task_idx]
        instruction = task_desc_from_filename(hdf5_path)
        print(f"\n[Task {task_idx}] {instruction}")

        f = h5py.File(hdf5_path, "r")
        demo_key = f"demo_{demo_idx}"
        images_ds = f["data"][demo_key]["obs"]["agentview_rgb"]
        T = images_ds.shape[0]
        actual_frame = min(frame_idx, T - 1)
        img = Image.fromarray(images_ds[actual_frame])
        f.close()

        print(f"  Frame {actual_frame}/{T}, extracting attention...")
        attn, token_info = extract_attention(model, processor, img, instruction, device)
        n_heads = attn.shape[0]
        seq_len = token_info["seq_len"]
        lp_idx = token_info["last_preaction_idx"]

        # Extract last_preaction row
        lp_attn = attn[:, lp_idx, :]  # (n_heads, seq_len)

        # Compute attention to each region
        bos_start, bos_end = token_info["bos_range"]
        img_start, img_end = token_info["image_range"]
        txt_start, txt_end = token_info["text_range"]

        attn_to_bos = lp_attn[:, bos_start:bos_end].sum(axis=1)
        attn_to_image = lp_attn[:, img_start:img_end].sum(axis=1)
        attn_to_text = lp_attn[:, txt_start:txt_end].sum(axis=1)

        print(f"  seq_len={seq_len}, n_heads={n_heads}")
        print(f"  Mean attn to BOS: {attn_to_bos.mean():.4f}")
        print(f"  Mean attn to image: {attn_to_image.mean():.4f}")
        print(f"  Mean attn to text: {attn_to_text.mean():.4f}")

        # Identify instruction-attending heads (>50% to text)
        instruction_heads = [int(h) for h in range(n_heads) if attn_to_text[h] > 0.5]
        print(f"  Instruction-attending heads (>50% to text): {instruction_heads}")

        results.append({
            "task_idx": task_idx,
            "instruction": instruction,
            "frame": actual_frame,
            "demo_length": T,
            "seq_len": seq_len,
            "n_heads": n_heads,
            "token_regions": {
                "bos": [bos_start, bos_end],
                "image": [img_start, img_end],
                "text": [txt_start, txt_end],
            },
            "per_head_attn_to_bos": attn_to_bos.tolist(),
            "per_head_attn_to_image": attn_to_image.tolist(),
            "per_head_attn_to_text": attn_to_text.tolist(),
            "mean_attn_to_bos": float(attn_to_bos.mean()),
            "mean_attn_to_image": float(attn_to_image.mean()),
            "mean_attn_to_text": float(attn_to_text.mean()),
            "instruction_attending_heads": instruction_heads,
            "lp_attn_full": lp_attn.tolist(),
        })

        torch.cuda.empty_cache()

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, f"l1_attention_{args.condition}_summary.json")
    # Save without the full attention matrix (too large for JSON)
    summary_for_json = []
    full_attn_data = []
    for r in results:
        full_attn_data.append(np.array(r["lp_attn_full"], dtype=np.float32))
        r_copy = {k: v for k, v in r.items() if k != "lp_attn_full"}
        summary_for_json.append(r_copy)

    with open(summary_path, "w") as f:
        json.dump(summary_for_json, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(args.output_dir, f"l1_attention_{args.condition}.pdf")

    with PdfPages(pdf_path) as pdf:
        for i, (r, lp_attn_arr) in enumerate(zip(results, full_attn_data)):
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

            # Heatmap
            ax = axes[0]
            n_heads = lp_attn_arr.shape[0]
            seq_len = lp_attn_arr.shape[1]
            im = ax.imshow(lp_attn_arr, aspect="auto", cmap="viridis",
                          vmin=0, vmax=lp_attn_arr.max())
            ax.set_xlabel("Source token position")
            ax.set_ylabel("Attention head")
            ax.set_title(f"L1 Attention from last_preaction — Task {r['task_idx']}\n\"{r['instruction']}\"",
                        fontsize=11)
            ax.set_yticks(range(0, n_heads, 4))

            # Add region boundaries
            img_end = r["token_regions"]["image"][1]
            txt_start = r["token_regions"]["text"][0]
            ax.axvline(x=0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axvline(x=img_end - 0.5, color="cyan", linewidth=1.2, linestyle="--", alpha=0.9)

            # Region labels at top
            ax.text(img_end / 2, -1.5, "IMAGE (256)", ha="center", fontsize=8, color="cyan")
            ax.text((txt_start + seq_len) / 2, -1.5, "TEXT", ha="center", fontsize=8, color="yellow")

            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

            # Bar chart: per-head region summary
            ax2 = axes[1]
            heads = np.arange(n_heads)
            width = 0.25
            ax2.bar(heads - width, r["per_head_attn_to_bos"], width, label="BOS", color="gray", alpha=0.7)
            ax2.bar(heads, r["per_head_attn_to_image"], width, label="Image", color="cyan", alpha=0.7)
            ax2.bar(heads + width, r["per_head_attn_to_text"], width, label="Text/Instruction", color="orange", alpha=0.7)
            ax2.set_xlabel("Attention head")
            ax2.set_ylabel("Total attention weight")
            ax2.set_title("Per-head attention allocation by region")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.set_xticks(range(0, n_heads, 2))
            ax2.set_xlim(-0.5, n_heads - 0.5)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Summary page
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        tasks = [r["instruction"][:30] for r in results]
        mean_img = [r["mean_attn_to_image"] for r in results]
        mean_txt = [r["mean_attn_to_text"] for r in results]
        mean_bos = [r["mean_attn_to_bos"] for r in results]

        x = np.arange(len(tasks))
        width = 0.25
        ax.bar(x - width, mean_bos, width, label="BOS", color="gray")
        ax.bar(x, mean_img, width, label="Image", color="cyan")
        ax.bar(x + width, mean_txt, width, label="Text/Instruction", color="orange")
        ax.set_xlabel("Task")
        ax.set_ylabel("Mean attention (across heads)")
        ax.set_title(f"L1 last_preaction attention allocation ({args.condition} model)")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved: {pdf_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY: L1 Attention from last_preaction token")
    print("=" * 60)
    for r in results:
        print(f"\nTask {r['task_idx']}: {r['instruction']}")
        print(f"  Attn to BOS:   {r['mean_attn_to_bos']:.4f}")
        print(f"  Attn to Image: {r['mean_attn_to_image']:.4f}")
        print(f"  Attn to Text:  {r['mean_attn_to_text']:.4f}")
        print(f"  Instruction-attending heads: {r['instruction_attending_heads']}")

    overall_img = np.mean([r["mean_attn_to_image"] for r in results])
    overall_txt = np.mean([r["mean_attn_to_text"] for r in results])
    overall_bos = np.mean([r["mean_attn_to_bos"] for r in results])
    print(f"\nOverall means: BOS={overall_bos:.4f}, Image={overall_img:.4f}, Text={overall_txt:.4f}")


if __name__ == "__main__":
    main()
