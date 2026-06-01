"""init_untrained.py — Reinitialize OpenVLA LLM backbone with random weights.

Keeps vision encoder (DINOv2 + SigLIP) and projector intact.
Uses AutoModelForCausalLM.from_config() to match LLaMA's default initialization.

Input:  Pretrained OpenVLA checkpoint
Output: New checkpoint with randomized LLM backbone

Usage:
  python init_untrained.py --model_path checkpoints/openvla-7b \
                           --output_path checkpoints/openvla-7b-untrained/
"""

import argparse
import gc
import os

import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def parse_args():
    p = argparse.ArgumentParser(description="Reinitialize OpenVLA LLM backbone")
    p.add_argument("--model_path", required=True,
                   help="Path to pretrained OpenVLA checkpoint")
    p.add_argument("--output_path", required=True,
                   help="Path to save reinitialized model")
    return p.parse_args()


def reinit_llm_backbone(model):
    """Reinitialize language_model.* using from_config() (matches LLaMA default init)."""
    from transformers import AutoModelForCausalLM

    llm_config = model.language_model.config
    fresh_llm = AutoModelForCausalLM.from_config(
        llm_config, torch_dtype=torch.bfloat16
    )
    fresh_sd = fresh_llm.state_dict()

    n_reinit = 0
    n_kept = 0
    reinit_names = []

    for name, param in model.named_parameters():
        if not name.startswith("language_model."):
            n_kept += 1
            continue

        llm_key = name[len("language_model."):]
        if llm_key in fresh_sd:
            with torch.no_grad():
                param.copy_(fresh_sd[llm_key])
            n_reinit += 1
            reinit_names.append(name)
        else:
            n_kept += 1

    del fresh_llm, fresh_sd
    gc.collect()
    return n_reinit, n_kept, reinit_names


def main():
    args = parse_args()

    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    total = sum(p.numel() for p in model.parameters())
    llm = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith("language_model."))
    vision = total - llm
    print(f"Total: {total/1e9:.2f}B | LLM: {llm/1e9:.2f}B | Vision+Proj: {vision/1e9:.2f}B")

    print("Reinitializing LLM backbone ...")
    n_reinit, n_kept, reinit_names = reinit_llm_backbone(model)
    print(f"Reinitialized: {n_reinit} params | Kept: {n_kept} params")

    module_counts = {}
    for name in reinit_names:
        mod = ".".join(name.split(".")[:3])
        module_counts[mod] = module_counts.get(mod, 0) + 1
    print("Reinitialized modules:")
    for mod, cnt in sorted(module_counts.items()):
        print(f"  {mod}: {cnt} params")

    print(f"\nSaving to {args.output_path} ...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)

    saved_size = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(args.output_path) for fn in fns
    ) / 1024**3
    print(f"Saved ({saved_size:.1f} GB)")

    for name, param in model.named_parameters():
        if name.startswith("language_model.") and param.dim() >= 2:
            print(f"  LLM sample std: {param.float().std().item():.6f} ({name})")
            break
    for name, param in model.named_parameters():
        if not name.startswith("language_model.") and param.dim() >= 2:
            print(f"  Vision sample std: {param.float().std().item():.6f} ({name})")
            break

    del model
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
