"""Random-subset ablation: verify dominant-24 head ablation effect is specific.

Strategy:
  Phase 1 (GPU): Single forward pass to cache attention intermediates at layer 1.
  Phase 2 (GPU, float16): Reconstruct L1 features for each ablation pattern using
    mixed-precision (autocast) to match the original experiment's numerical behavior.
  Phase 3 (CPU): Train probes, compute statistics, generate plot.
"""

import argparse
import gc
import glob
import json
import os
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008

INSTRUCTION_HEADS_24 = [0,1,2,4,7,8,10,11,12,14,15,16,17,18,19,20,22,23,25,26,27,29,30,31]

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
HIDDEN_DIM = 256
N_RANDOM = 20
N_ABLATE = 24


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/openvla-7b")
    p.add_argument("--data_dir", default="./data/libero/libero_goal/")
    p.add_argument("--output", default="./results/random_subset_ablation.json")
    p.add_argument("--fig_dir", default="./results/figures/")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--demos_per_task", type=int, default=50)
    p.add_argument("--n_random", type=int, default=N_RANDOM)
    return p.parse_args()


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


# =============================================================================
# Phase 1: GPU caching
# =============================================================================

class AttentionCacher:
    def __init__(self):
        self.full_context = None
        self.non_inst_sum = None
        self.non_inst_weight = None
        self.residual_last = None

    def reset(self):
        self.full_context = None
        self.non_inst_sum = None
        self.non_inst_weight = None
        self.residual_last = None


def install_caching_hooks(model, cacher):
    import transformers.models.llama.modeling_llama as llama_mod
    from transformers.models.llama.modeling_llama import repeat_kv

    layer1 = model.language_model.model.layers[1]
    layer1_attn = layer1.self_attn

    original_fn = llama_mod.eager_attention_forward

    def caching_eager_attention_forward(module, query, key, value, attention_mask,
                                        scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        if module is layer1_attn:
            w = attn_weights[0]
            v = value_states[0]
            last_pos = w.shape[1] - 1
            w_last = w[:, last_pos, :]

            full_ctx = torch.bmm(w_last.unsqueeze(1), v).squeeze(1)

            seq_len = w_last.shape[1]
            if INST_START < seq_len:
                w_non_inst = w_last[:, :INST_START]
                v_non_inst = v[:, :INST_START, :]
                non_inst_s = torch.bmm(w_non_inst.unsqueeze(1), v_non_inst).squeeze(1)
                non_inst_w = w_non_inst.sum(dim=1)
            else:
                non_inst_s = full_ctx.clone()
                non_inst_w = torch.ones(NUM_HEADS, device=w.device, dtype=w.dtype)

            # Keep in float16 on CPU to preserve original precision
            cacher.full_context = full_ctx.cpu()
            cacher.non_inst_sum = non_inst_s.cpu()
            cacher.non_inst_weight = non_inst_w.cpu()

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    llama_mod.eager_attention_forward = caching_eager_attention_forward

    def capture_residual(module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        # Keep in float16 to preserve original precision
        cacher.residual_last = hidden_states[0, -1, :].detach().cpu()

    pre_hook = layer1.register_forward_pre_hook(capture_residual, with_kwargs=True)

    class EarlyExit(Exception):
        pass

    layer2 = model.language_model.model.layers[2]

    def early_exit_hook(module, args, kwargs):
        raise EarlyExit()

    exit_hook = layer2.register_forward_pre_hook(early_exit_hook, with_kwargs=True)

    return original_fn, [pre_hook, exit_hook], EarlyExit


def remove_caching_hooks(model, original_fn, hooks):
    import transformers.models.llama.modeling_llama as llama_mod
    llama_mod.eager_attention_forward = original_fn
    for h in hooks:
        h.remove()


def load_model(model_path, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model, processor


def cache_all_frames(model, processor, data_dir, demos_per_task, device):
    cacher = AttentionCacher()
    original_fn, hooks, EarlyExit = install_caching_hooks(model, cacher)

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    task_names = [task_desc_from_filename(f) for f in hdf5_files]

    all_full_context = []
    all_non_inst_sum = []
    all_non_inst_weight = []
    all_residual_last = []
    all_actions = []
    task_boundaries = []

    total_frames = 0
    t0 = time.time()

    for task_idx, (hdf5_path, task_name) in enumerate(zip(hdf5_files, task_names)):
        print(f"\n  Task {task_idx}: {task_name}")
        prompt = f"In: What action should the robot take to {task_name}?\nOut:"

        f = h5py.File(hdf5_path, "r")
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))[:demos_per_task]

        for demo_idx, dk in enumerate(demo_keys):
            imgs_np = data[dk]["obs"]["agentview_rgb"][:]
            acts_np = np.asarray(data[dk]["actions"], dtype=np.float32)
            n_frames = len(imgs_np)

            for frame_idx in range(n_frames):
                img = Image.fromarray(imgs_np[frame_idx])
                inputs = processor(text=[prompt], images=[img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                cacher.reset()
                try:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                        model(**inputs)
                except EarlyExit:
                    pass

                all_full_context.append(cacher.full_context)
                all_non_inst_sum.append(cacher.non_inst_sum)
                all_non_inst_weight.append(cacher.non_inst_weight)
                all_residual_last.append(cacher.residual_last)

                del inputs
                total_frames += 1

            all_actions.append(acts_np)
            task_boundaries.append((task_idx, demo_idx, n_frames))

            if (demo_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                fps = total_frames / elapsed
                print(f"    {demo_idx+1}/{len(demo_keys)} demos, {total_frames} frames, {fps:.1f} fps")

        f.close()
        gc.collect()
        torch.cuda.empty_cache()

    remove_caching_hooks(model, original_fn, hooks)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {total_frames} frames in {elapsed/60:.1f} min ({total_frames/elapsed:.1f} fps)")

    cache = {
        "full_context": torch.stack(all_full_context),
        "non_inst_sum": torch.stack(all_non_inst_sum),
        "non_inst_weight": torch.stack(all_non_inst_weight),
        "residual_last": torch.stack(all_residual_last),
        "actions": all_actions,
        "task_boundaries": task_boundaries,
        "task_names": task_names,
    }
    return cache


# =============================================================================
# Phase 2: GPU reconstruction in float16 (matching original precision)
# =============================================================================

def extract_layer1_weights(model):
    layer1 = model.language_model.model.layers[1]
    weights = {
        "o_proj": layer1.self_attn.o_proj.weight.detach().cpu(),
        "gate_proj": layer1.mlp.gate_proj.weight.detach().cpu(),
        "up_proj": layer1.mlp.up_proj.weight.detach().cpu(),
        "down_proj": layer1.mlp.down_proj.weight.detach().cpu(),
        "post_attn_norm_weight": layer1.post_attention_layernorm.weight.detach().cpu(),
        "norm_eps": layer1.post_attention_layernorm.variance_epsilon,
    }
    return weights


def rms_norm(x, weight, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def reconstruct_l1_features(cache, weights, ablate_heads, device, chunk_size=4096):
    """Reconstruct L1 features on GPU with autocast to match original precision."""
    N = cache["full_context"].shape[0]
    full_ctx = cache["full_context"]
    non_inst_s = cache["non_inst_sum"]
    non_inst_w = cache["non_inst_weight"]
    residual = cache["residual_last"]

    # Move weights to GPU
    w_o = weights["o_proj"].to(device)
    w_gate = weights["gate_proj"].to(device)
    w_up = weights["up_proj"].to(device)
    w_down = weights["down_proj"].to(device)
    norm_w = weights["post_attn_norm_weight"].to(device)
    norm_eps = weights["norm_eps"]

    all_outputs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        ctx = full_ctx[start:end].clone().to(device)
        if ablate_heads is not None and len(ablate_heads) > 0:
            heads = list(ablate_heads)
            ns = non_inst_s[start:end, heads, :].to(device)
            nw = non_inst_w[start:end, heads].unsqueeze(-1).to(device)
            nw = torch.clamp(nw, min=torch.tensor(1e-9, dtype=nw.dtype, device=device))
            ctx[:, heads, :] = (ns / nw).to(ctx.dtype)
            del ns, nw

        concat_ctx = ctx.reshape(end - start, HIDDEN_SIZE)
        res = residual[start:end].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            attn_out = concat_ctx @ w_o.T
            post_attn = res + attn_out
            normed = rms_norm(post_attn, norm_w, norm_eps)
            gate = normed @ w_gate.T
            up = normed @ w_up.T
            mlp_out = (nn.functional.silu(gate) * up) @ w_down.T
            output = post_attn + mlp_out

        all_outputs.append(output.cpu().float())
        del ctx, concat_ctx, res, attn_out, post_attn, normed, gate, up, mlp_out, output
        torch.cuda.empty_cache()

    # Move weights back
    del w_o, w_gate, w_up, w_down, norm_w
    torch.cuda.empty_cache()

    return torch.cat(all_outputs).numpy()


# =============================================================================
# Phase 3: Probe training
# =============================================================================

class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim=HIDDEN_SIZE, out_dim=7, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_train_val_from_features(features, cache):
    task_boundaries = cache["task_boundaries"]
    actions = cache["actions"]
    task_names = cache["task_names"]
    n_tasks = len(task_names)

    task_demos = {i: [] for i in range(n_tasks)}
    offset = 0
    for idx, (task_idx, demo_idx, n_frames) in enumerate(task_boundaries):
        task_demos[task_idx].append((offset, n_frames, idx))
        offset += n_frames

    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list = [], []
    val_task_labels = []

    action_idx = 0
    for task_idx in range(n_tasks):
        demos = task_demos[task_idx]
        n_demos = len(demos)
        n_train = int(n_demos * 0.8)

        for i, (frame_offset, n_frames, boundary_idx) in enumerate(demos):
            feat = features[frame_offset:frame_offset + n_frames]
            act = actions[action_idx]
            action_idx += 1

            T = min(len(feat), len(act))
            if T < 2:
                continue

            x = torch.from_numpy(feat[:T-1].copy()).float()
            y = torch.from_numpy(act[1:T].copy()).float()

            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_task_labels.extend([task_idx] * len(x))

    X_tr = torch.cat(X_tr_list)
    y_tr = torch.cat(y_tr_list)
    X_val = torch.cat(X_val_list)
    y_val = torch.cat(y_val_list)
    val_task_labels = np.array(val_task_labels)

    return X_tr, y_tr, X_val, y_val, val_task_labels


def train_and_eval(X_tr, y_tr, X_val, y_val, val_task_labels, task_names, device="cpu"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp(min=1e-6)
    y_tr_n = (y_tr - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    probe = ActionMLPProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_tr_d = X_tr.to(device)
    y_tr_d = y_tr_n.to(device)
    X_val_d = X_val.to(device)
    y_val_d = y_val_n.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    n_train = len(X_tr_d)
    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            pred = probe(X_tr_d[idx])
            loss = criterion(pred, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val_d)
            val_loss = criterion(val_pred, y_val_d).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    probe.load_state_dict(best_state)
    probe.eval()

    with torch.no_grad():
        val_pred = probe(X_val_d).cpu()

    y_val_orig = y_val
    val_pred_orig = val_pred * y_std + y_mean

    n_tasks = len(task_names)
    per_task_r2 = {}
    for t in range(n_tasks):
        mask = val_task_labels == t
        if mask.sum() == 0:
            continue
        y_t = y_val_orig[mask]
        p_t = val_pred_orig[mask]
        ss_res = ((y_t - p_t) ** 2).sum().item()
        ss_tot = ((y_t - y_t.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        per_task_r2[task_names[t]] = round(r2, 4)

    mean_r2 = np.mean(list(per_task_r2.values()))
    return round(float(mean_r2), 4), per_task_r2


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"

    print("=" * 60)
    print("Random-Subset Ablation Experiment")
    print("=" * 60)

    print("\n[Phase 1] Loading model and caching attention intermediates...")
    model, processor = load_model(args.model_path, device)

    t0 = time.time()
    cache = cache_all_frames(model, processor, args.data_dir, args.demos_per_task, device)
    layer1_weights = extract_layer1_weights(model)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model unloaded. Phase 1 took {(time.time()-t0)/60:.1f} min")

    print("\n[Phase 2] Reconstructing L1 features (GPU, float16 autocast)...")

    print("  Computing baseline (no ablation)...")
    t1 = time.time()
    baseline_features = reconstruct_l1_features(cache, layer1_weights, ablate_heads=None, device=device)
    print(f"    Reconstruction took {time.time()-t1:.1f}s")

    print("  Training baseline probe...")
    X_tr, y_tr, X_val, y_val, val_task_labels = build_train_val_from_features(baseline_features, cache)
    print(f"    Train: {len(X_tr)}, Val: {len(X_val)}")
    baseline_r2, baseline_per_task = train_and_eval(X_tr, y_tr, X_val, y_val, val_task_labels, cache["task_names"])
    print(f"    Baseline R2 = {baseline_r2}")
    del baseline_features, X_tr, y_tr, X_val, y_val

    print("\n  Computing dominant-24 ablation...")
    t1 = time.time()
    dom_features = reconstruct_l1_features(cache, layer1_weights, ablate_heads=INSTRUCTION_HEADS_24, device=device)
    print(f"    Reconstruction took {time.time()-t1:.1f}s")

    print("  Training dominant-24 probe...")
    X_tr, y_tr, X_val, y_val, val_task_labels = build_train_val_from_features(dom_features, cache)
    dominant_r2, dominant_per_task = train_and_eval(X_tr, y_tr, X_val, y_val, val_task_labels, cache["task_names"])
    dominant_r2_increase = round(dominant_r2 - baseline_r2, 4)
    print(f"    Dominant-24 R2 = {dominant_r2}, increase = {dominant_r2_increase}")
    del dom_features, X_tr, y_tr, X_val, y_val

    print(f"\n  Computing {args.n_random} random-24 ablations...")
    random_results = []
    for i in range(args.n_random):
        rng = np.random.RandomState(i)
        random_heads = sorted(rng.choice(NUM_HEADS, size=N_ABLATE, replace=False).tolist())

        t1 = time.time()
        rand_features = reconstruct_l1_features(cache, layer1_weights, ablate_heads=random_heads, device=device)
        recon_time = time.time() - t1

        X_tr, y_tr, X_val, y_val, val_task_labels = build_train_val_from_features(rand_features, cache)
        rand_r2, rand_per_task = train_and_eval(X_tr, y_tr, X_val, y_val, val_task_labels, cache["task_names"])
        r2_increase = round(rand_r2 - baseline_r2, 4)

        random_results.append({
            "seed": i,
            "ablated_heads": random_heads,
            "r2": rand_r2,
            "r2_increase": r2_increase,
            "per_task_r2": rand_per_task,
        })
        print(f"    [{i+1}/{args.n_random}] seed={i}, heads={random_heads[:5]}..., "
              f"R2={rand_r2}, increase={r2_increase} ({recon_time:.1f}s)")

        del rand_features, X_tr, y_tr, X_val, y_val
        gc.collect()

    # --- Phase 3: Statistics ---
    print("\n[Phase 3] Computing statistics...")
    increases = [r["r2_increase"] for r in random_results]
    random_mean = float(np.mean(increases))
    random_sd = float(np.std(increases, ddof=1))
    z_score = (dominant_r2_increase - random_mean) / max(random_sd, 1e-9)
    percentile = float(np.mean([x <= dominant_r2_increase for x in increases])) * 100
    from scipy import stats
    t_stat, p_two_sided = stats.ttest_1samp(increases, dominant_r2_increase)
    p_value = float(p_two_sided / 2) if t_stat < 0 else 1 - float(p_two_sided / 2)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline R2:           {baseline_r2}")
    print(f"  Dominant-24 R2:        {dominant_r2}")
    print(f"  Dominant-24 increase:  {dominant_r2_increase}")
    print(f"  Random-24 mean:        {random_mean:.4f} +/- {random_sd:.4f}")
    print(f"  z-score:               {z_score:.2f}")
    print(f"  Percentile:            {percentile:.1f}%")
    print(f"  p-value (one-sided):   {p_value:.6f}")

    output = {
        "baseline_r2": baseline_r2,
        "baseline_per_task_r2": baseline_per_task,
        "dominant_r2": dominant_r2,
        "dominant_r2_increase": dominant_r2_increase,
        "dominant_per_task_r2": dominant_per_task,
        "dominant_heads": INSTRUCTION_HEADS_24,
        "random_iterations": random_results,
        "statistics": {
            "random_mean": round(random_mean, 4),
            "random_sd": round(random_sd, 4),
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "p_value": round(p_value, 6),
            "n_random": args.n_random,
            "n_ablate": N_ABLATE,
        },
        "metadata": {
            "probe": f"MLP({HIDDEN_SIZE},{HIDDEN_DIM},7)",
            "optimizer": f"Adam lr={LR}",
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "target": "next_action a_{t+1} from feature at time t",
            "normalization": "z-score on targets only (train mean/std)",
            "split": "80/20 per-task by demo index",
            "readout": "last_token at L1",
            "demos_per_task": args.demos_per_task,
            "n_tasks": len(cache["task_names"]),
            "reconstruction": "cached attn intermediates + layer1 recomputation (GPU float16 autocast)",
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(increases, bins=10, color="#4C72B0", edgecolor="white", alpha=0.8, label="Random-24 subsets")
    ax.axvline(dominant_r2_increase, color="red", linewidth=2, linestyle="--",
               label=f"Dominant-24 ({dominant_r2_increase:.4f})")
    ax.axvline(random_mean, color="gray", linewidth=1, linestyle=":",
               label=f"Random mean ({random_mean:.4f})")
    ax.set_xlabel(r"R$^2$ increase (ablated $-$ baseline)")
    ax.set_ylabel("Count")
    ax.set_title(f"Random-Subset Ablation (n={args.n_random})\n"
                 f"z={z_score:.2f}, p={p_value:.4f}, percentile={percentile:.0f}%")
    ax.legend(fontsize=9)
    plt.tight_layout()

    pdf_path = os.path.join(args.fig_dir, "fig_random_subset_ablation.pdf")
    png_path = os.path.join(args.fig_dir, "fig_random_subset_ablation.png")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {pdf_path}")


if __name__ == "__main__":
    main()
