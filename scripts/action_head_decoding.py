"""Action-Head Decoding: apply frozen LM head to each layer's hidden state.

Teacher forcing with ground truth action tokens. Compare decoded actions
against ground truth to get per-layer R², then compare with independent
probe R² to distinguish genuine information collapse from format specialization.
"""

import os, json, time, glob
import numpy as np
import torch
import h5py
from pathlib import Path
from PIL import Image

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

CKPT = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal"
OUT_DIR = "./results/action_head_decoding"
PROBE_JSON = "./results/functional_validation/fulllayer_action_probe.json"
DEVICE = "cuda:0"
MAX_DEMOS = 50
FRAMES_PER_DEMO = 10
N_HIDDEN = 33
ACTION_DIM = 7
N_ACTION_BINS = 256
EMPTY_TOKEN = 29871

os.makedirs(OUT_DIR, exist_ok=True)


def task_desc_from_filename(fname):
    return Path(fname).stem.replace("_demo", "").replace("_", " ")


def load_model(ckpt, device):
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    ModelClass = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction", ckpt)
    model = ModelClass.from_pretrained(
        ckpt, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.1f}B on {device}")
    return model, proc


def get_codec(model):
    stats = model.norm_stats["libero_goal"]["action"]
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    bins = np.linspace(-1, 1, N_ACTION_BINS)
    return dict(
        q01=np.array(stats["q01"]), q99=np.array(stats["q99"]),
        mask=np.array(stats["mask"]), vocab_size=vocab_size,
        bins=bins, bin_centers=(bins[:-1] + bins[1:]) / 2.0,
    )


def raw_to_norm(action, c):
    n = np.where(c["mask"], 2*(action - c["q01"])/(c["q99"] - c["q01"] + 1e-8) - 1, action)
    return np.clip(n, -1, 1)


def norm_to_tokens(norm_action, c):
    idx = np.clip(np.digitize(norm_action, c["bins"]) - 1, 0, N_ACTION_BINS - 2)
    return c["vocab_size"] - 1 - idx


def tokens_to_norm(token_ids, c):
    d = np.clip(c["vocab_size"] - token_ids - 1, 0, len(c["bin_centers"]) - 1)
    return c["bin_centers"][d]


def make_tf_input(proc, prompt, image, gt_action, codec, device):
    inp = proc(text=prompt, images=image, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}

    if inp["input_ids"][0, -1].item() != EMPTY_TOKEN:
        inp["input_ids"] = torch.cat([
            inp["input_ids"],
            torch.tensor([[EMPTY_TOKEN]], dtype=torch.long, device=device)
        ], dim=1)
        if "attention_mask" in inp:
            inp["attention_mask"] = torch.cat([
                inp["attention_mask"],
                torch.ones(1, 1, dtype=inp["attention_mask"].dtype, device=device)
            ], dim=1)

    gt_norm = raw_to_norm(gt_action, codec)
    gt_toks = norm_to_tokens(gt_norm, codec)
    action_ids = torch.tensor(gt_toks, dtype=torch.long, device=device).unsqueeze(0)
    inp["input_ids"] = torch.cat([inp["input_ids"], action_ids], dim=1)
    if "attention_mask" in inp:
        inp["attention_mask"] = torch.cat([
            inp["attention_mask"],
            torch.ones(1, ACTION_DIM, dtype=inp["attention_mask"].dtype, device=device)
        ], dim=1)
    return inp, gt_norm


@torch.no_grad()
def decode_layers(model, inp, codec):
    norm_fn = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inp, output_hidden_states=True)

    # IMPORTANT: actual seq_len includes 256 image tokens inserted by Prismatic
    # Use hidden_states tensor shape, NOT input_ids shape
    actual_seq_len = out.hidden_states[0].shape[1]

    # 7 positions predicting action tokens (relative to end of actual sequence):
    #   pos[-8] = empty_token → predicts action_0
    #   pos[-7] = action_0   → predicts action_1
    #   ...
    #   pos[-2] = action_5   → predicts action_6
    positions = list(range(actual_seq_len - ACTION_DIM - 1, actual_seq_len - 1))

    preds = {}
    for li in range(N_HIDDEN):
        hs = out.hidden_states[li][0, positions, :]
        logits = lm_head(norm_fn(hs))
        tok_ids = logits.argmax(dim=-1).cpu().numpy()
        preds[li] = tokens_to_norm(tok_ids, codec)

    del out
    return preds


@torch.no_grad()
def dry_run(model, proc, codec, device):
    hf_path = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))[0]
    task = task_desc_from_filename(hf_path)
    prompt = f"In: What action should the robot take to {task}?\nOut:"

    with h5py.File(hf_path, "r") as hf:
        img = Image.fromarray(np.array(hf["data/demo_0/obs/agentview_rgb"][0]))
        gt = np.array(hf["data/demo_0/actions"][0])

    gt_norm = raw_to_norm(gt, codec)
    print(f"Task: {task}")
    print(f"GT norm: {gt_norm}")

    inp, _ = make_tf_input(proc, prompt, img, gt, codec, device)
    print(f"Input input_ids len: {inp['input_ids'].shape[1]}")

    preds = decode_layers(model, inp, codec)
    # Print hidden_states seq len for verification
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_check = model(**inp, output_hidden_states=True)
    hs_seq_len = out_check.hidden_states[0].shape[1]
    print(f"Actual hidden_states seq len: {hs_seq_len}  (input_ids + 256 image tokens)")
    del out_check

    for li in [0, 8, 16, 24, 31, 32]:
        p = preds[li]
        print(f"  L{li:2d}: [{', '.join(f'{x:+.4f}' for x in p)}]")
    print(f"  GT:   [{', '.join(f'{x:+.4f}' for x in gt_norm)}]")

    # Verify exit layer: prompt-only forward → first predicted token
    inp0 = proc(text=prompt, images=img, return_tensors="pt")
    inp0 = {k: v.to(device) for k, v in inp0.items()}
    if inp0["input_ids"][0, -1].item() != EMPTY_TOKEN:
        inp0["input_ids"] = torch.cat([
            inp0["input_ids"],
            torch.tensor([[EMPTY_TOKEN]], dtype=torch.long, device=device)
        ], dim=1)
        if "attention_mask" in inp0:
            inp0["attention_mask"] = torch.cat([
                inp0["attention_mask"],
                torch.ones(1, 1, dtype=inp0["attention_mask"].dtype, device=device)
            ], dim=1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out0 = model(**inp0, output_hidden_states=True)
    hs_exit = out0.hidden_states[32][0, -1, :]
    logit0 = model.language_model.lm_head(model.language_model.model.norm(hs_exit.unsqueeze(0)))
    tok0 = logit0[0].argmax().item()
    pred0_norm = tokens_to_norm(np.array([tok0]), codec)[0]
    del out0

    tf_first = preds[32][0]
    print(f"\nExit-layer first-dim check:")
    print(f"  Teacher-forcing: {tf_first:+.6f}")
    print(f"  Prompt-only:     {pred0_norm:+.6f}")
    print(f"  Match: {np.isclose(tf_first, pred0_norm)}")
    assert np.isclose(tf_first, pred0_norm), "EXIT LAYER MISMATCH - implementation bug!"
    print("Dry run OK\n")


@torch.no_grad()
def run_experiment(model, proc, codec, device):
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
    print(f"Tasks: {len(hdf5_files)}")

    all_gt, all_pred = [], {li: [] for li in range(N_HIDDEN)}
    n = 0
    t_start = time.time()

    for hf_path in hdf5_files:
        task = task_desc_from_filename(hf_path)
        prompt = f"In: What action should the robot take to {task}?\nOut:"

        with h5py.File(hf_path, "r") as hf:
            dkeys = sorted(hf["data"].keys(), key=lambda x: int(x.split("_")[1]))[:MAX_DEMOS]
            for dk in dkeys:
                imgs = hf[f"data/{dk}/obs/agentview_rgb"]
                acts = np.array(hf[f"data/{dk}/actions"])
                T = imgs.shape[0]
                idxs = (list(range(T)) if T <= FRAMES_PER_DEMO
                        else np.linspace(0, T-1, FRAMES_PER_DEMO, dtype=int).tolist())

                for t in idxs:
                    img = Image.fromarray(imgs[t])
                    inp, gt_n = make_tf_input(proc, prompt, img, acts[t], codec, device)
                    preds = decode_layers(model, inp, codec)

                    all_gt.append(gt_n)
                    for li in range(N_HIDDEN):
                        all_pred[li].append(preds[li])
                    n += 1
                    del inp

        elapsed = time.time() - t_start
        rate = n / elapsed if elapsed > 0 else 0
        print(f"  {task[:45]:45s}  [{n:5d} samples, {rate:.1f} samp/s]")

    gt_arr = np.array(all_gt)
    results = {}
    for li in range(N_HIDDEN):
        pa = np.array(all_pred[li])
        ss_res = np.sum((pa - gt_arr)**2, axis=0)
        ss_tot = np.sum((gt_arr - gt_arr.mean(0, keepdims=True))**2, axis=0)
        r2_dim = 1 - ss_res / (ss_tot + 1e-12)
        results[f"L{li}"] = dict(
            mean_R2=float(np.mean(r2_dim)),
            overall_R2=float(1 - ss_res.sum()/(ss_tot.sum()+1e-12)),
            r2_per_dim=[float(x) for x in r2_dim],
            mse=float(np.mean((pa - gt_arr)**2)),
        )
    return results, n


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    with open(PROBE_JSON) as f:
        pd = json.load(f)

    layers = list(range(N_HIDDEN))
    ah = [results[f"L{l}"]["mean_R2"] for l in layers]
    pr = [pd["trained"]["last_preaction"][f"layer_{l}"]["mean_R2"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(layers, ah, "o-",  color="#e74c3c", lw=2, ms=4, label="Action Head (frozen LM head)")
    ax.plot(layers, pr, "s--", color="#3498db", lw=2, ms=4, label="Independent Probe (MLP)")
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("R²", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.5, N_HIDDEN - 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf = os.path.join(OUT_DIR, "fig_action_head_vs_probe.pdf")
    plt.savefig(pdf, dpi=150)
    plt.savefig(pdf.replace(".pdf", ".png"), dpi=150)
    print(f"Figure: {pdf}")


def main():
    print("=== Action Head Decoding: OpenVLA LIBERO-Goal ===")
    t0 = time.time()

    model, proc = load_model(CKPT, DEVICE)
    codec = get_codec(model)
    print(f"Action vocab_size: {codec['vocab_size']}")

    print("\n-- DRY RUN --")
    dry_run(model, proc, codec, DEVICE)

    print("-- FULL EXPERIMENT --")
    results, n = run_experiment(model, proc, codec, DEVICE)

    elapsed = time.time() - t0
    out = dict(
        experiment="action_head_decoding",
        model="openvla-7b-finetuned-libero-goal",
        n_samples=n, frames_per_demo=FRAMES_PER_DEMO, max_demos=MAX_DEMOS,
        per_layer=results, elapsed_seconds=elapsed,
    )
    jpath = os.path.join(OUT_DIR, "openvla_goal_action_head_r2.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON: {jpath}")

    plot(results)

    print("\n-- SUMMARY --")
    for l in [0, 8, 13, 16, 24, 31, 32]:
        print(f"  L{l:2d}: R2={results[f'L{l}']['mean_R2']:.4f}")
    print(f"Done: {n} samples, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
