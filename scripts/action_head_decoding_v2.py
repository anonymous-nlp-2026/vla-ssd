"""Action-Head Decoding v2: argmax + softmax expected-value decoding.

For each layer, apply frozen LayerNorm + LM head to decode actions.
Two decoding methods:
  1) argmax: take the highest-probability token, de-tokenize
  2) softmax EV: E[action] = sum(softmax(action_logits) * bin_centers)

Compare both with ground truth and with independent probe R².
"""
import os, json, time, glob
import numpy as np, torch, h5py
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
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return dict(
        q01=np.array(stats["q01"]), q99=np.array(stats["q99"]),
        mask=np.array(stats["mask"]), vocab_size=vocab_size,
        bins=bins, bin_centers=bin_centers,
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
def decode_layers(model, inp, codec, action_bin_centers_t):
    norm_fn = model.language_model.model.norm
    lm_head = model.language_model.lm_head
    vs = codec["vocab_size"]
    action_lo = vs - N_ACTION_BINS + 1
    action_hi = vs  # exclusive upper bound for slice

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inp, output_hidden_states=True)

    actual_seq_len = out.hidden_states[0].shape[1]
    positions = list(range(actual_seq_len - ACTION_DIM - 1, actual_seq_len - 1))

    preds_argmax = {}
    preds_ev = {}
    for li in range(N_HIDDEN):
        hs = out.hidden_states[li][0, positions, :]  # [7, 4096]
        logits = lm_head(norm_fn(hs)).float()  # [7, vocab_size]

        # Argmax decoding
        tok_ids = logits.argmax(dim=-1).cpu().numpy()
        preds_argmax[li] = tokens_to_norm(tok_ids, codec)

        # Softmax EV over action tokens
        action_logits = logits[:, action_lo:action_hi]  # [7, 255]
        action_probs = torch.softmax(action_logits, dim=-1)  # [7, 255]
        ev = (action_probs * action_bin_centers_t.unsqueeze(0)).sum(dim=-1)
        preds_ev[li] = ev.cpu().numpy()  # [7]

    del out
    return preds_argmax, preds_ev


@torch.no_grad()
def run_experiment(model, proc, codec, device, action_bin_centers_t):
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not hdf5_files:
        hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files")

    all_gt = []
    all_argmax = {li: [] for li in range(N_HIDDEN)}
    all_ev = {li: [] for li in range(N_HIDDEN)}
    n = 0
    t_start = time.time()

    for hf_path in hdf5_files:
        task = task_desc_from_filename(hf_path)
        prompt = f"In: What action should the robot take to {task}?\nOut:"

        with h5py.File(hf_path, "r") as hf:
            dkeys = sorted(hf["data"].keys(), key=lambda x: int(x.split("_")[1]))[:MAX_DEMOS]
            for dk in dkeys:
                imgs = np.array(hf[f"data/{dk}/obs/agentview_rgb"])
                acts = np.array(hf[f"data/{dk}/actions"])
                T = imgs.shape[0]
                idxs = (list(range(T)) if T <= FRAMES_PER_DEMO
                        else np.linspace(0, T-1, FRAMES_PER_DEMO, dtype=int).tolist())

                for t in idxs:
                    img = Image.fromarray(imgs[t])
                    inp, gt_n = make_tf_input(proc, prompt, img, acts[t], codec, device)
                    p_argmax, p_ev = decode_layers(model, inp, codec, action_bin_centers_t)

                    all_gt.append(gt_n)
                    for li in range(N_HIDDEN):
                        all_argmax[li].append(p_argmax[li])
                        all_ev[li].append(p_ev[li])
                    n += 1
                    del inp

        elapsed = time.time() - t_start
        rate = n / elapsed if elapsed > 0 else 0
        print(f"  {task[:45]:45s}  [{n:5d} samples, {rate:.1f} samp/s]")

    gt_arr = np.array(all_gt)
    results = {}
    for li in range(N_HIDDEN):
        pa = np.array(all_argmax[li])
        pe = np.array(all_ev[li])
        ss_tot = np.sum((gt_arr - gt_arr.mean(0, keepdims=True))**2, axis=0)

        ss_res_a = np.sum((pa - gt_arr)**2, axis=0)
        r2_a = 1 - ss_res_a / (ss_tot + 1e-12)

        ss_res_e = np.sum((pe - gt_arr)**2, axis=0)
        r2_e = 1 - ss_res_e / (ss_tot + 1e-12)

        results[f"L{li}"] = dict(
            argmax_mean_R2=float(np.mean(r2_a)),
            argmax_overall_R2=float(1 - ss_res_a.sum()/(ss_tot.sum()+1e-12)),
            argmax_r2_per_dim=[float(x) for x in r2_a],
            argmax_mse=float(np.mean((pa - gt_arr)**2)),
            ev_mean_R2=float(np.mean(r2_e)),
            ev_overall_R2=float(1 - ss_res_e.sum()/(ss_tot.sum()+1e-12)),
            ev_r2_per_dim=[float(x) for x in r2_e],
            ev_mse=float(np.mean((pe - gt_arr)**2)),
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
    ah_argmax = [results[f"L{l}"]["argmax_mean_R2"] for l in layers]
    ah_ev = [results[f"L{l}"]["ev_mean_R2"] for l in layers]
    pr = [pd["trained"]["last_preaction"][f"layer_{l}"]["mean_R2"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, ah_argmax, "o-",  color="#e74c3c", lw=2, ms=4, label="Action Head: argmax")
    ax.plot(layers, ah_ev, "^-",  color="#e67e22", lw=2, ms=4, label="Action Head: softmax EV")
    ax.plot(layers, pr, "s--", color="#3498db", lw=2, ms=4, label="Independent Probe (trained MLP)")
    ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("R²", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.5, N_HIDDEN - 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf = os.path.join(OUT_DIR, "fig_action_head_vs_probe.pdf")
    plt.savefig(pdf, dpi=150)
    plt.savefig(pdf.replace(".pdf", ".png"), dpi=150)
    print(f"Figure: {pdf}")


def main():
    print("=== Action Head Decoding v2: OpenVLA LIBERO-Goal ===")
    t0 = time.time()

    model, proc = load_model(CKPT, DEVICE)
    codec = get_codec(model)
    vs = codec["vocab_size"]
    action_lo = vs - N_ACTION_BINS + 1
    print(f"Action vocab_size: {vs}, action token range: [{action_lo}, {vs-1}]")

    # Precompute bin centers for action tokens on GPU
    action_bin_centers_t = torch.tensor(
        [codec["bin_centers"][min(vs-1-tok, len(codec["bin_centers"])-1)]
         for tok in range(action_lo, vs)],
        dtype=torch.float32, device=DEVICE
    )

    print("\n-- FULL EXPERIMENT --")
    results, n = run_experiment(model, proc, codec, DEVICE, action_bin_centers_t)

    elapsed = time.time() - t0
    out = dict(
        experiment="action_head_decoding_v2",
        model="openvla-7b-finetuned-libero-goal",
        n_samples=n, frames_per_demo=FRAMES_PER_DEMO, max_demos=MAX_DEMOS,
        per_layer=results, elapsed_seconds=elapsed,
    )
    jpath = os.path.join(OUT_DIR, "openvla_goal_action_head_r2_v2.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON: {jpath}")

    plot(results)

    print("\n-- SUMMARY --")
    print(f"{'Layer':>6} {'Argmax_R2':>10} {'EV_R2':>10}")
    for l in [0, 4, 8, 13, 16, 20, 24, 28, 31, 32]:
        r = results[f"L{l}"]
        print(f"  L{l:2d}  {r['argmax_mean_R2']:10.4f} {r['ev_mean_R2']:10.4f}")
    print(f"Done: {n} samples, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
