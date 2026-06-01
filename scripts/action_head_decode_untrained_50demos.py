import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json, time, glob
import numpy as np
import torch
import h5py
from PIL import Image
from sklearn.metrics import r2_score, mean_squared_error
from transformers import AutoModelForImageTextToText, AutoProcessor

UNTRAINED_PATH = "./checkpoints/openvla-7b"
TRAINED_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal"
OUT_DIR = "./results/action_head_decoding"

N_LAYERS = 32
N_BINS = 256
N_DEMOS = 50
N_FRAMES_PER_DEMO = 10
VOCAB_SIZE_EFFECTIVE = 32064 - 64
BINS = np.linspace(-1, 1, N_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0

with open(os.path.join(TRAINED_PATH, "dataset_statistics.json")) as f:
    stats = json.load(f)["libero_goal"]["action"]
Q01 = np.array(stats["q01"])
Q99 = np.array(stats["q99"])
MASK = np.array(stats["mask"])


def normalize_action(raw_action):
    normed = np.where(
        MASK,
        np.clip(2.0 * (raw_action - Q01) / (Q99 - Q01 + 1e-8) - 1.0, -1.0, 1.0),
        raw_action,
    )
    return normed


def action_to_token_ids(normed_action):
    bin_indices = np.digitize(normed_action, BINS) - 1
    bin_indices = np.clip(bin_indices, 0, N_BINS - 2)
    token_ids = VOCAB_SIZE_EFFECTIVE - 1 - bin_indices
    return token_ids


def token_ids_to_action(token_ids):
    bin_indices = VOCAB_SIZE_EFFECTIVE - token_ids - 1
    bin_indices = np.clip(bin_indices, 0, len(BIN_CENTERS) - 1)
    return BIN_CENTERS[bin_indices]


def decode_layer(hs_layer, model, lm_head_dtype, norm_dtype):
    action_hs = hs_layer[-7:]
    action_hs = action_hs.to(norm_dtype)
    normed_hs = model.language_model.model.norm(action_hs)
    normed_hs = normed_hs.to(lm_head_dtype)
    logits = model.language_model.lm_head(normed_hs)
    pred_token_ids = logits.float().argmax(dim=-1).cpu().numpy()
    return token_ids_to_action(pred_token_ids)


def main():
    t0 = time.time()

    print("Loading untrained model...")
    model = AutoModelForImageTextToText.from_pretrained(
        UNTRAINED_PATH, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to("cuda:0")
    model.eval()

    lm_head_dtype = next(model.language_model.lm_head.parameters()).dtype
    norm_dtype = next(model.language_model.model.norm.parameters()).dtype
    print(f"LM head dtype: {lm_head_dtype}, Norm dtype: {norm_dtype}")

    processor = AutoProcessor.from_pretrained(UNTRAINED_PATH, trust_remote_code=True)

    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
    print(f"Found {len(hdf5_files)} task files, N_DEMOS={N_DEMOS}, N_FRAMES={N_FRAMES_PER_DEMO}")

    preds_per_layer = [[] for _ in range(N_LAYERS)]
    gt_normed_all = []
    sample_count = 0

    for fi, hf_path in enumerate(hdf5_files):
        fname = os.path.basename(hf_path).replace("_demo.hdf5", "").replace("_", " ")
        instruction = fname
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        with h5py.File(hf_path, "r") as hf:
            demo_keys = sorted(
                [k for k in hf["data"].keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1])
            )[:N_DEMOS]

            for di, dk in enumerate(demo_keys):
                images = hf[f"data/{dk}/obs/agentview_rgb"]
                actions = hf[f"data/{dk}/actions"]
                T = len(actions)
                indices = np.linspace(0, T - 1, N_FRAMES_PER_DEMO, dtype=int)

                for t in indices:
                    img = Image.fromarray(images[t])
                    gt_raw = actions[t]
                    gt_normed = normalize_action(gt_raw)
                    gt_tokens = action_to_token_ids(gt_normed)

                    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
                    input_ids = inputs["input_ids"]
                    gt_token_tensor = torch.tensor(gt_tokens, dtype=torch.long, device="cuda:0").unsqueeze(0)
                    input_ids_tf = torch.cat([input_ids, gt_token_tensor], dim=1)

                    with torch.no_grad():
                        out = model(
                            input_ids=input_ids_tf,
                            attention_mask=torch.ones_like(input_ids_tf),
                            pixel_values=inputs.get("pixel_values"),
                            output_hidden_states=True,
                        )

                    for l in range(N_LAYERS):
                        hs = out.hidden_states[l + 1][0]
                        pred_action = decode_layer(hs, model, lm_head_dtype, norm_dtype)
                        preds_per_layer[l].append(pred_action)

                    gt_normed_all.append(token_ids_to_action(gt_tokens))
                    sample_count += 1

                    if sample_count % 100 == 0:
                        elapsed = time.time() - t0
                        rate = sample_count / elapsed
                        print(f"  Processed {sample_count} frames... ({rate:.1f} samp/s)")

                    del out, inputs, input_ids_tf

        print(f"Task {fi+1}/{len(hdf5_files)}: {fname} done ({sample_count} total)")

    elapsed_total = time.time() - t0

    gt_array = np.array(gt_normed_all)
    per_layer = {}
    per_layer_r2_flat = []
    per_layer_mse_flat = []

    for l in range(N_LAYERS):
        pred_array = np.array(preds_per_layer[l])
        ss_res = np.sum((pred_array - gt_array)**2, axis=0)
        ss_tot = np.sum((gt_array - gt_array.mean(0, keepdims=True))**2, axis=0)
        r2_dim = 1 - ss_res / (ss_tot + 1e-12)
        mean_r2 = float(np.mean(r2_dim))
        overall_r2 = float(1 - ss_res.sum() / (ss_tot.sum() + 1e-12))
        mse = float(np.mean((pred_array - gt_array)**2))

        per_layer[f"L{l}"] = dict(
            mean_R2=mean_r2,
            overall_R2=overall_r2,
            r2_per_dim=[float(x) for x in r2_dim],
            mse=mse,
        )
        per_layer_r2_flat.append(overall_r2)
        per_layer_mse_flat.append(mse)

    print("\n=== RESULTS ===")
    for l in range(N_LAYERS):
        print(f"Layer {l:2d}: R²={per_layer[f'L{l}']['overall_R2']:.4f}, MSE={per_layer[f'L{l}']['mse']:.4f}")

    results = {
        "experiment": "action_head_decoding",
        "model": "openvla-7b (OXE-pretrained, untrained on LIBERO)",
        "dataset": "libero_goal",
        "n_demos": N_DEMOS,
        "n_frames_per_demo": N_FRAMES_PER_DEMO,
        "n_samples": sample_count,
        "per_layer": per_layer,
        "per_layer_r2": per_layer_r2_flat,
        "per_layer_mse": per_layer_mse_flat,
        "elapsed_seconds": elapsed_total,
        "gpu": "cuda:1",
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "openvla_goal_untrained_50demos_action_head_r2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total: {sample_count} samples, {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
