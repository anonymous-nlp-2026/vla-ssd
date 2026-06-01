import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import glob
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
tokenizer = processor.tokenizer

hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
print(f"Found {len(hdf5_files)} task files")

preds_per_layer = [[] for _ in range(N_LAYERS)]
gt_normed_all = []
sample_count = 0


def decode_layer(hs_layer, model):
    action_hs = hs_layer[-7:]
    action_hs = action_hs.to(norm_dtype)
    normed_hs = model.language_model.model.norm(action_hs)
    normed_hs = normed_hs.to(lm_head_dtype)
    logits = model.language_model.lm_head(normed_hs)
    pred_token_ids = logits.float().argmax(dim=-1).cpu().numpy()
    return token_ids_to_action(pred_token_ids), pred_token_ids


# --- Dry run ---
print("\n=== DRY RUN (1 frame) ===")
with h5py.File(hdf5_files[0], "r") as hf:
    demo_key = sorted([k for k in hf["data"].keys() if k.startswith("demo_")])[0]
    img = Image.fromarray(hf[f"data/{demo_key}/obs/agentview_rgb"][0])
    gt_raw = hf[f"data/{demo_key}/actions"][0]

fname = os.path.basename(hdf5_files[0]).replace("_demo.hdf5", "").replace("_", " ")
instruction = fname
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

gt_normed = normalize_action(gt_raw)
gt_tokens = action_to_token_ids(gt_normed)
print(f"GT raw: {gt_raw}")
print(f"GT normed: {gt_normed}")
print(f"GT tokens: {gt_tokens}")
print(f"GT roundtrip: {token_ids_to_action(gt_tokens)}")

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

print(f"Hidden states count: {len(out.hidden_states)} (embed + {len(out.hidden_states)-1} layers)")
print(f"Input IDs shape: {input_ids_tf.shape}")
print(f"HS dtype: {out.hidden_states[1].dtype}")

for l in [0, 15, 31]:
    hs = out.hidden_states[l + 1][0]
    pred_action, pred_tokens = decode_layer(hs, model)
    print(f"Layer {l:2d}: tokens={pred_tokens}, action={pred_action}")

print("\n=== FULL RUN ===")

for fi, hf_path in enumerate(hdf5_files):
    fname = os.path.basename(hf_path).replace("_demo.hdf5", "").replace("_", " ")
    instruction = fname
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    with h5py.File(hf_path, "r") as hf:
        demo_keys = sorted([k for k in hf["data"].keys() if k.startswith("demo_")])[:5]

        for di, dk in enumerate(demo_keys):
            images = hf[f"data/{dk}/obs/agentview_rgb"]
            actions = hf[f"data/{dk}/actions"]
            T = len(actions)
            indices = np.linspace(0, T - 1, 10, dtype=int)

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
                    pred_action, _ = decode_layer(hs, model)
                    preds_per_layer[l].append(pred_action)

                gt_normed_all.append(token_ids_to_action(gt_tokens))
                sample_count += 1

                if sample_count % 50 == 0:
                    print(f"  Processed {sample_count} frames...")

    print(f"Task {fi+1}/{len(hdf5_files)}: {fname} done")

gt_array = np.array(gt_normed_all)
per_layer_r2 = []
per_layer_mse = []

for l in range(N_LAYERS):
    pred_array = np.array(preds_per_layer[l])
    r2 = r2_score(gt_array, pred_array)
    mse = mean_squared_error(gt_array, pred_array)
    per_layer_r2.append(float(r2))
    per_layer_mse.append(float(mse))

print("\n=== RESULTS ===")
for l in range(N_LAYERS):
    print(f"Layer {l:2d}: R²={per_layer_r2[l]:.4f}, MSE={per_layer_mse[l]:.4f}")

results = {
    "model": "openvla-7b (OXE-pretrained, untrained on LIBERO)",
    "dataset": "libero_goal",
    "n_samples": sample_count,
    "per_layer_r2": per_layer_r2,
    "per_layer_mse": per_layer_mse,
    "gpu": "cuda:1",
}

out_path = os.path.join(OUT_DIR, "openvla_goal_untrained_action_head_r2.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

# --- Plot ---
trained_json = os.path.join(OUT_DIR, "openvla_goal_trained_action_head_r2.json")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(8, 4))
layers = np.arange(N_LAYERS)
ax.plot(layers, per_layer_r2, "o-", color="#d62728", label="Untrained (OXE-pretrained)", markersize=4)

if os.path.exists(trained_json):
    with open(trained_json) as f:
        trained = json.load(f)
    ax.plot(layers, trained["per_layer_r2"], "s-", color="#1f77b4", label="Trained (LIBERO-Goal finetuned)", markersize=4)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("Action-Head Decoding: Per-Layer R²", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-0.5, N_LAYERS - 0.5)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig_path = os.path.join(OUT_DIR, "fig_action_head_trained_vs_untrained.pdf")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
print("DONE")
