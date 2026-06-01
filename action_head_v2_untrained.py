"""Action-Head Decoding v2: Untrained OpenVLA on LIBERO-Goal.
Control group for trained v2. Uses same 3 bug fixes.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch, json, glob, h5py, time
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from sklearn.metrics import r2_score

# === Config ===
MODEL_PATH = "./checkpoints/openvla-7b"
TRAINED_MODEL_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal/"
N_DEMOS = 50
N_FRAMES_PER_DEMO = 10
N_LAYERS = 32
OUT_DIR = "./results/action_head_decoding_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load model ===
print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

vocab_size = model.config.text_config.vocab_size
n_bins = 256
bins = np.linspace(-1, 1, n_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

final_norm = model.language_model.model.norm
lm_head = model.language_model.lm_head

def detokenize_actions(token_ids):
    bin_idx = np.clip(vocab_size - 1 - token_ids, 0, len(bin_centers) - 1)
    return bin_centers[bin_idx]

def tokenize_actions(actions_norm):
    bin_idx = np.digitize(actions_norm, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 2)
    return vocab_size - 1 - bin_idx

# === Sanity check (1 sample) ===
print("\n=== SANITY CHECK ===")
hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
with h5py.File(hdf5_files[0], "r") as hf:
    img = Image.fromarray(hf["data/demo_0/obs/agentview_rgb"][0])
    gt_action_raw = hf["data/demo_0/actions"][0]

fname = os.path.basename(hdf5_files[0]).replace("_demo.hdf5", "").replace("_", " ")
prompt = f"In: What action should the robot take to {fname}?\nOut:"
inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

print("Aliasing check:")
for i in range(len(out.hidden_states) - 1):
    if out.hidden_states[i].data_ptr() == out.hidden_states[i+1].data_ptr():
        print(f"  WARNING: hidden_states[{i}] and [{i+1}] share data_ptr!")

for i in range(len(out.hidden_states) - 1):
    diff = (out.hidden_states[i] - out.hidden_states[i+1]).abs().max().item()
    if diff == 0.0:
        print(f"  IDENTICAL content: hidden_states[{i}] == [{i+1}]")

hs_exit = out.hidden_states[-1][0, -1, :].clone()
logits_manual = lm_head(hs_exit.unsqueeze(0))
logits_model = out.logits[0, -1, :]
diff = (logits_manual[0] - logits_model).abs().max().item()
print(f"Exit layer logits match (no norm): max diff = {diff:.6f}")

logits_double = lm_head(final_norm(hs_exit.unsqueeze(0)))
diff2 = (logits_double[0] - logits_model).abs().max().item()
print(f"Exit layer logits match (with norm = double): max diff = {diff2:.6f}")

with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=7, do_sample=False)
gen_tokens = gen_ids[0, -7:].cpu().numpy()
gen_action = detokenize_actions(gen_tokens)
print(f"Generate tokens: {gen_tokens}")
print(f"Generate action:  {gen_action}")
print(f"Unique tokens: {len(set(gen_tokens))}")
if len(set(gen_tokens)) == 1:
    print("FATAL: All generated tokens identical! Model still broken.")
    print("Check transformers version and model loading.")
    exit(1)
else:
    print("PASS: Generate produces diverse tokens.")

exit_argmax = logits_manual[0].argmax().item()
print(f"Exit layer argmax token: {exit_argmax}")
print(f"Generate first token:    {gen_tokens[0]}")

print("\n=== SANITY CHECK PASSED ===\n")

print("Checking norm status of hidden_states...")
hs_31 = out.hidden_states[31][0, -1, :].clone()
hs_32 = out.hidden_states[32][0, -1, :].clone()
print(f"  hs[31] L2 norm: {hs_31.float().norm().item():.2f}")
print(f"  hs[32] L2 norm: {hs_32.float().norm().item():.2f}")

# === Main experiment ===
print("Starting main experiment (50 demos/task, 10 frames/demo)...")
start_time = time.time()

# Load norm stats from TRAINED model for fair comparison
norm_stats = None
stats_path = os.path.join(TRAINED_MODEL_PATH, "dataset_statistics.json")
if os.path.exists(stats_path):
    with open(stats_path) as f:
        all_stats = json.load(f)
    for k in all_stats:
        if "libero" in k.lower() or "goal" in k.lower():
            norm_stats = all_stats[k]
            print(f"Using norm stats from trained model, key: {k}")
            break
    if norm_stats is None:
        first_key = list(all_stats.keys())[0]
        norm_stats = all_stats[first_key]
        print(f"Using norm stats from trained model, first key: {first_key}")
else:
    print("WARNING: No dataset_statistics.json found, using raw actions")

def normalize_action(action_raw, stats):
    if stats is None:
        return action_raw
    mean = np.array(stats["action"]["mean"])
    std = np.array(stats["action"]["std"])
    if "q01" in stats["action"] and "q99" in stats["action"]:
        low = np.array(stats["action"]["q01"])
        high = np.array(stats["action"]["q99"])
        normalized = 2.0 * (action_raw - low) / (high - low + 1e-8) - 1.0
    else:
        normalized = (action_raw - mean) / (std + 1e-8)
    return np.clip(normalized, -1.0, 1.0)

all_preds = {l: [] for l in range(N_LAYERS + 1)}
all_gt_norm = []
n_total = 0

for task_idx, hf_path in enumerate(hdf5_files):
    fname = os.path.basename(hf_path).replace("_demo.hdf5", "").replace("_", " ")
    instruction = fname
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    with h5py.File(hf_path, "r") as hf:
        demo_keys = sorted([k for k in hf["data"].keys() if k.startswith("demo_")],
                          key=lambda x: int(x.split("_")[1]))

        for dk in demo_keys[:N_DEMOS]:
            actions_raw = hf[f"data/{dk}/actions"][:]
            images = hf[f"data/{dk}/obs/agentview_rgb"]
            T = len(actions_raw)
            indices = np.linspace(0, T-1, N_FRAMES_PER_DEMO, dtype=int)

            for t in indices:
                img = Image.fromarray(images[t])
                gt_raw = actions_raw[t]
                gt_norm = normalize_action(gt_raw, norm_stats)
                gt_tokens = tokenize_actions(gt_norm)

                inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
                gt_token_tensor = torch.tensor([gt_tokens], dtype=torch.long, device="cuda:0")
                input_ids = torch.cat([inputs["input_ids"], gt_token_tensor], dim=1)

                with torch.no_grad():
                    out = model(
                        input_ids=input_ids,
                        pixel_values=inputs.get("pixel_values", inputs.get("pixel_values")).to("cuda:0", dtype=torch.bfloat16),
                        output_hidden_states=True
                    )

                n_hs = len(out.hidden_states)

                for l in range(min(n_hs, N_LAYERS + 1)):
                    hs = out.hidden_states[l][0, -7:, :].clone()

                    if l < N_LAYERS:
                        normed = final_norm(hs)
                        logits = lm_head(normed)
                    else:
                        logits = lm_head(hs)

                    pred_tokens = logits.argmax(dim=-1).cpu().numpy()
                    pred_action = detokenize_actions(pred_tokens)
                    all_preds[l].append(pred_action)

                all_gt_norm.append(gt_norm)
                n_total += 1

                if n_total % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = n_total / elapsed
                    print(f"  {n_total} samples, {rate:.1f} samp/s, elapsed {elapsed:.0f}s")

    print(f"Task {task_idx+1}/10 done ({fname[:40]}...)")

# === Compute R2 per layer ===
gt_array = np.array(all_gt_norm)
results = {
    "model": "openvla-7b (untrained, v2)",
    "dataset": "libero_goal",
    "n_demos": N_DEMOS,
    "n_samples": n_total,
    "transformers_version": "4.40.1",
    "fixes": ["no_double_norm", "clone_hidden_states", "correct_transformers"],
}

per_layer_r2 = []
per_layer_mse = []
for l in range(min(len(all_preds), N_LAYERS + 1)):
    if len(all_preds[l]) == 0:
        continue
    preds = np.array(all_preds[l])
    r2 = r2_score(gt_array, preds)
    mse = float(np.mean((gt_array - preds) ** 2))
    per_layer_r2.append(float(r2))
    per_layer_mse.append(mse)

results["per_layer_r2"] = per_layer_r2
results["per_layer_mse"] = per_layer_mse

print(f"\n=== RESULTS ({n_total} samples) ===")
for l, (r2, mse) in enumerate(zip(per_layer_r2, per_layer_mse)):
    marker = " <- EXIT" if l == N_LAYERS else ""
    print(f"  Layer {l:2d}: R2={r2:+.4f}, MSE={mse:.4f}{marker}")

print(f"\nPeak R2: {max(per_layer_r2):.4f} at layer {np.argmax(per_layer_r2)}")
print(f"Exit R2: {per_layer_r2[-1]:.4f}")

out_path = os.path.join(OUT_DIR, "untrained_v2_action_head_r2.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

elapsed = time.time() - start_time
print(f"Total time: {elapsed:.0f}s ({n_total/elapsed:.1f} samp/s)")
