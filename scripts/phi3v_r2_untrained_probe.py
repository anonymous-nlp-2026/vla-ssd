"""Phi3V UNTRAINED (base Phi-3 LLM) per-layer action R² probe.

Same pipeline as phi3v_r2_action_probe.py, but with base Phi-3-mini weights
replacing the LLM decoder layers (vision encoder + projector kept from TracVLA).

Phase 1: Extract last-preaction hidden states from all 33 layers.
Phase 2: Train per-layer MLP probes (5 seeds), compute R².
"""
import gc
import glob
import json
import os
import time
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn

TRACVLA_PATH = "./checkpoints/tracevla-phi3v/"
BASE_PHI3_PATHS = [
    "<HF_CACHE>/Phi-3-mini-4k-instruct",
    "<HF_CACHE>/models--microsoft--Phi-3-mini-4k-instruct",
]
DATA_DIR = "./data/libero/libero_goal/"
FEATURE_DIR = "./features/phi3v_untrained_libero_goal/"
OUT_JSON = "./results/phi3v_r2_untrained_probe/phi3v_r2_untrained_probe.json"

N_LAYERS = 33
NUM_DECODER_LAYERS = 32
HIDDEN_DIM_MODEL = 3072
PROBE_HIDDEN = 256
ACTION_DIM = 7
SEEDS = list(range(5))
PROBE_BATCH = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
MAX_DEMOS = 50
DEVICE = "cuda:0"


# ── Phase 1: Feature Extraction ─────────────────────────────────────────

def task_desc_from_filename(fname):
    return Path(fname).stem.replace("_demo", "").replace("_", " ")


def find_base_phi3():
    for p in BASE_PHI3_PATHS:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    for p in BASE_PHI3_PATHS:
        parent = Path(p)
        if parent.exists():
            for child in parent.rglob("config.json"):
                return str(child.parent)
    raise FileNotFoundError(f"Base Phi-3-mini not found in {BASE_PHI3_PATHS}")


def load_untrained_model(device_str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        TRACVLA_PATH, trust_remote_code=True, local_files_only=True
    )

    print("Loading TracVLA-Phi3V architecture (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        TRACVLA_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    base_path = find_base_phi3()
    print(f"Loading base Phi-3-mini from {base_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"Replacing LLM decoder layer weights ({NUM_DECODER_LAYERS} layers)...")
    for i in range(NUM_DECODER_LAYERS):
        model.model.layers[i].load_state_dict(base_model.model.layers[i].state_dict())

    try:
        model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        print("Replaced embed_tokens + lm_head")
    except RuntimeError as e:
        print(f"Skipping embed_tokens/lm_head (dim mismatch): {e}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device_str)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Untrained hybrid model on {device_str}: {n_params:.1f}B params")
    return model, processor


@torch.no_grad()
def extract_one_task(model, processor, hdf5_path, task_desc, device, out_path):
    from PIL import Image

    prompt = (
        f"<|user|>\n<|image_1|>\n"
        f"What action should the robot take to {task_desc}?<|end|>\n"
        f"<|assistant|>\n"
    )

    f = h5py.File(hdf5_path, "r")
    demo_keys = sorted(
        [k for k in f["data"].keys() if k.startswith("demo_")],
        key=lambda x: int(x.split("_")[-1]),
    )[:MAX_DEMOS]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    out_f = h5py.File(tmp_path, "w")

    total_frames = sum(f[f"data/{k}/obs/agentview_rgb"].shape[0] for k in demo_keys)
    print(f"  {len(demo_keys)} demos, {total_frames} frames")

    for di, demo_key in enumerate(demo_keys):
        images_ds = f[f"data/{demo_key}/obs/agentview_rgb"]
        T = images_ds.shape[0]
        lp_list = []

        for t in range(T):
            img = Image.fromarray(images_ds[t])
            inputs = processor(text=prompt, images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states

            lp = torch.stack([hs[li][0, -1, :] for li in range(N_LAYERS)])  # (33, 3072)
            lp_list.append(lp.cpu().half())

            del out, hs, inputs
            if t % 20 == 0:
                torch.cuda.empty_cache()

        lp_arr = torch.stack(lp_list).numpy()  # (T, 33, 3072)
        g = out_f.create_group(demo_key)
        g.create_dataset("last_preaction", data=lp_arr, dtype="float16",
                         compression="gzip", compression_opts=4)
        out_f.flush()
        del lp_list, lp_arr

        if (di + 1) % 10 == 0 or di == len(demo_keys) - 1:
            print(f"    demo {di+1}/{len(demo_keys)}")

    f.close()
    out_f.close()
    os.rename(tmp_path, out_path)
    sz = os.path.getsize(out_path) / 1024**3
    print(f"  Saved {out_path} ({sz:.2f} GB)")
    gc.collect()
    torch.cuda.empty_cache()


def run_extraction(device):
    hdf5_files = sorted(Path(DATA_DIR).glob("*_demo.hdf5"))
    print(f"Found {len(hdf5_files)} tasks in {DATA_DIR}")

    tasks_to_do = []
    for hp in hdf5_files:
        task_id = hp.stem.replace("_demo", "")
        out_path = os.path.join(FEATURE_DIR, f"{task_id}.h5")
        if os.path.exists(out_path):
            print(f"  Skip {task_id} (exists)")
        else:
            tasks_to_do.append((hp, task_id, out_path))

    if not tasks_to_do:
        print("All features already extracted, skipping Phase 1.")
        return

    print(f"\nLoading untrained hybrid model for feature extraction...")
    model, processor = load_untrained_model(device)

    for i, (hp, task_id, out_path) in enumerate(tasks_to_do):
        task_desc = task_desc_from_filename(hp.name)
        print(f"\n[{i+1}/{len(tasks_to_do)}] {task_desc}")
        t0 = time.time()
        extract_one_task(model, processor, hp, task_desc, device, out_path)
        print(f"  Took {time.time()-t0:.0f}s")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 1 complete.")


# ── Phase 2: R² Probe ──────────────────────────────────────────────────

class ActionMLPProbe(nn.Module):
    def __init__(self, in_dim=HIDDEN_DIM_MODEL, out_dim=ACTION_DIM, hidden_dim=PROBE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_features_and_actions(layer):
    task_data = {}
    for h5_path in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.h5"))):
        task = Path(h5_path).stem
        action_path = os.path.join(DATA_DIR, f"{task}_demo.hdf5")
        if not os.path.exists(action_path):
            continue
        demos = []
        with h5py.File(h5_path, "r") as ff, h5py.File(action_path, "r") as fa:
            demo_keys = sorted(
                [k for k in ff.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            for k in demo_keys:
                idx = int(k.split("_")[-1])
                feat = np.asarray(ff[k]["last_preaction"][:, layer, :], dtype=np.float32)
                act = np.asarray(fa[f"data/demo_{idx}/actions"], dtype=np.float32)
                demos.append((feat, act))
        task_data[task] = demos
    return task_data


def build_train_val(task_data):
    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list, val_tasks = [], [], []
    for task in sorted(task_data.keys()):
        demos = task_data[task]
        n_train = int(len(demos) * 0.8)
        for i, (feat, act) in enumerate(demos):
            T = min(feat.shape[0], act.shape[0])
            if T < 2:
                continue
            x = torch.from_numpy(feat[:T - 1])
            y = torch.from_numpy(act[1:T])
            if i < n_train:
                X_tr_list.append(x)
                y_tr_list.append(y)
            else:
                X_val_list.append(x)
                y_val_list.append(y)
                val_tasks.extend([task] * (T - 1))
    return (torch.cat(X_tr_list), torch.cat(y_tr_list),
            torch.cat(X_val_list), torch.cat(y_val_list), val_tasks)


def train_and_eval(X_tr, y_tr_raw, X_val, y_val_raw, val_tasks, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_mean = y_tr_raw.mean(dim=0)
    y_std = y_tr_raw.std(dim=0).clamp(min=1e-6)
    y_tr = (y_tr_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    X_tr_d = X_tr.to(device)
    y_tr_d = y_tr.to(device)
    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    model = ActionMLPProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = len(X_tr_d)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, PROBE_BATCH):
            idx = perm[start:start + PROBE_BATCH]
            pred = model(X_tr_d[idx])
            loss = criterion(pred, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_d)
            val_loss = criterion(val_pred, y_val_d).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_val_d).cpu()
    pred_orig = (pred_norm * y_std + y_mean).numpy()
    y_true = y_val_raw.numpy()

    unique_tasks = sorted(set(val_tasks))
    per_task_r2 = {}
    for t in unique_tasks:
        mask = [i for i, vt in enumerate(val_tasks) if vt == t]
        yt = y_true[mask]
        yp = pred_orig[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean(axis=0)) ** 2)
        per_task_r2[t] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0

    return float(np.mean(list(per_task_r2.values())))


def run_probe(device):
    results = {"untrained": {}}

    for layer in range(N_LAYERS):
        t0 = time.time()
        task_data = load_features_and_actions(layer)
        X_tr, y_tr, X_val, y_val, val_tasks = build_train_val(task_data)

        seed_r2s = []
        for seed in SEEDS:
            r2 = train_and_eval(X_tr, y_tr, X_val, y_val, val_tasks, device, seed)
            seed_r2s.append(r2)
            torch.cuda.empty_cache()

        results["untrained"][f"L{layer}"] = {
            "seeds": seed_r2s,
            "mean": float(np.mean(seed_r2s)),
            "std": float(np.std(seed_r2s)),
        }

        elapsed = time.time() - t0
        print(f"  L{layer}: mean={np.mean(seed_r2s):.4f} +/- {np.std(seed_r2s):.4f} "
              f"({[round(v, 4) for v in seed_r2s]}) {elapsed:.1f}s")

        del task_data, X_tr, y_tr, X_val, y_val

    return results


def main():
    device = torch.device(DEVICE)
    t_start = time.time()

    print("=" * 60)
    print("Phi3V UNTRAINED (base Phi-3 LLM) Per-Layer Action R² Probe")
    print(f"TracVLA arch: {TRACVLA_PATH}")
    print(f"Base LLM: Phi-3-mini-4k-instruct")
    print(f"Seeds: {SEEDS}")
    print(f"Layers: {N_LAYERS} (L0-L32)")
    print(f"Device: {device}")
    print(f"Feature: last_preaction (last token hidden state)")
    print(f"Split: sequential 80/20 (no shuffle)")
    print(f"R² space: original (denormalized)")
    print("=" * 60)

    print("\n--- Phase 1: Feature Extraction ---")
    run_extraction(DEVICE)

    print("\n--- Phase 2: R² Probe ---")
    results = run_probe(device)

    base_path = find_base_phi3()
    output = {
        "model": "Phi3V-Untrained-Base",
        "tracvla_checkpoint": TRACVLA_PATH,
        "base_llm": base_path,
        "conditions": ["untrained"],
        "seeds": SEEDS,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM_MODEL,
        "feature": "last_preaction",
        "split": "sequential_80_20_no_shuffle",
        "r2_space": "original_denormalized",
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {OUT_JSON}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
