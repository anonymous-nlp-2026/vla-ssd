"""shuffled_token_control.py v2 - Self-consistent: extract ALL features from the
same checkpoint, train probe, then evaluate shuffled.

If shuffled accuracy drops to chance (~10%): token ORDER creates separable representations.
If shuffled accuracy stays ~100%: token SET (bag-of-tokens) is sufficient.
"""

import os
import json
import random
import glob
import gc
import time
from collections import defaultdict
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH = "./checkpoints/openvla-7b-untrained/"
DATA_DIR = "./data/libero/libero_goal/"
OUTPUT_PATH = "./results/shuffled_token_control.json"
DEVICE = "cuda:0"
TARGET_LAYER = 1
NUM_SHUFFLES = 5
SEED = 42
HIDDEN_DIM = 4096
BATCH_SIZE = 8


class GoalClassificationProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(X_tr, y_tr, X_val, y_val, in_dim, n_classes, device,
                lr=1e-3, epochs=50, patience=5, batch_size=256):
    model = GoalClassificationProbe(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    best_acc, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for s in range(0, len(X_tr), batch_size):
            idx = perm[s : s + batch_size]
            loss = crit(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def shuffle_tokens(prompt, tokenizer, rng):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    shuffled = list(token_ids)
    rng.shuffle(shuffled)
    return tokenizer.decode(shuffled, skip_special_tokens=False)


def extract_l1_features_batch(vla_model, processor, images, prompt, device):
    """Extract layer 1 hidden state at last position for a batch of images."""
    all_feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i : i + BATCH_SIZE]
        B = len(batch_imgs)
        inputs = processor(
            text=[prompt] * B, images=batch_imgs,
            padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = vla_model(**inputs, output_hidden_states=True)

        hs = out.hidden_states[TARGET_LAYER]
        feats = hs[:, -1, :].float().cpu()
        all_feats.append(feats)
        del out, inputs
    torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0)


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device(DEVICE)

    # Phase 1: Load model
    print("Phase 1: Load untrained VLA model")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    vla = vla.to(device).eval()
    print(f"  Model loaded ({time.time() - t_start:.0f}s)")

    # Phase 2: Load t=0 images and extract original features
    print("\nPhase 2: Load t=0 images & extract original features")
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    task_names = []
    task_prompts = []

    # task_data[task_idx] = list of (demo_key, PIL.Image)
    task_data = []

    for hdf5_path in hdf5_files:
        stem = Path(hdf5_path).stem.replace("_demo", "")
        task_names.append(stem)
        task_desc = stem.replace("_", " ")
        prompt = f"In: What action should the robot take to {task_desc}?\nOut:"
        task_prompts.append(prompt)

        demos = []
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            for dk in demo_keys[:50]:
                img_arr = f[f"data/{dk}"]["obs"]["agentview_rgb"][0]
                demos.append((dk, Image.fromarray(img_arr)))
        task_data.append(demos)

    n_classes = len(task_names)
    print(f"  {n_classes} tasks, {[len(d) for d in task_data]} demos each")

    # Extract original features for all tasks
    all_features = []  # list of (task_idx, demo_key, feature_tensor)
    for task_idx in range(n_classes):
        images = [img for _, img in task_data[task_idx]]
        prompt = task_prompts[task_idx]
        feats = extract_l1_features_batch(vla, processor, images, prompt, device)
        for j, (dk, _) in enumerate(task_data[task_idx]):
            all_features.append((task_idx, dk, feats[j]))
        print(f"  Task {task_idx}: {task_names[task_idx]} ({len(images)} demos)")

    print(f"  Total: {len(all_features)} samples extracted ({time.time()-t_start:.0f}s)")

    # Phase 3: Split 80/20, train probe
    print("\nPhase 3: Train probe on original features")
    by_task = defaultdict(list)
    for i, (task_idx, dk, feat) in enumerate(all_features):
        by_task[task_idx].append(i)

    train_ids, val_ids = [], []
    for task_idx in sorted(by_task.keys()):
        ids = by_task[task_idx]
        n_train = int(len(ids) * 0.8)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:])

    X_tr = torch.stack([all_features[i][2] for i in train_ids])
    y_tr = torch.tensor([all_features[i][0] for i in train_ids], dtype=torch.long)
    X_val = torch.stack([all_features[i][2] for i in val_ids])
    y_val = torch.tensor([all_features[i][0] for i in val_ids], dtype=torch.long)

    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val")
    probe, probe_val_acc = train_probe(
        X_tr, y_tr, X_val, y_val, HIDDEN_DIM, n_classes, device
    )
    print(f"  Probe val accuracy: {probe_val_acc:.4f}")

    # Per-task accuracy on val (original)
    probe.eval()
    with torch.no_grad():
        val_preds_orig = probe(X_val.to(device)).argmax(1).cpu().numpy()
    val_labels = y_val.numpy()
    per_task_orig = {}
    for c in range(n_classes):
        mask = val_labels == c
        if mask.sum() > 0:
            per_task_orig[task_names[c]] = float((val_preds_orig[mask] == c).mean())

    # Phase 4: Evaluate with shuffled instructions
    print("\nPhase 4: Evaluate shuffled instructions")
    shuffle_run_results = []

    for shuf_idx in range(1, NUM_SHUFFLES + 1):
        rng = random.Random(SEED + shuf_idx)
        all_feats_shuf = []
        all_labels_shuf = []

        for task_idx in range(n_classes):
            prompt = shuffle_tokens(task_prompts[task_idx], processor.tokenizer, rng)

            # Only val demos
            task_val = [(i, all_features[i][1]) for i in val_ids
                       if all_features[i][0] == task_idx]
            if not task_val:
                continue

            images = [dict(task_data[task_idx])[dk] for _, dk in task_val]
            feats = extract_l1_features_batch(vla, processor, images, prompt, device)

            for f_vec in feats:
                all_feats_shuf.append(f_vec)
                all_labels_shuf.append(task_idx)

        X_shuf = torch.stack(all_feats_shuf).to(device)
        y_shuf = torch.tensor(all_labels_shuf, dtype=torch.long)

        probe.eval()
        with torch.no_grad():
            preds = probe(X_shuf).argmax(1).cpu().numpy()
        y_np = y_shuf.numpy()
        acc = float((preds == y_np).mean())

        per_task_shuf = {}
        for c in range(n_classes):
            mask = y_np == c
            if mask.sum() > 0:
                per_task_shuf[task_names[c]] = float((preds[mask] == c).mean())

        print(f"  shuffle_{shuf_idx}: acc={acc:.4f}")
        shuffle_run_results.append({
            "condition": f"shuffle_{shuf_idx}",
            "accuracy": acc,
            "per_task_accuracy": per_task_shuf,
        })

    # Summary
    shuffle_accs = [r["accuracy"] for r in shuffle_run_results]

    results = {
        "experiment": "shuffled_token_control",
        "model": "openvla-7b-untrained",
        "layer": TARGET_LAYER,
        "num_shuffles": NUM_SHUFFLES,
        "num_train_samples": len(train_ids),
        "num_val_samples": len(val_ids),
        "num_classes": n_classes,
        "original_probe_val_accuracy": probe_val_acc,
        "original_per_task_accuracy": per_task_orig,
        "shuffled_accuracy_mean": float(np.mean(shuffle_accs)),
        "shuffled_accuracy_std": float(np.std(shuffle_accs)),
        "shuffled_accuracy_individual": shuffle_accs,
        "shuffle_run_details": shuffle_run_results,
        "task_names": task_names,
        "elapsed_seconds": time.time() - t_start,
    }

    if results["shuffled_accuracy_mean"] < 0.2:
        results["conclusion"] = (
            "Token ORDER matters: shuffling destroys separability"
        )
    elif results["shuffled_accuracy_mean"] > 0.8:
        results["conclusion"] = (
            "Token SET sufficient: bag-of-tokens already separable in high-dim space"
        )
    else:
        results["conclusion"] = (
            f"Partial effect: accuracy {results['shuffled_accuracy_mean']:.1%}"
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Original (probe val accuracy):  {probe_val_acc:.4f}")
    print(
        f"Shuffled ({NUM_SHUFFLES}x mean+/-std): "
        f"{results['shuffled_accuracy_mean']:.4f} +/- {results['shuffled_accuracy_std']:.4f}"
    )
    print(f"Individual: {shuffle_accs}")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Total time: {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
