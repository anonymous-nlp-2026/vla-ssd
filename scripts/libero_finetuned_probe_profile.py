"""Cross-instruction probe profile for openvla-7b-finetuned-libero-goal.

Phase 1: Extract hidden states from all 32 layers (10 tasks x 50 demos x 10 instructions).
Phase 2: LOOCV 10-way instruction classification (PCA256 + LogReg) at each layer.
"""

import gc
import glob
import json
import os
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

NUM_IMAGE_TOKENS = 256
NUM_LAYERS = 32
FINETUNED_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal/"
FEATURES_DIR = "./results/cross_instruction_finetuned/"
FEATURES_PATH = os.path.join(FEATURES_DIR, "finetuned_cross_inst_L0-L31.h5")
RESULT_PATH = "./results/libero_finetuned_probe_profile.json"
GPU = 0
FRAMES_PER_DEMO = 1
MAX_DEMOS = 50
N_TASKS = 10
N_INSTRUCTIONS = 10
PCA_COMPONENTS = 256
SEED = 42
TOKEN_POSITIONS = ["last_preaction", "image_mean"]


def task_desc_from_filename(fname):
    stem = Path(fname).stem.replace("_demo", "")
    return stem.replace("_", " ")


def sample_frame_indices(demo_length, frames_per_demo):
    if frames_per_demo == 1:
        return [demo_length // 2]
    return np.linspace(0, demo_length - 1, frames_per_demo, dtype=int).tolist()


def load_model(device_str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        FINETUNED_PATH, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        FINETUNED_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
        device_map=device_str,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.1f}B params, device_map={device_str}")
    return model, processor


@torch.no_grad()
def extract_features_single(model, processor, image, instruction, device, layer_indices):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    image_mean = torch.stack(
        [hs[li][:, 1:1+NUM_IMAGE_TOKENS, :].mean(dim=1).squeeze(0) for li in layer_indices]
    )
    last_preaction = torch.stack(
        [hs[li][:, -1, :].squeeze(0) for li in layer_indices]
    )

    image_mean = image_mean.cpu().half().numpy()
    last_preaction = last_preaction.cpu().half().numpy()

    del out, hs, inputs
    return image_mean, last_preaction


def extract_features():
    device = torch.device(f"cuda:{GPU}")
    device_str = f"cuda:{GPU}"
    layer_indices = list(range(NUM_LAYERS))

    print(f"=== Phase 1: Feature Extraction (L0-L31) ===")
    print(f"Model: {FINETUNED_PATH}")
    print(f"Output: {FEATURES_PATH}")

    if os.path.exists(FEATURES_PATH):
        with h5py.File(FEATURES_PATH, "r") as check_f:
            n_tasks_done = sum(1 for k in check_f.keys() if k.startswith("task_"))
        if n_tasks_done >= N_TASKS:
            print(f"Features already exist ({n_tasks_done} tasks), skipping extraction.")
            return
        else:
            print(f"Incomplete features ({n_tasks_done}/{N_TASKS} tasks), re-extracting.")
            os.remove(FEATURES_PATH)

    model, processor = load_model(device_str)

    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    assert len(hdf5_files) == N_TASKS, f"Expected {N_TASKS} task files, found {len(hdf5_files)}"
    all_instructions = [task_desc_from_filename(f) for f in hdf5_files]
    print(f"Tasks: {len(hdf5_files)}, Instructions: {len(all_instructions)}")
    for i, inst in enumerate(all_instructions):
        print(f"  [{i}] {inst}")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_f = h5py.File(FEATURES_PATH, "w")
    out_f.attrs["layers"] = layer_indices

    meta = out_f.create_group("metadata")
    meta.create_dataset("instructions", data=[s.encode() for s in all_instructions])
    meta.create_dataset("layers", data=layer_indices)
    meta.create_dataset("task_files", data=[Path(f).name.encode() for f in hdf5_files])

    total_forwards = 0
    t_start = time.time()

    for task_idx, hdf5_path in enumerate(hdf5_files):
        task_name = Path(hdf5_path).stem.replace("_demo", "")
        print(f"\n[Task {task_idx}] {task_name}")

        f = h5py.File(hdf5_path, "r")
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        demo_keys = demo_keys[:MAX_DEMOS]

        task_grp = out_f.create_group(f"task_{task_idx}")
        frame_counter = 0

        for demo_idx, dk in enumerate(tqdm(demo_keys, desc=f"task_{task_idx}")):
            images_ds = data[dk]["obs"]["agentview_rgb"]
            T = images_ds.shape[0]
            frame_indices = sample_frame_indices(T, FRAMES_PER_DEMO)

            for t in frame_indices:
                img = Image.fromarray(images_ds[t])
                frame_grp = task_grp.create_group(f"frame_{frame_counter}")
                frame_grp.attrs["demo_key"] = dk
                frame_grp.attrs["timestep"] = t
                frame_grp.attrs["demo_length"] = T

                for inst_idx, instruction in enumerate(all_instructions):
                    im_mean, lp = extract_features_single(
                        model, processor, img, instruction, device, layer_indices
                    )
                    inst_grp = frame_grp.create_group(f"instruction_{inst_idx}")
                    inst_grp.create_dataset("image_mean", data=im_mean, dtype="float16")
                    inst_grp.create_dataset("last_preaction", data=lp, dtype="float16")
                    total_forwards += 1

                frame_counter += 1

        f.close()
        out_f.flush()
        elapsed = time.time() - t_start
        rate = total_forwards / elapsed if elapsed > 0 else 0
        print(f"  frames={frame_counter}, total_fwd={total_forwards}, "
              f"rate={rate:.1f}/s, elapsed={elapsed:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    out_f.close()
    total_time = time.time() - t_start
    sz = os.path.getsize(FEATURES_PATH) / 1024**2
    print(f"\nExtraction done. {total_forwards} forwards in {total_time/60:.1f} min, {sz:.1f} MB")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def run_loocv_probes():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    print(f"\n=== Phase 2: LOOCV Probe Classification ===")
    print(f"Features: {FEATURES_PATH}")

    f = h5py.File(FEATURES_PATH, "r")
    layers = list(f["metadata/layers"][:])
    layer_names = [f"L{l}" for l in layers]
    n_layers = len(layers)

    print("Loading features into memory...")
    t_load = time.time()
    data = {tp: {ln: [] for ln in layer_names} for tp in TOKEN_POSITIONS}

    for task_idx in range(N_TASKS):
        task_grp = f[f"task_{task_idx}"]
        frame_keys = sorted(task_grp.keys(), key=lambda x: int(x.split("_")[1]))

        task_samples = {tp: {ln: [] for ln in layer_names} for tp in TOKEN_POSITIONS}
        for frame_key in frame_keys:
            frame_grp = task_grp[frame_key]
            for inst_idx in range(N_INSTRUCTIONS):
                inst_grp = frame_grp[f"instruction_{inst_idx}"]
                for tp in TOKEN_POSITIONS:
                    arr = np.array(inst_grp[tp], dtype=np.float32)
                    for li, ln in enumerate(layer_names):
                        task_samples[tp][ln].append(arr[li])

        for tp in TOKEN_POSITIONS:
            for ln in layer_names:
                data[tp][ln].append(np.array(task_samples[tp][ln], dtype=np.float32))
        print(f"  Task {task_idx} loaded")

    for tp in TOKEN_POSITIONS:
        for ln in layer_names:
            data[tp][ln] = np.array(data[tp][ln])

    f.close()
    print(f"  Load time: {time.time() - t_load:.1f}s")
    print(f"  Shape per layer: {data['last_preaction'][layer_names[0]].shape}")

    labels = np.tile(np.arange(N_INSTRUCTIONS), MAX_DEMOS)

    results = {}
    t_probe = time.time()
    for tp in TOKEN_POSITIONS:
        results[tp] = {}
        print(f"\n--- {tp} ---")
        for ln in layer_names:
            fold_accs = []
            all_data = data[tp][ln]

            for test_task in range(N_TASKS):
                train_mask = [i for i in range(N_TASKS) if i != test_task]
                X_train = all_data[train_mask].reshape(-1, all_data.shape[-1])
                y_train = np.tile(labels, len(train_mask))
                X_test = all_data[test_task]
                y_test = labels

                pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
                clf.fit(X_train_pca, y_train)
                acc = clf.score(X_test_pca, y_test)
                fold_accs.append(round(float(acc), 4))

            results[tp][ln] = {
                "mean": round(float(np.mean(fold_accs)), 4),
                "std": round(float(np.std(fold_accs)), 4),
                "per_fold": fold_accs,
            }
            print(f"  {tp:15s} | {ln:3s} | "
                  f"mean={results[tp][ln]['mean']:.4f} std={results[tp][ln]['std']:.4f}")

    print(f"\nProbe time: {(time.time() - t_probe)/60:.1f} min")
    return results, layer_names


def save_results(loocv_results, layer_names):
    per_layer_accuracy = {}
    loocv_detail = {}
    for ln in layer_names:
        r = loocv_results["last_preaction"][ln]
        per_layer_accuracy[ln] = r["mean"]
        loocv_detail[ln] = {"mean": r["mean"], "std": r["std"]}

    reference = {
        "L0": {"untrained": 0.1, "llama2base": 0.1, "openvla7b": 0.1},
        "L1": {"untrained": 1.0, "llama2base": 0.802, "openvla7b": 0.996},
        "L8": {"untrained": 0.986, "llama2base": 0.999, "openvla7b": 1.0},
        "L31": {"untrained": 0.82, "llama2base": 1.0, "openvla7b": 1.0},
    }

    comparison = {}
    for ln in ["L0", "L1", "L8", "L31"]:
        ref = reference.get(ln, {})
        comparison[ln] = {
            "untrained": ref.get("untrained"),
            "llama2base": ref.get("llama2base"),
            "openvla7b": ref.get("openvla7b"),
            "libero_finetuned": per_layer_accuracy.get(ln),
        }

    output = {
        "model": "openvla-7b-finetuned-libero-goal",
        "method": "PCA256+LogReg, LOOCV (10-fold leave-one-task-out)",
        "token_position": "last_preaction",
        "per_layer_accuracy": per_layer_accuracy,
        "loocv_results": loocv_detail,
        "full_results": loocv_results,
        "comparison_4conditions": comparison,
        "metadata": {
            "n_folds": 10,
            "samples_per_task": MAX_DEMOS * N_INSTRUCTIONS,
            "pca_components": PCA_COMPONENTS,
            "classifier": "LogisticRegression(max_iter=2000, C=1.0)",
            "seed": SEED,
            "features_path": FEATURES_PATH,
        },
    }

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to {RESULT_PATH}")


def main():
    t0 = time.time()
    extract_features()
    loocv_results, layer_names = run_loocv_probes()
    save_results(loocv_results, layer_names)
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
