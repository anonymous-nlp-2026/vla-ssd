"""Behavioral smoke test: 1 task x 3 episodes x baseline only."""

import os
import sys
sys.path.insert(0, "./data/LIBERO")

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
UNNORM_KEY = "libero_goal"
TASK_IDS = [0]
N_EPISODES = 3
MAX_STEPS = 300
RESULTS_DIR = "./results"

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero

LIB_DIR = os.path.dirname(libero.libero.__file__)


def make_env(task_id):
    bench = benchmark.get_benchmark("libero_goal")()
    task = bench.get_task(task_id)
    bddl_path = os.path.join(LIB_DIR, "bddl_files", task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=224,
        camera_widths=224,
        camera_names="agentview",
    )
    return env, task.language, task.name


def predict_action_manual(model, input_ids, pixel_values, unnorm_key):
    action_dim = model.get_action_dim(unnorm_key)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)),
            dim=1,
        )
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values,
            past_key_values=None,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    generated_ids = []
    for _ in range(action_dim):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.squeeze().item())
        attn_mask = torch.ones(
            (1, past_key_values[0][0].shape[2] + 1),
            device=input_ids.device, dtype=input_ids.dtype,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                pixel_values=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    predicted_action_token_ids = np.array(generated_ids)
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="cuda:0"
    ).eval()
    return model, processor


def center_crop_90(img):
    w, h = img.size
    new_w, new_h = int(w * 0.9), int(h * 0.9)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def get_action(model, processor, image_array, instruction):
    img = Image.fromarray(image_array)
    img = center_crop_90(img)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    return predict_action_manual(model, inputs["input_ids"], inputs["pixel_values"], UNNORM_KEY)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading model...", flush=True)
    t0 = time.time()
    model, processor = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    bench = benchmark.get_benchmark("libero_goal")()
    all_results = {}

    for task_id in TASK_IDS:
        task = bench.get_task(task_id)
        task_name = task.name
        task_lang = task.language
        print(f"\n{'='*60}", flush=True)
        print(f"Task {task_id}: {task_name}", flush=True)
        print(f"  Instruction: {task_lang}", flush=True)
        print(f"{'='*60}", flush=True)

        episodes = []
        for ep in range(N_EPISODES):
            ep_t0 = time.time()
            env, _, _ = make_env(task_id)
            np.random.seed(42 + ep)
            obs = env.reset()

            cumulative_reward = 0.0
            success = False

            for step_i in range(MAX_STEPS):
                image = obs["agentview_image"]
                action = get_action(model, processor, image, task_lang)
                obs, reward, done, info = env.step(action)
                cumulative_reward += reward
                if env.env._check_success():
                    success = True
                    break

            ep_result = {
                "success": bool(success),
                "reward": float(cumulative_reward),
                "steps": step_i + 1,
            }
            episodes.append(ep_result)
            env.close()

            ep_time = time.time() - ep_t0
            status = "SUCCESS" if success else "fail"
            print(f"    ep {ep:2d}: {status} | R={cumulative_reward:.3f} | steps={step_i+1} | {ep_time:.1f}s", flush=True)

        sr = sum(e["success"] for e in episodes) / len(episodes)
        avg_r = np.mean([e["reward"] for e in episodes])
        print(f"  >> baseline SR={sr:.2f}, avg_reward={avg_r:.3f}", flush=True)

        all_results[task_name] = {"baseline": episodes}

    total_success = sum(e["success"] for t in all_results.values() for e in t["baseline"])
    total_eps = sum(len(t["baseline"]) for t in all_results.values())
    summary = {
        "baseline_sr": total_success / total_eps,
        "n_tasks": len(TASK_IDS),
        "n_episodes": N_EPISODES,
        "task_ids": TASK_IDS,
    }

    output = {
        "config": {
            "checkpoint": "openvla-7b-finetuned-libero-goal",
            "unnorm_key": UNNORM_KEY,
            "n_tasks": len(TASK_IDS),
            "n_episodes": N_EPISODES,
            "max_steps": MAX_STEPS,
            "task_ids": TASK_IDS,
            "model_path": MODEL_PATH,
        },
        "results": all_results,
        "summary": summary,
    }

    out_path = os.path.join(RESULTS_DIR, "behavioral_smoketest.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
