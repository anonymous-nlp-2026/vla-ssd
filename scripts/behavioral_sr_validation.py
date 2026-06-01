"""Quick SR validation: 1 task, 3 episodes, baseline only."""

import os
import sys
sys.path.insert(0, "./data/LIBERO")

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import json
import time

import numpy as np
import torch
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
UNNORM_KEY = "libero_goal"
TASK_ID = 0
N_EPISODES = 3
MAX_STEPS = 300

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero

LIB_DIR = os.path.dirname(libero.libero.__file__)

# Pre-init MuJoCo EGL before model loading (device_map changes CUDA context)
print("Pre-initializing MuJoCo EGL...", flush=True)
_bench = benchmark.get_benchmark("libero_goal")()
_task = _bench.get_task(0)
_bddl = os.path.join(LIB_DIR, "bddl_files", _task.problem_folder, _task.bddl_file)
_env = OffScreenRenderEnv(bddl_file_name=_bddl, camera_heights=224, camera_widths=224, camera_names="agentview")
_env.reset()
_env.close()
del _env, _bench, _task, _bddl
print("MuJoCo EGL initialized.", flush=True)


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
    print("Loading model...", flush=True)
    t0 = time.time()
    model, processor = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    bench = benchmark.get_benchmark("libero_goal")()
    task = bench.get_task(TASK_ID)
    task_name = task.name
    task_lang = task.language
    print(f"\nTask {TASK_ID}: {task_name}", flush=True)
    print(f"  Instruction: {task_lang}", flush=True)

    action_samples = []
    episodes = []

    for ep in range(N_EPISODES):
        ep_t0 = time.time()
        env, _, _ = make_env(TASK_ID)
        np.random.seed(42 + ep)
        obs = env.reset()

        cumulative_reward = 0.0
        success = False

        for step_i in range(MAX_STEPS):
            image = obs["agentview_image"]
            action = get_action(model, processor, image, task_lang)
            if step_i < 5 and ep == 0:
                action_samples.append(action.tolist())
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
        print(f"  ep {ep}: {status} | R={cumulative_reward:.3f} | steps={step_i+1} | {ep_time:.1f}s", flush=True)

    sr = sum(e["success"] for e in episodes) / len(episodes)
    avg_r = np.mean([e["reward"] for e in episodes])
    print(f"\n=== RESULT: SR={sr:.2f} ({sum(e['success'] for e in episodes)}/{N_EPISODES}), avg_reward={avg_r:.3f} ===", flush=True)
    print(f"Action samples (ep0, first 5 steps): {json.dumps(action_samples, indent=2)}", flush=True)

    if sr > 0:
        print("\n>>> PASS: SR > 0, proceed to Phase 2", flush=True)
    else:
        print("\n>>> FAIL: SR = 0, do NOT proceed to Phase 2", flush=True)
        print("Diagnostics:", flush=True)
        print(f"  Action range: min={np.min(action_samples):.4f}, max={np.max(action_samples):.4f}", flush=True)
        print(f"  Reward trajectory: {[e['reward'] for e in episodes]}", flush=True)


if __name__ == "__main__":
    main()
