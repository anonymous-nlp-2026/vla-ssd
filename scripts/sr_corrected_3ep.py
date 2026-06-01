"""Corrected smoketest: 3 episodes with proper preprocessing (256x256 + center crop to 224 + flip)."""

import os, sys, json, time
sys.path.insert(0, "./data/LIBERO")
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
UNNORM_KEY = "libero_goal"
MAX_STEPS = 400
TASK_ID = 0
N_EPISODES = 3

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero
LIB_DIR = os.path.dirname(libero.libero.__file__)
from transformers import AutoModelForVision2Seq, AutoProcessor

print("Loading model...", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda:0"
).eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

bench = benchmark.get_benchmark("libero_goal")()
task = bench.get_task(TASK_ID)
instruction = task.language
prompt = f"In: What action should the robot take to {instruction}?\nOut:"
print(f"Task: {task.name}", flush=True)
print(f"Instruction: {instruction}", flush=True)

episodes = []
for ep in range(N_EPISODES):
    bddl_path = os.path.join(LIB_DIR, "bddl_files", task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_names="agentview",
    )
    np.random.seed(42 + ep)
    obs = env.reset()

    cumulative_reward = 0.0
    success = False
    ep_t0 = time.time()

    for step_i in range(MAX_STEPS):
        raw_img = obs["agentview_image"][::-1].copy()
        img = Image.fromarray(raw_img)
        # Center crop 256 -> 224
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            action = model.predict_action(inputs["input_ids"], unnorm_key=UNNORM_KEY, pixel_values=inputs["pixel_values"])

        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if env.env._check_success():
            success = True
            break

    ep_time = time.time() - ep_t0
    status = "SUCCESS" if success else "fail"
    print(f"  ep {ep}: {status} | R={cumulative_reward:.3f} | steps={step_i+1} | {ep_time:.1f}s", flush=True)
    episodes.append({"success": success, "reward": float(cumulative_reward), "steps": step_i + 1})
    env.close()
    torch.cuda.empty_cache()

sr = sum(e["success"] for e in episodes) / N_EPISODES
print(f"\n=== CORRECTED: SR={sr:.2f} ({sum(e['success'] for e in episodes)}/{N_EPISODES}) ===", flush=True)

result = {
    "config": "256x256_crop224_flip_native_predict",
    "task": task.name,
    "instruction": instruction,
    "episodes": episodes,
    "sr": sr,
}
out_path = "./results/sr_corrected_3ep.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {out_path}", flush=True)
