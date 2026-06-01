"""Quick 1-episode test with corrected preprocessing: 256x256 camera, center crop to 224, image flip."""

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

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero
LIB_DIR = os.path.dirname(libero.libero.__file__)

from transformers import AutoModelForVision2Seq, AutoProcessor

print("Loading model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda:0"
).eval()

bench = benchmark.get_benchmark("libero_goal")()
task = bench.get_task(TASK_ID)
instruction = task.language
prompt = f"In: What action should the robot take to {instruction}?\nOut:"
print(f"Task: {task.name}", flush=True)
print(f"Instruction: {instruction}", flush=True)

# Test 3 configs: (A) 256+crop+flip, (B) 224+flip (no crop), (C) original 224+crop90+no_flip
configs = {
    "A_256_crop224_flip": {"cam_size": 256, "flip": True, "crop": "center_to_224"},
    "B_224_nocrp_flip": {"cam_size": 224, "flip": True, "crop": None},
    "C_orig_224_crop90_noflip": {"cam_size": 224, "flip": False, "crop": "center_crop_90"},
}

for config_name, cfg in configs.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name}", flush=True)
    print(f"  cam={cfg['cam_size']}, flip={cfg['flip']}, crop={cfg['crop']}", flush=True)
    print(f"{'='*60}", flush=True)

    bddl_path = os.path.join(LIB_DIR, "bddl_files", task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cfg["cam_size"],
        camera_widths=cfg["cam_size"],
        camera_names="agentview",
    )
    np.random.seed(42)
    obs = env.reset()

    cumulative_reward = 0.0
    success = False
    action_log = []

    t0 = time.time()
    for step_i in range(MAX_STEPS):
        raw_img = obs["agentview_image"]
        if cfg["flip"]:
            raw_img = raw_img[::-1].copy()

        img = Image.fromarray(raw_img)

        if cfg["crop"] == "center_to_224" and cfg["cam_size"] == 256:
            left = (256 - 224) // 2
            top = (256 - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
        elif cfg["crop"] == "center_crop_90":
            w, h = img.size
            new_w, new_h = int(w * 0.9), int(h * 0.9)
            left, top = (w - new_w) // 2, (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))

        inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            action = model.predict_action(inputs["input_ids"], unnorm_key=UNNORM_KEY, pixel_values=inputs["pixel_values"])

        if step_i < 10:
            action_log.append(action.tolist())
            print(f"  step {step_i}: action={np.round(action, 3).tolist()}", flush=True)

        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if env.env._check_success():
            success = True
            print(f"  >>> SUCCESS at step {step_i+1}!", flush=True)
            break

    elapsed = time.time() - t0
    status = "SUCCESS" if success else "fail"
    print(f"\n  Result: {status} | R={cumulative_reward:.3f} | steps={step_i+1} | {elapsed:.1f}s", flush=True)

    env.close()
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
