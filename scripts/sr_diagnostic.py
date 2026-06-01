"""Diagnostic: compare native predict_action vs manual, with/without image flip."""

import os, sys
sys.path.insert(0, "./data/LIBERO")
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "./checkpoints/openvla-7b-finetuned-libero-goal"
UNNORM_KEY = "libero_goal"

# 1. Create env, get observation
print("=== Phase 1: Environment ===", flush=True)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero
LIB_DIR = os.path.dirname(libero.libero.__file__)
bench = benchmark.get_benchmark("libero_goal")()
task = bench.get_task(0)
bddl_path = os.path.join(LIB_DIR, "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224, camera_names="agentview")
obs = env.reset()
raw_image = obs["agentview_image"]
print(f"Image shape: {raw_image.shape}, dtype: {raw_image.dtype}", flush=True)
print(f"Image value range: [{raw_image.min()}, {raw_image.max()}]", flush=True)
print(f"Instruction: {task.language}", flush=True)

# Also get a 256x256 env for comparison
env2 = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, camera_names="agentview")
obs2 = env2.reset()
raw_image_256 = obs2["agentview_image"]
env2.close()

# Get second observation for comparison
obs_step = env.step(np.zeros(7))
raw_image2 = obs_step[0]["agentview_image"]
env.close()

# 2. Load model
print("\n=== Phase 2: Model ===", flush=True)
from transformers import AutoModelForVision2Seq, AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda:0"
).eval()
print(f"model.vocab_size = {model.vocab_size}", flush=True)
print(f"model.bin_centers shape = {model.bin_centers.shape}", flush=True)
print(f"model.bin_centers[:5] = {model.bin_centers[:5]}", flush=True)
print(f"model.bin_centers[-5:] = {model.bin_centers[-5:]}", flush=True)

instruction = task.language
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

def center_crop_90(img):
    w, h = img.size
    new_w, new_h = int(w * 0.9), int(h * 0.9)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

# 3. Test all combinations
print("\n=== Phase 3: Action Predictions ===", flush=True)

test_configs = [
    ("224_no_crop_no_flip", raw_image, False, False),
    ("224_crop90_no_flip", raw_image, True, False),
    ("224_no_crop_flip", raw_image, False, True),
    ("224_crop90_flip", raw_image, True, True),
    ("256_no_crop_flip", raw_image_256, False, True),
    ("256_crop_to_224_flip", raw_image_256, "crop224", True),
    ("224_obs2_no_crop_no_flip", raw_image2, False, False),
]

for name, img_arr, crop, flip in test_configs:
    if flip:
        img_arr_use = img_arr[::-1].copy()
    else:
        img_arr_use = img_arr

    img = Image.fromarray(img_arr_use)

    if crop == "crop224":
        # Center crop 256 to 224
        w, h = img.size
        left = (w - 224) // 2
        top = (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
    elif crop:
        img = center_crop_90(img)

    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)

    # Native predict_action
    with torch.no_grad():
        action_native = model.predict_action(inputs["input_ids"], unnorm_key=UNNORM_KEY, pixel_values=inputs["pixel_values"])

    # Also get raw generated token IDs via generate
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)), dim=1
            )
        gen_ids = model.generate(input_ids, max_new_tokens=7, pixel_values=inputs["pixel_values"], do_sample=False)
        token_ids = gen_ids[0, -7:].cpu().numpy()

    print(f"\n  [{name}]", flush=True)
    print(f"    action = {np.round(action_native, 4).tolist()}", flush=True)
    print(f"    token_ids = {token_ids.tolist()}", flush=True)
    print(f"    img_size_before_proc = {img.size}", flush=True)

print("\n=== Phase 4: Check image orientation ===", flush=True)
print(f"raw_image[0,0] (top-left pixel) = {raw_image[0,0]}", flush=True)
print(f"raw_image[-1,0] (bottom-left pixel) = {raw_image[-1,0]}", flush=True)
print(f"raw_image[0,:,:].mean() (top row mean) = {raw_image[0].mean():.1f}", flush=True)
print(f"raw_image[-1,:,:].mean() (bottom row mean) = {raw_image[-1].mean():.1f}", flush=True)

# Check if top and bottom are very different (indicating possible flip needed)
top_half_mean = raw_image[:112].mean()
bot_half_mean = raw_image[112:].mean()
print(f"Top half mean: {top_half_mean:.1f}, Bottom half mean: {bot_half_mean:.1f}", flush=True)

print("\nDONE", flush=True)
