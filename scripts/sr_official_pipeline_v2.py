"""Path B final test: official checkpoint closed-loop SR with all 6 bug fixes."""

import os, sys, time
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
N_EPISODES = 5
TASK_IDS = [0, 3, 7]

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero
from transformers import AutoModelForVision2Seq, AutoProcessor

def resize_image_no_jpeg(img, resize_size):
    """Lanczos resize WITHOUT JPEG roundtrip."""
    assert isinstance(resize_size, tuple)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((resize_size[1], resize_size[0]), Image.LANCZOS)
    return np.array(pil_img)

def get_libero_image(obs, resize_size):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180-degree rotation
    img = resize_image_no_jpeg(img, resize_size)
    return img

def crop_and_resize(image_np, crop_scale, target_size=224):
    h, w = image_np.shape[:2]
    crop_h = int(h * np.sqrt(crop_scale))
    crop_w = int(w * np.sqrt(crop_scale))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = image_np[top:top+crop_h, left:left+crop_w]
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_img)

def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action

def get_vla_action(model, processor, obs_image, task_label, unnorm_key, center_crop=True):
    if center_crop:
        obs_image = crop_and_resize(obs_image, crop_scale=0.9, target_size=224)
    image = Image.fromarray(obs_image).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=True)
    return action

# --- Main ---
LIB_DIR = os.path.dirname(libero.libero.__file__)

print("Loading model...", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda:0"
).eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

bench = benchmark.get_benchmark("libero_goal")()
resize_size = 224

all_results = {}
total_success = 0
total_episodes = 0

for task_id in TASK_IDS:
    task = bench.get_task(task_id)
    instruction = task.language
    print(f"\nTask {task_id}: {task.name}", flush=True)
    print(f"  Instruction: {instruction}", flush=True)

    episodes = []
    for ep in range(N_EPISODES):
        bddl_path = os.path.join(LIB_DIR, "bddl_files", task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, camera_names="agentview")
        seed = task_id * 100 + ep
        env.seed(seed)
        obs = env.reset()

        # NO wait steps

        cumulative_reward = 0.0
        success = False
        ep_t0 = time.time()

        for step_i in range(MAX_STEPS):
            img = get_libero_image(obs, resize_size)

            with torch.no_grad():
                action = get_vla_action(model, processor, img, instruction, UNNORM_KEY, center_crop=True)

            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            if step_i < 5 and ep == 0:
                print(f"  step {step_i}: action={np.round(action, 3).tolist()}", flush=True)

            obs, reward, done, info = env.step(action.tolist())
            cumulative_reward += reward
            if done:
                success = True
                break

        ep_time = time.time() - ep_t0
        status = "SUCCESS" if success else "fail"
        print(f"  ep {ep} (seed={seed}): {status} | R={cumulative_reward:.3f} | steps={step_i+1} | {ep_time:.1f}s", flush=True)
        episodes.append({"success": success, "reward": float(cumulative_reward), "steps": step_i + 1, "seed": seed})
        env.close()

    sr = sum(e["success"] for e in episodes) / N_EPISODES
    print(f"  Task {task_id} SR={sr:.2f} ({sum(e['success'] for e in episodes)}/{N_EPISODES})", flush=True)
    all_results[task_id] = {"sr": sr, "episodes": episodes}
    total_success += sum(e["success"] for e in episodes)
    total_episodes += N_EPISODES

overall_sr = total_success / total_episodes
print(f"\n{'='*60}", flush=True)
print(f"OVERALL SR = {overall_sr:.2f} ({total_success}/{total_episodes})", flush=True)
pts = ", ".join("T" + str(tid) + "=" + "{:.2f}".format(all_results[tid]["sr"]) for tid in TASK_IDS)
print("Per-task: " + pts, flush=True)
print("DONE", flush=True)
