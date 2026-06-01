"""
LIBERO closed-loop feasibility probe for OpenVLA finetuned checkpoint.
Recreates env per trial to avoid reset issues.
"""
import os
import sys
import json
import time
import math
import io
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "./data/LIBERO")

from libero.libero.benchmark import get_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from transformers import AutoModelForVision2Seq, AutoProcessor

DEVICE = torch.device("cuda:0")
CHECKPOINT = "./checkpoints/openvla-7b-finetuned-libero-goal"
UNNORM_KEY = "libero_goal"
NUM_TRIALS = 10
MAX_STEPS = 400
NUM_WAIT_STEPS = 10
RESOLUTION = 256
RESIZE_SIZE = 224
CENTER_CROP = True
CROP_SCALE = 0.9


def resize_image_pil(img, size=(224, 224)):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    pil_img = Image.open(buf)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


def center_crop_pil(pil_img, crop_scale=0.9):
    w, h = pil_img.size
    new_h = int(h * math.sqrt(crop_scale))
    new_w = int(w * math.sqrt(crop_scale))
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = pil_img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def get_libero_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img = resize_image_pil(img, (RESIZE_SIZE, RESIZE_SIZE))
    return img


def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading model from {CHECKPOINT}")
    print(f"  center_crop={CENTER_CROP}, crop_scale={CROP_SCALE}")
    
    model = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    
    stats_path = os.path.join(CHECKPOINT, "dataset_statistics.json")
    with open(stats_path, "r") as f:
        model.norm_stats = json.load(f)
    
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    
    benchmark = get_benchmark("libero_goal")(0)
    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")
    
    results = {}
    total_success = 0
    total_episodes = 0
    
    for task_id in range(10):
        task = benchmark.get_task(task_id)
        task_description = task.language
        print(f"\n[{time.strftime('%H:%M:%S')}] === Task {task_id}: {task_description} ===")
        
        task_bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        init_states_path = os.path.join(init_states_folder, task.problem_folder, task.init_states_file)
        init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
        
        task_success = 0
        
        for trial in range(NUM_TRIALS):
            t0 = time.time()
            try:
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": RESOLUTION,
                    "camera_widths": RESOLUTION,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(0)
                
                init_idx = trial % init_states.shape[0]
                obs = env.set_init_state(init_states[init_idx])
                
                for _ in range(NUM_WAIT_STEPS):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                
                done = False
                first_action = None
                
                for t in range(MAX_STEPS):
                    img = get_libero_image(obs)
                    pil_img = Image.fromarray(img).convert("RGB")
                    
                    if CENTER_CROP:
                        pil_img = center_crop_pil(pil_img, CROP_SCALE)
                    
                    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                    
                    inputs = processor(prompt, pil_img).to(DEVICE, dtype=torch.bfloat16)
                    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
                    
                    if t == 0:
                        first_action = action.copy()
                    
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_success += 1
                        break
                
                elapsed = time.time() - t0
                status = "SUCCESS" if done else "FAIL"
                act_str = np.array2string(first_action, precision=3, suppress_small=True) if first_action is not None else "N/A"
                print(f"  Trial {trial}: {status} (steps={t+1}, {elapsed:.0f}s), a0={act_str}")
                
                env.close()
                
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  Trial {trial}: ERROR ({elapsed:.0f}s): {e}")
                try:
                    env.close()
                except:
                    pass
            
            total_episodes += 1
        
        sr = task_success / NUM_TRIALS
        results[task_description] = {"success": task_success, "total": NUM_TRIALS, "rate": sr}
        total_success += task_success
        print(f"  >>> Task SR: {sr:.1%} ({task_success}/{NUM_TRIALS})")
    
    overall_sr = total_success / total_episodes if total_episodes > 0 else 0
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall_sr:.1%} ({total_success}/{total_episodes})")
    print(f"{'='*60}")
    for desc, r in results.items():
        print(f"  {r['rate']:.0%}  {desc}")
    
    output = {
        "checkpoint": CHECKPOINT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_trials": NUM_TRIALS,
        "max_steps": MAX_STEPS,
        "center_crop": CENTER_CROP,
        "crop_scale": CROP_SCALE,
        "overall_success_rate": overall_sr,
        "per_task": results,
    }
    os.makedirs("./results/libero_closedloop_probe", exist_ok=True)
    out_path = "./results/libero_closedloop_probe/results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
