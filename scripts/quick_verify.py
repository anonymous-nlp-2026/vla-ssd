"""Quick 3-episode baseline verification for OpenVLA LIBERO-Goal."""
import os, sys
sys.path.insert(0, './data/LIBERO')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import get_libero_path

MODEL_PATH = './checkpoints/openvla-7b-finetuned-libero-goal'
UNNORM_KEY = 'libero_goal'
TASK_ID = 0
N_EPISODES = 3
MAX_STEPS = 300

np.random.seed(42)
torch.manual_seed(42)

print('Loading model...', flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
).to('cuda:0').eval()
print(f'Model loaded. vocab_size={model.vocab_size}', flush=True)

def center_crop_90(img):
    w, h = img.size
    new_w, new_h = int(w * 0.9), int(h * 0.9)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def get_action(model, processor, image_array, instruction):
    img = Image.fromarray(image_array)
    img = center_crop_90(img)
    prompt = f'In: What action should the robot take to {instruction}?\nOut:'
    inputs = processor(prompt, img).to('cuda:0', dtype=torch.bfloat16)
    action = model.predict_action(inputs['input_ids'], unnorm_key=UNNORM_KEY, do_sample=False, pixel_values=inputs['pixel_values'])
    return action

# Setup LIBERO env
from libero.libero import benchmark
bm = benchmark.get_benchmark('libero_goal')()
task = bm.get_task(TASK_ID)
task_description = task.language
bddl_path = bm.get_task_bddl_file_path(TASK_ID)
print(f'Task {TASK_ID}: {task.name}', flush=True)
print(f'Instruction: {task_description}', flush=True)

init_states_path = os.path.join(
    get_libero_path('init_states'),
    task.problem_folder,
    task.init_states_file,
)
init_states = torch.load(init_states_path, weights_only=False)

from libero.libero.envs import OffScreenRenderEnv
env_args = {
    'bddl_file_name': bddl_path,
    'camera_heights': 256,
    'camera_widths': 256,
}

# Quick test: single action
print('Testing single action...', flush=True)
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
obs = env.set_init_state(init_states[0])
img = obs['agentview_image'][::-1]
action = get_action(model, processor, img, task_description)
print(f'Action: {action}', flush=True)
env.close()
print('Single action OK. Starting episodes...', flush=True)

successes = []
for ep in range(N_EPISODES):
    env = OffScreenRenderEnv(**env_args)
    env.seed(ep)
    env.reset()
    obs = env.set_init_state(init_states[ep % len(init_states)])
    done = False
    for step in range(MAX_STEPS):
        img = obs['agentview_image'][::-1]
        action = get_action(model, processor, img, task_description)
        obs, reward, done, info = env.step(action)
        if done:
            break
    success = done or (reward > 0)
    successes.append(success)
    print(f'  Episode {ep}: {"SUCCESS" if success else "FAIL"} (steps={step+1})', flush=True)
    env.close()

sr = sum(successes) / len(successes)
print(f'\nTask {TASK_ID} SR = {sr:.2f} ({sum(successes)}/{len(successes)})', flush=True)
print(f'Verification {"PASSED" if sr > 0 else "FAILED"}: SR={sr:.2f}', flush=True)
