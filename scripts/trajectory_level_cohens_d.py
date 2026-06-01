"""Trajectory-level Cohen's d: re-run all experiments with per-demo MSE tracking.

GPU 0: head_ablation_top24 (5 tasks) + specificity_bottom24 (3 tasks) + counterfactual (3 tasks)
GPU 1: dose_response (all 8 conditions, 3 tasks)

Outputs: ./results/trajectory_level_cohens_d.json
"""

import gc
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, "./data/LIBERO")

os.environ["MUJOCO_GL"] = "egl"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats

SEED = 42
MODEL_PATH = "./checkpoints/openvla-7b"
UNNORM_KEY = "bridge_orig"
DATA_DIR = "./data/libero/libero_goal"
RESULTS_DIR = "./results"

NUM_IMAGE_TOKENS = 256
INST_START = 1 + NUM_IMAGE_TOKENS

N_TASKS_3 = 3
N_TASKS_5 = 5
N_DEMOS = 10
N_TIMESTEPS = 50

INSTRUCTION_HEADS_24 = [
    0, 1, 2, 4, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20,
    22, 23, 25, 26, 27, 29, 30, 31,
]

BOTTOM_HEADS_24 = [
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20,
    21, 23, 24, 26, 28, 29, 30, 31,
]

_FRAC_T0 = [0.8536376953125,0.53509521484375,0.68304443359375,0.471466064453125,0.5867233276367188,0.498046875,0.394805908203125,0.569671630859375,0.58123779296875,0.452606201171875,0.7152099609375,0.5701904296875,0.5364990234375,0.41016387939453125,0.69647216796875,0.6108169555664062,0.525848388671875,0.55120849609375,0.5181121826171875,0.6407318115234375,0.5516357421875,0.458770751953125,0.8058147430419922,0.57220458984375,0.366424560546875,0.769256591796875,0.56707763671875,0.653472900390625,0.503387451171875,0.589263916015625,0.5357666015625,0.5589599609375]
_FRAC_T1 = [0.856719970703125,0.5569610595703125,0.7030029296875,0.4571533203125,0.5877685546875,0.4925537109375,0.40106201171875,0.557342529296875,0.5824584960937,0.437347412109375,0.71026611328125,0.6070327758789062,0.53277587890625,0.5465087890625,0.5405426025390625,0.6549072265625,0.52325439453125,0.452239990234375,0.8254013061523438,0.590728759765625,0.3663177490234375,0.781494140625,0.55816650390625,0.6743011474609375,0.498626708984375,0.611358642578125,0.542572021484375,0.547607421875,0.5181121826171875,0.6407318115234375,0.5516357421875,0.458770751953125]
_FRAC_T2 = [0.852569580078125,0.54840087890625,0.7081298828125,0.475555419921875,0.5866012573242188,0.50164794921875,0.413299560546875,0.573699951171875,0.5865478515625,0.4401397705078125,0.71649169921875,0.57843017578125,0.534912109375,0.4060821533203125,0.71636962890625,0.6230087280273438,0.538482666015625,0.5455322265625,0.5452117919921875,0.646087646484375,0.53546142578125,0.45086669921875,0.8309307098388672,0.576568603515625,0.3722991943359375,0.787872314453125,0.5629119873046875,0.686920166015625,0.503082275390625,0.60833740234375,0.54925537109375,0.55242919921875]
MEAN_INSTR_FRAC = np.mean([_FRAC_T0, _FRAC_T1, _FRAC_T2], axis=0)
HEADS_SORTED_DESC = np.argsort(-MEAN_INSTR_FRAC).tolist()

TASK_INSTRUCTIONS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "correct": "open the middle drawer of the cabinet",
        "paraphrased": "pull open the center drawer on the cabinet",
        "wrong": "push the plate to the front of the stove",
    },
    "open_the_top_drawer_and_put_the_bowl_inside": {
        "correct": "open the top drawer and put the bowl inside",
        "paraphrased": "open the upper drawer and place the bowl in it",
        "wrong": "open the middle drawer of the cabinet",
    },
    "push_the_plate_to_the_front_of_the_stove": {
        "correct": "push the plate to the front of the stove",
        "paraphrased": "slide the plate toward the front of the stove",
        "wrong": "open the top drawer and put the bowl inside",
    },
}


def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, processor


def predict_action(model, input_ids, pixel_values, unnorm_key):
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
                input_ids=next_token, attention_mask=attn_mask,
                pixel_values=None, past_key_values=past_key_values, use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    predicted_action_token_ids = np.array(generated_ids)
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    action = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
    return action


def install_ablation_hook(model):
    import math
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    layer_attn = model.language_model.model.layers[1].self_attn
    layer_attn._ablate_heads = None
    original_forward = layer_attn.forward

    def ablated_forward(hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False,
                        use_cache=False, cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = layer_attn.q_proj(hidden_states)
        key_states = layer_attn.k_proj(hidden_states)
        value_states = layer_attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, layer_attn.num_heads, layer_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer_attn.num_key_value_heads, layer_attn.head_dim).transpose(1, 2)
        pkv = getattr(layer_attn, "past_key_value", past_key_value)
        cos, sin = layer_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if pkv is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = pkv.update(key_states, value_states, layer_attn.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, layer_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, layer_attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(layer_attn.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=layer_attn.attention_dropout, training=layer_attn.training)
        if layer_attn._ablate_heads is not None:
            kv_len = attn_weights.shape[-1]
            if INST_START < kv_len:
                attn_weights[:, layer_attn._ablate_heads, :, INST_START:] = 0.0
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn_weights = attn_weights / row_sums
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, layer_attn.hidden_size)
        attn_output = layer_attn.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, pkv

    layer_attn.forward = ablated_forward
    return layer_attn


def get_task_files(n_tasks):
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))[:n_tasks]


def get_language_instruction(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        info = json.loads(f["data"].attrs["problem_info"])
    return info["language_instruction"]


def get_shuffled_instruction(instruction, rng):
    words = instruction.split()
    rng.shuffle(words)
    return " ".join(words)


def bootstrap_cohens_d(demo_mses, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(demo_mses)
    arr = np.array(demo_mses)
    d_boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        sample = arr[idx]
        d_boots[i] = np.mean(sample) / (np.std(sample, ddof=1) + 1e-12)
    return float(np.percentile(d_boots, 2.5)), float(np.percentile(d_boots, 97.5))


def compute_trajectory_d(demo_mses):
    arr = np.array(demo_mses)
    d = float(np.mean(arr) / (np.std(arr, ddof=1) + 1e-12))
    ci_lo, ci_hi = bootstrap_cohens_d(arr)
    return d, ci_lo, ci_hi, len(arr)


# ── GPU 0: Head ablation + counterfactual ──────────────────────────────────
def gpu0_worker(result_dict):
    device = "cuda:0"
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    print(f"[GPU 0] Loading model...")
    model, processor = load_model(device)
    layer_attn = install_ablation_hook(model)
    print(f"[GPU 0] Model loaded.")

    task_files_5 = get_task_files(N_TASKS_5)
    task_files_3 = task_files_5[:N_TASKS_3]

    results = {}
    total_calls = 0
    t_start = time.time()

    # ── 1. Head ablation top-24 (5 tasks) ──
    print(f"\n[GPU 0] === Head ablation top-24 (5 tasks) ===")
    demo_mses_top24 = []
    for task_idx, hdf5_path in enumerate(task_files_5):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        instruction = get_language_instruction(hdf5_path)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        print(f"[GPU 0] Task {task_idx}: {task_name}")

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(N_DEMOS):
                demo_key = f"data/demo_{demo_idx}"
                actions_data = f[f"{demo_key}/actions"][:]
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(N_TIMESTEPS, len(actions_data))
                step_mses = []

                for step_idx in range(n_steps):
                    img = Image.fromarray(images_data[step_idx])
                    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]

                    layer_attn._ablate_heads = None
                    b_action = predict_action(model, input_ids, pixel_values, UNNORM_KEY)

                    layer_attn._ablate_heads = INSTRUCTION_HEADS_24
                    a_action = predict_action(model, input_ids, pixel_values, UNNORM_KEY)

                    mse = float(np.mean((b_action - a_action) ** 2))
                    step_mses.append(mse)
                    total_calls += 2
                    torch.cuda.empty_cache()

                demo_mses_top24.append(float(np.mean(step_mses)))

                if total_calls % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{total_calls} calls, {elapsed:.0f}s, {total_calls/elapsed:.1f}/s]")

        gc.collect()
        torch.cuda.empty_cache()

    results["head_ablation_top24"] = demo_mses_top24
    print(f"[GPU 0] top24 done: {len(demo_mses_top24)} demos, mean={np.mean(demo_mses_top24):.6f}")

    # ── 2. Specificity control bottom-24 (3 tasks) ──
    print(f"\n[GPU 0] === Specificity control bottom-24 (3 tasks) ===")
    demo_mses_bottom24 = []
    for task_idx, hdf5_path in enumerate(task_files_3):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        instruction = get_language_instruction(hdf5_path)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        print(f"[GPU 0] Task {task_idx}: {task_name}")

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(N_DEMOS):
                demo_key = f"data/demo_{demo_idx}"
                actions_data = f[f"{demo_key}/actions"][:]
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(N_TIMESTEPS, len(actions_data))
                step_mses = []

                for step_idx in range(n_steps):
                    img = Image.fromarray(images_data[step_idx])
                    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)

                    layer_attn._ablate_heads = None
                    b_action = predict_action(model, inputs["input_ids"], inputs["pixel_values"], UNNORM_KEY)

                    layer_attn._ablate_heads = BOTTOM_HEADS_24
                    a_action = predict_action(model, inputs["input_ids"], inputs["pixel_values"], UNNORM_KEY)

                    mse = float(np.mean((b_action - a_action) ** 2))
                    step_mses.append(mse)
                    total_calls += 2
                    torch.cuda.empty_cache()

                demo_mses_bottom24.append(float(np.mean(step_mses)))

        gc.collect()
        torch.cuda.empty_cache()

    results["specificity_bottom24"] = demo_mses_bottom24
    print(f"[GPU 0] bottom24 done: {len(demo_mses_bottom24)} demos, mean={np.mean(demo_mses_bottom24):.6f}")

    # ── 3. Counterfactual (3 tasks) ──
    print(f"\n[GPU 0] === Counterfactual (3 tasks, 4 conditions) ===")
    layer_attn._ablate_heads = None
    cf_demo_mses = {c: [] for c in ["shuffled", "paraphrased", "wrong", "empty"]}

    for task_idx, hdf5_path in enumerate(task_files_3):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        correct_instruction = get_language_instruction(hdf5_path)
        task_cfg = TASK_INSTRUCTIONS[task_name]
        print(f"[GPU 0] Task {task_idx}: {task_name}")

        cf_instructions = {
            "wrong": task_cfg["wrong"],
            "empty": "",
            "shuffled": get_shuffled_instruction(correct_instruction, rng),
            "paraphrased": task_cfg["paraphrased"],
        }

        correct_prompt = f"In: What action should the robot take to {correct_instruction}?\nOut:"
        cf_prompts = {}
        for cond, inst in cf_instructions.items():
            if inst == "":
                cf_prompts[cond] = "In: What action should the robot take to ?\nOut:"
            else:
                cf_prompts[cond] = f"In: What action should the robot take to {inst}?\nOut:"

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(N_DEMOS):
                demo_key = f"data/demo_{demo_idx}"
                actions_data = f[f"{demo_key}/actions"][:]
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(N_TIMESTEPS, len(actions_data))
                demo_step_mses = {c: [] for c in cf_demo_mses}

                for step_idx in range(n_steps):
                    img = Image.fromarray(images_data[step_idx])
                    inputs_c = processor(correct_prompt, img).to(device, dtype=torch.bfloat16)
                    c_action = predict_action(model, inputs_c["input_ids"], inputs_c["pixel_values"], UNNORM_KEY)
                    total_calls += 1

                    for cond in cf_demo_mses:
                        inputs_cf = processor(cf_prompts[cond], img).to(device, dtype=torch.bfloat16)
                        cf_action = predict_action(model, inputs_cf["input_ids"], inputs_cf["pixel_values"], UNNORM_KEY)
                        mse = float(np.mean((c_action - cf_action) ** 2))
                        demo_step_mses[cond].append(mse)
                        total_calls += 1

                    torch.cuda.empty_cache()

                for cond in cf_demo_mses:
                    cf_demo_mses[cond].append(float(np.mean(demo_step_mses[cond])))

                if total_calls % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{total_calls} calls, {elapsed:.0f}s, {total_calls/elapsed:.1f}/s]")

        gc.collect()
        torch.cuda.empty_cache()

    for cond, mses in cf_demo_mses.items():
        results[f"counterfactual_{cond}"] = mses
        print(f"[GPU 0] cf_{cond}: {len(mses)} demos, mean={np.mean(mses):.6f}")

    elapsed = time.time() - t_start
    print(f"\n[GPU 0] Done. {total_calls} calls in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    result_dict["gpu0"] = results


# ── GPU 1: Dose-response ──────────────────────────────────────────────────
def gpu1_worker(result_dict):
    device = "cuda:1"
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[GPU 1] Loading model...")
    model, processor = load_model(device)
    layer_attn = install_ablation_hook(model)
    print(f"[GPU 1] Model loaded.")

    task_files_3 = get_task_files(N_TASKS_3)
    dose_conditions = [4, 8, 12, 16, 20, 24, 28, 32]
    condition_heads = {n: HEADS_SORTED_DESC[:n] for n in dose_conditions}

    results = {}
    dose_demo_mses = {n: [] for n in dose_conditions}
    total_calls = 0
    t_start = time.time()

    print(f"\n[GPU 1] === Dose-response (3 tasks, 8 conditions) ===")
    for task_idx, hdf5_path in enumerate(task_files_3):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        instruction = get_language_instruction(hdf5_path)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        print(f"[GPU 1] Task {task_idx}: {task_name}")

        with h5py.File(hdf5_path, "r") as f:
            for demo_idx in range(N_DEMOS):
                demo_key = f"data/demo_{demo_idx}"
                actions_data = f[f"{demo_key}/actions"][:]
                images_data = f[f"{demo_key}/obs/agentview_rgb"][:]
                n_steps = min(N_TIMESTEPS, len(actions_data))
                demo_step_mses = {n: [] for n in dose_conditions}

                for step_idx in range(n_steps):
                    img = Image.fromarray(images_data[step_idx])
                    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]

                    layer_attn._ablate_heads = None
                    b_action = predict_action(model, input_ids, pixel_values, UNNORM_KEY)
                    total_calls += 1

                    for n in dose_conditions:
                        layer_attn._ablate_heads = condition_heads[n]
                        a_action = predict_action(model, input_ids, pixel_values, UNNORM_KEY)
                        mse = float(np.mean((b_action - a_action) ** 2))
                        demo_step_mses[n].append(mse)
                        total_calls += 1

                    torch.cuda.empty_cache()

                for n in dose_conditions:
                    dose_demo_mses[n].append(float(np.mean(demo_step_mses[n])))

                if total_calls % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{total_calls} calls, {elapsed:.0f}s, {total_calls/elapsed:.1f}/s]")

        gc.collect()
        torch.cuda.empty_cache()

    for n in dose_conditions:
        results[f"dose_response_top{n}"] = dose_demo_mses[n]
        print(f"[GPU 1] top{n}: {len(dose_demo_mses[n])} demos, mean={np.mean(dose_demo_mses[n]):.6f}")

    elapsed = time.time() - t_start
    print(f"\n[GPU 1] Done. {total_calls} calls in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    result_dict["gpu1"] = results


# ── Token-level d values from original experiments ──────────────────────
TOKEN_LEVEL_D = {
    "head_ablation_top24": 0.651,
    "specificity_bottom24": 0.565,
    "dose_response_top4": 0.482,
    "dose_response_top8": 0.508,
    "dose_response_top12": 0.560,
    "dose_response_top16": 0.592,
    "dose_response_top20": 0.617,
    "dose_response_top24": 0.637,
    "dose_response_top28": 0.654,
    "dose_response_top32": 0.639,
    "counterfactual_shuffled": 0.625,
    "counterfactual_paraphrased": 0.697,
    "counterfactual_wrong": 0.844,
    "counterfactual_empty": 0.968,
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_global = time.time()

    manager = mp.Manager()
    result_dict = manager.dict()

    p0 = mp.Process(target=gpu0_worker, args=(result_dict,))
    p1 = mp.Process(target=gpu1_worker, args=(result_dict,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    if p0.exitcode != 0 or p1.exitcode != 0:
        print(f"ERROR: GPU 0 exit={p0.exitcode}, GPU 1 exit={p1.exitcode}")
        sys.exit(1)

    all_results = {}
    all_results.update(result_dict["gpu0"])
    all_results.update(result_dict["gpu1"])

    output_results = []
    inflation_factors = []

    for condition, demo_mses in all_results.items():
        d_traj, ci_lo, ci_hi, n = compute_trajectory_d(demo_mses)
        token_d = TOKEN_LEVEL_D.get(condition, None)
        if token_d and token_d > 0:
            inflation = token_d / d_traj if d_traj > 0 else float("inf")
            inflation_factors.append(inflation)
        else:
            inflation = None

        entry = {
            "experiment": condition.rsplit("_", 1)[0] if "dose_response" not in condition else "dose_response",
            "condition": condition,
            "token_level_d": token_d,
            "trajectory_level_d": round(d_traj, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n_demos": n,
            "demo_mses": [round(x, 6) for x in demo_mses],
            "inflation_factor": round(inflation, 3) if inflation and inflation != float("inf") else None,
        }
        output_results.append(entry)
        print(
            f"{condition:30s}: token_d={token_d:.3f}  traj_d={d_traj:.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}]  n={n}  "
            f"inflation={inflation:.2f}x" if inflation and inflation != float("inf") else
            f"{condition:30s}: traj_d={d_traj:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] n={n}"
        )

    output_results.sort(key=lambda x: x["condition"])

    if inflation_factors:
        inf_mean = float(np.mean(inflation_factors))
        inf_range = [float(np.min(inflation_factors)), float(np.max(inflation_factors))]
    else:
        inf_mean = None
        inf_range = None

    output = {
        "method": "demo-level aggregation, paired Cohen's d, bootstrap CI B=10000",
        "description": "Each demo's MSE is the mean of its 50 timestep-level MSEs. "
                       "Cohen's d = mean(demo_mses) / std(demo_mses, ddof=1). "
                       "Bootstrap CI: 10000 resamples of demo-level values.",
        "n_timesteps_per_demo": N_TIMESTEPS,
        "results": output_results,
        "inflation_factor": {
            "description": "token_level_d / trajectory_level_d (how much token-level d overestimates)",
            "mean": round(inf_mean, 3) if inf_mean else None,
            "range": [round(x, 3) for x in inf_range] if inf_range else None,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "trajectory_level_cohens_d.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t_global
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
