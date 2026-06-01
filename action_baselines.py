import h5py, numpy as np, json, os, glob
from sklearn.metrics import r2_score

def load_all_actions(data_dir):
    task_actions = {}
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*_demo.hdf5"))):
        task_name = os.path.basename(fpath).replace("_demo.hdf5", "")
        with h5py.File(fpath, "r") as f:
            demos = []
            for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
                demos.append(f["data"][k]["actions"][()])
            task_actions[task_name] = demos
    return task_actions

def compute_r2(all_pred, all_target):
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    per_dim = []
    for d in range(all_pred.shape[1]):
        per_dim.append(r2_score(all_target[:, d], all_pred[:, d]))
    overall = r2_score(all_target, all_pred, multioutput="uniform_average")
    return per_dim, overall

def copy_baseline(actions_by_task):
    results = {}
    all_pred, all_target = [], []
    for task, demos in actions_by_task.items():
        task_pred, task_target = [], []
        for actions in demos:
            pred = actions[:-1]   # a_t
            target = actions[1:]  # a_{t+1}
            task_pred.append(pred)
            task_target.append(target)
        per_dim, overall = compute_r2(task_pred, task_target)
        results[task] = {"per_dim_r2": per_dim, "overall_r2": overall}
        all_pred.extend(task_pred)
        all_target.extend(task_target)
    global_per_dim, global_overall = compute_r2(all_pred, all_target)
    per_task_mean = np.mean([v["overall_r2"] for v in results.values()])
    return {
        "per_task": results,
        "global_per_dim_r2": global_per_dim,
        "global_overall_r2": global_overall,
        "per_task_mean_r2": float(per_task_mean),
    }

def random_frame_baseline(actions_by_task, seed=42):
    rng = np.random.RandomState(seed)
    results = {}
    all_pred, all_target = [], []
    for task, demos in actions_by_task.items():
        task_pred, task_target = [], []
        for actions in demos:
            T = len(actions)
            target = actions[1:]  # a_{t+1}, length T-1
            rand_idx = rng.randint(0, T, size=T - 1)
            pred = actions[rand_idx]
            task_pred.append(pred)
            task_target.append(target)
        per_dim, overall = compute_r2(task_pred, task_target)
        results[task] = {"per_dim_r2": per_dim, "overall_r2": overall}
        all_pred.extend(task_pred)
        all_target.extend(task_target)
    global_per_dim, global_overall = compute_r2(all_pred, all_target)
    per_task_mean = np.mean([v["overall_r2"] for v in results.values()])
    return {
        "per_task": results,
        "global_per_dim_r2": global_per_dim,
        "global_overall_r2": global_overall,
        "per_task_mean_r2": float(per_task_mean),
    }

def print_table(copy_res, rand_res):
    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    print("\n" + "=" * 80)
    print("ACTION PREDICTION BASELINES")
    print("=" * 80)
    
    print(f"\n{'Metric':<12} ", end="")
    for d in dim_names:
        print(f"{d:>8}", end="")
    print(f"  {'overall':>8}")
    print("-" * 80)
    
    print(f"{'Copy':<12} ", end="")
    for v in copy_res["global_per_dim_r2"]:
        print(f"{v:>8.4f}", end="")
    print(f"  {copy_res['global_overall_r2']:>8.4f}")
    
    print(f"{'Random':<12} ", end="")
    for v in rand_res["global_per_dim_r2"]:
        print(f"{v:>8.4f}", end="")
    print(f"  {rand_res['global_overall_r2']:>8.4f}")
    
    print(f"\n{'Metric':<20} {'Copy':>10} {'Random':>10}")
    print("-" * 42)
    print(f"{'Global Overall R²':<20} {copy_res['global_overall_r2']:>10.4f} {rand_res['global_overall_r2']:>10.4f}")
    print(f"{'Per-Task Mean R²':<20} {copy_res['per_task_mean_r2']:>10.4f} {rand_res['per_task_mean_r2']:>10.4f}")
    
    print(f"\n{'Task':<85} {'Copy R²':>8} {'Rand R²':>8}")
    print("-" * 103)
    for task in sorted(copy_res["per_task"].keys()):
        short = task[:82] + "..." if len(task) > 85 else task
        c = copy_res["per_task"][task]["overall_r2"]
        r = rand_res["per_task"][task]["overall_r2"]
        print(f"{short:<85} {c:>8.4f} {r:>8.4f}")

if __name__ == "__main__":
    data_dir = "./data/libero/libero_10/"
    print("Loading actions from LIBERO-10...")
    actions = load_all_actions(data_dir)
    print(f"Loaded {len(actions)} tasks, {sum(len(v) for v in actions.values())} demos total")

    print("Computing copy baseline...")
    copy_results = copy_baseline(actions)

    print("Computing random-frame baseline...")
    random_results = random_frame_baseline(actions)

    print_table(copy_results, random_results)

    out_dir = "./results/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "action_baselines.json")
    with open(out_path, "w") as f:
        json.dump({"copy_baseline": copy_results, "random_frame_baseline": random_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")
