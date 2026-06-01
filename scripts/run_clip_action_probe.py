import os, json, glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLIP_DIR = "./features/clip_libero_goal"
ACTION_DIR = "./data/libero/libero_goal"
OUT_PATH = "./results/functional_validation/action_probe_clip.json"
DEVICE = "cuda:0"
SEED = 42
N_BOOTSTRAP = 10000

torch.manual_seed(SEED)
np.random.seed(SEED)

class ActionProbe(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_task_data(task_name, feature_key):
    clip_file = os.path.join(CLIP_DIR, f"{task_name}.h5")
    # Find action file
    action_files = glob.glob(os.path.join(ACTION_DIR, f"{task_name}*demo.hdf5"))
    if not action_files:
        action_files = glob.glob(os.path.join(ACTION_DIR, f"*{task_name}*demo.hdf5"))
    action_file = action_files[0]

    feats_list, acts_list = [], []
    with h5py.File(clip_file, 'r') as fc, h5py.File(action_file, 'r') as fa:
        for traj_key in sorted(fc.keys()):
            idx = int(traj_key.split('_')[1])
            demo_key = f"demo_{idx}"
            feat = np.array(fc[traj_key][feature_key], dtype=np.float32)  # (T, 1024)
            act = np.array(fa['data'][demo_key]['actions'], dtype=np.float32)  # (T, 7)
            T = min(len(feat), len(act))
            # Predict next action: feat[t] -> act[t+1] (shift by 1)
            # Actually in existing probes, feat[t] predicts act[t] (current action)
            # Keep consistent: feat[t] -> act[t]
            feats_list.append(feat[:T])
            acts_list.append(act[:T])

    feats = np.concatenate(feats_list, axis=0)
    acts = np.concatenate(acts_list, axis=0)
    return feats, acts

def train_probe(X_train, y_train, X_val, y_val, max_epochs=100, patience=10):
    model = ActionProbe(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train).to(DEVICE), torch.tensor(y_train).to(DEVICE))
    val_X = torch.tensor(X_val).to(DEVICE)
    val_y = torch.tensor(y_val).to(DEVICE)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()

        epochs_trained = epoch + 1
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Compute R² on val
    with torch.no_grad():
        pred = model(val_X).cpu().numpy()
    y_np = val_y.cpu().numpy()

    ss_res = np.sum((y_np - pred) ** 2)
    ss_tot = np.sum((y_np - y_np.mean(axis=0)) ** 2)
    overall_r2 = 1 - ss_res / ss_tot

    per_dim_r2 = {}
    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    for i, name in enumerate(dim_names):
        ss_r = np.sum((y_np[:, i] - pred[:, i]) ** 2)
        ss_t = np.sum((y_np[:, i] - y_np[:, i].mean()) ** 2)
        per_dim_r2[name] = float(1 - ss_r / ss_t)

    # Gripper accuracy (binary: gripper < 0 = close, >= 0 = open)
    gripper_acc = float(np.mean((pred[:, 6] >= 0) == (y_np[:, 6] >= 0)))

    return {
        'overall_R2': float(overall_r2),
        'per_dim_R2': per_dim_r2,
        'gripper_acc': gripper_acc,
        'val_loss': float(best_val_loss),
        'epochs_trained': epochs_trained
    }

def run_probe(feature_key):
    task_files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.h5")))
    task_names = [os.path.splitext(os.path.basename(f))[0] for f in task_files]

    per_task_r2 = {}
    all_results = {}

    for task_name in task_names:
        print(f"  {task_name}...", end=" ", flush=True)
        feats, acts = load_task_data(task_name, feature_key)
        n = len(feats)
        n_train = int(n * 0.8)
        idx = np.random.permutation(n)
        X_train, y_train = feats[idx[:n_train]], acts[idx[:n_train]]
        X_val, y_val = feats[idx[n_train:]], acts[idx[n_train:]]

        result = train_probe(X_train, y_train, X_val, y_val)
        per_task_r2[task_name] = result['overall_R2']
        all_results[task_name] = result
        print(f"R²={result['overall_R2']:.4f} (ep={result['epochs_trained']})")

    mean_r2 = float(np.mean(list(per_task_r2.values())))

    # Aggregate overall R² from per-dim across all tasks
    # Actually compute overall R² by pooling all val predictions
    # For simplicity, use mean of per-task R²
    print(f"  mean_R²={mean_r2:.4f}")
    return per_task_r2, mean_r2, all_results

def bootstrap_ci(r2_a, r2_b, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap CI for difference r2_a - r2_b at task level."""
    tasks = sorted(r2_a.keys())
    a = np.array([r2_a[t] for t in tasks])
    b = np.array([r2_b[t] for t in tasks])
    delta = float(np.mean(a) - np.mean(b))

    np.random.seed(SEED)
    n = len(tasks)
    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_deltas.append(np.mean(a[idx]) - np.mean(b[idx]))
    boot_deltas = np.array(boot_deltas)

    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    p_value = float(np.mean(boot_deltas <= 0)) if delta > 0 else float(np.mean(boot_deltas >= 0))
    p_value = min(p_value * 2, 1.0)  # two-sided

    return {'delta': delta, 'ci_95': [ci_lo, ci_hi], 'p_value': p_value}

# Reference baselines (per-task R²)
PROJ_R2 = {
    "open_the_middle_drawer_of_the_cabinet": 0.8524968028068542,
    "open_the_top_drawer_and_put_the_bowl_inside": 0.7634366750717163,
    "push_the_plate_to_the_front_of_the_stove": 0.8558578491210938,
    "put_the_bowl_on_the_plate": 0.6052101850509644,
    "put_the_bowl_on_the_stove": 0.6865708231925964,
    "put_the_bowl_on_top_of_the_cabinet": 0.6757302284240723,
    "put_the_cream_cheese_in_the_bowl": 0.6336746215820312,
    "put_the_wine_bottle_on_the_rack": 0.5454312562942505,
    "put_the_wine_bottle_on_top_of_the_cabinet": 0.6573578119277954,
    "turn_on_the_stove": 0.7868210673332214
}

DINO_R2 = {
    "open_the_middle_drawer_of_the_cabinet": 0.8478043675422668,
    "open_the_top_drawer_and_put_the_bowl_inside": 0.7292581796646118,
    "push_the_plate_to_the_front_of_the_stove": 0.8544756174087524,
    "put_the_bowl_on_the_plate": 0.6344392895698547,
    "put_the_bowl_on_the_stove": 0.7261061072349548,
    "put_the_bowl_on_top_of_the_cabinet": 0.6704243421554565,
    "put_the_cream_cheese_in_the_bowl": 0.6190762519836426,
    "put_the_wine_bottle_on_the_rack": 0.5744811296463013,
    "put_the_wine_bottle_on_top_of_the_cabinet": 0.6501692533493042,
    "turn_on_the_stove": 0.8102135062217712
}

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results = {}

    for fkey in ['cls_token', 'patch_mean']:
        print(f"\n=== CLIP {fkey} ===")
        per_task_r2, mean_r2, task_results = run_probe(fkey)

        # Collect per-dim R2 from one representative aggregate
        all_per_dim = {dim: [] for dim in ['dx','dy','dz','drx','dry','drz','gripper']}
        for t, r in task_results.items():
            for dim in all_per_dim:
                all_per_dim[dim].append(r['per_dim_R2'][dim])
        avg_per_dim = {dim: float(np.mean(vals)) for dim, vals in all_per_dim.items()}

        avg_gripper_acc = float(np.mean([r['gripper_acc'] for r in task_results.values()]))
        avg_epochs = float(np.mean([r['epochs_trained'] for r in task_results.values()]))

        results[f'clip_{fkey}'] = {
            'per_task_R2': per_task_r2,
            'mean_R2': mean_r2,
            'per_dim_R2': avg_per_dim,
            'gripper_acc': avg_gripper_acc,
            'epochs_trained': avg_epochs
        }

    # Bootstrap CIs
    print("\n=== Bootstrap CIs ===")
    for fkey in ['cls_token', 'patch_mean']:
        clip_r2 = results[f'clip_{fkey}']['per_task_R2']

        ci_vs_proj = bootstrap_ci(clip_r2, PROJ_R2)
        ci_vs_dino = bootstrap_ci(clip_r2, DINO_R2)

        results[f'bootstrap_clip_{fkey}_vs_proj'] = ci_vs_proj
        results[f'bootstrap_clip_{fkey}_vs_dino'] = ci_vs_dino

        print(f"CLIP {fkey} vs Proj: delta={ci_vs_proj['delta']:.4f}, CI={ci_vs_proj['ci_95']}, p={ci_vs_proj['p_value']:.4f}")
        print(f"CLIP {fkey} vs DINO: delta={ci_vs_dino['delta']:.4f}, CI={ci_vs_dino['ci_95']}, p={ci_vs_dino['p_value']:.4f}")

    # Summary
    results['summary'] = {
        'clip_cls_token_mean_R2': results['clip_cls_token']['mean_R2'],
        'clip_patch_mean_mean_R2': results['clip_patch_mean']['mean_R2'],
        'projector_mean_R2': 0.7063,
        'dino_cls_token_mean_R2': 0.7116,
        'ordering': None  # fill below
    }

    vals = {
        'CLIP_cls': results['clip_cls_token']['mean_R2'],
        'CLIP_patch': results['clip_patch_mean']['mean_R2'],
        'Proj': 0.7063,
        'DINO': 0.7116
    }
    ordering = ' > '.join([k for k, v in sorted(vals.items(), key=lambda x: -x[1])])
    results['summary']['ordering'] = ordering

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")
