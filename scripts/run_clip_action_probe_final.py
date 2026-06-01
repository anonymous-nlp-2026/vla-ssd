"""
CLIP action probe with fair DINO comparison (same code, same hyperparams).
"""
import os, json, glob, numpy as np, h5py, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLIP_DIR = "./features/clip_libero_goal"
DINO_DIR = "./features/dinov2_libero_goal"
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
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

def load_task_data(feat_dir, task_name, feature_key):
    fc = h5py.File(f'{feat_dir}/{task_name}.h5', 'r')
    fa_files = glob.glob(f'{ACTION_DIR}/{task_name}*demo.hdf5')
    fa = h5py.File(fa_files[0], 'r')
    feats_list, acts_list = [], []
    for traj_key in sorted(fc.keys()):
        idx = int(traj_key.split('_')[1])
        feat = np.array(fc[traj_key][feature_key], dtype=np.float32)
        act = np.array(fa['data'][f'demo_{idx}']['actions'], dtype=np.float32)
        T = min(len(feat), len(act))
        feats_list.append(feat[:T]); acts_list.append(act[:T])
    fc.close(); fa.close()
    return np.concatenate(feats_list), np.concatenate(acts_list)

def train_and_eval(feats, acts, seed=SEED):
    n = len(feats); n_train = int(n * 0.8)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    X_train, y_train = feats[idx[:n_train]], acts[idx[:n_train]]
    X_val, y_val = feats[idx[n_train:]], acts[idx[n_train:]]

    model = ActionProbe(input_dim=feats.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_ds = TensorDataset(torch.tensor(X_train).to(DEVICE), torch.tensor(y_train).to(DEVICE))
    val_X = torch.tensor(X_val).to(DEVICE); val_y = torch.tensor(y_val).to(DEVICE)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    best_val_loss = float('inf'); best_state = None; wait = 0; ep_trained = 0
    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            pred = model(xb); loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = criterion(model(val_X), val_y).item()
        ep_trained = epoch + 1
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred = model(val_X).cpu().numpy()
    y_np = val_y.cpu().numpy()

    ss_res = np.sum((y_np - pred) ** 2)
    ss_tot = np.sum((y_np - y_np.mean(axis=0)) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot)

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    per_dim_r2 = {}
    for i, name in enumerate(dim_names):
        ss_r = np.sum((y_np[:, i] - pred[:, i]) ** 2)
        ss_t = np.sum((y_np[:, i] - y_np[:, i].mean()) ** 2)
        per_dim_r2[name] = float(1 - ss_r / ss_t) if ss_t > 1e-10 else None

    gripper_acc = float(np.mean((pred[:, 6] >= 0) == (y_np[:, 6] >= 0)))

    return {
        'overall_R2': overall_r2,
        'per_dim_R2': per_dim_r2,
        'gripper_acc': gripper_acc,
        'val_loss': float(best_val_loss),
        'epochs_trained': ep_trained
    }

def run_all_tasks(feat_dir, feature_key, label):
    task_files = sorted(glob.glob(f'{feat_dir}/*.h5'))
    task_names = [os.path.splitext(os.path.basename(f))[0] for f in task_files]
    per_task_r2 = {}
    for t in task_names:
        feats, acts = load_task_data(feat_dir, t, feature_key)
        result = train_and_eval(feats, acts)
        per_task_r2[t] = result['overall_R2']
        print(f"  {label}/{feature_key}/{t}: R²={result['overall_R2']:.4f} (ep={result['epochs_trained']})", flush=True)
    mean_r2 = float(np.mean(list(per_task_r2.values())))
    print(f"  {label}/{feature_key} mean_R²={mean_r2:.4f}", flush=True)
    return per_task_r2, mean_r2

def bootstrap_ci(r2_a, r2_b, n_bootstrap=N_BOOTSTRAP):
    tasks = sorted(r2_a.keys())
    a = np.array([r2_a[t] for t in tasks])
    b = np.array([r2_b[t] for t in tasks])
    delta = float(np.mean(a) - np.mean(b))
    rng = np.random.RandomState(SEED)
    n = len(tasks)
    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_deltas.append(np.mean(a[idx]) - np.mean(b[idx]))
    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    p_value = float(np.mean(boot_deltas <= 0)) if delta > 0 else float(np.mean(boot_deltas >= 0))
    p_value = min(p_value * 2, 1.0)
    return {'delta': delta, 'ci_95': [ci_lo, ci_hi], 'p_value': round(p_value, 4)}

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results = {}

    # Run CLIP
    for fkey in ['cls_token', 'patch_mean']:
        print(f"\n=== CLIP {fkey} ===", flush=True)
        r2, mean = run_all_tasks(CLIP_DIR, fkey, 'CLIP')
        results[f'clip_{fkey}'] = {'per_task_R2': r2, 'mean_R2': mean}

    # Run DINO (same code for fair comparison)
    for fkey in ['cls_token', 'patch_mean']:
        print(f"\n=== DINO {fkey} (reproduced) ===", flush=True)
        r2, mean = run_all_tasks(DINO_DIR, fkey, 'DINO')
        results[f'dino_{fkey}_reproduced'] = {'per_task_R2': r2, 'mean_R2': mean}

    # Bootstrap CIs (same-code comparisons)
    print("\n=== Bootstrap CIs (same-code) ===", flush=True)
    for fkey in ['cls_token', 'patch_mean']:
        ci = bootstrap_ci(results[f'clip_{fkey}']['per_task_R2'], results[f'dino_{fkey}_reproduced']['per_task_R2'])
        results[f'bootstrap_clip_vs_dino_{fkey}'] = ci
        print(f"CLIP vs DINO ({fkey}): delta={ci['delta']:.4f}, CI={ci['ci_95']}, p={ci['p_value']}", flush=True)

    # Also bootstrap against original reference values
    PROJ_R2_REF = {
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
    ci_proj = bootstrap_ci(results['clip_cls_token']['per_task_R2'], PROJ_R2_REF)
    results['bootstrap_clip_cls_vs_proj_ref'] = ci_proj
    print(f"CLIP cls vs Proj (ref): delta={ci_proj['delta']:.4f}, CI={ci_proj['ci_95']}, p={ci_proj['p_value']}", flush=True)

    # Summary
    results['summary'] = {
        'clip_cls_token_mean_R2': results['clip_cls_token']['mean_R2'],
        'clip_patch_mean_mean_R2': results['clip_patch_mean']['mean_R2'],
        'dino_cls_token_reproduced_mean_R2': results['dino_cls_token_reproduced']['mean_R2'],
        'dino_patch_mean_reproduced_mean_R2': results['dino_patch_mean_reproduced']['mean_R2'],
        'projector_ref_mean_R2': 0.7063,
        'note': 'CLIP and DINO run with identical code/hyperparams. Projector ref from original code (not directly comparable).',
        'conclusion': None
    }

    clip_cls = results['clip_cls_token']['mean_R2']
    dino_cls = results['dino_cls_token_reproduced']['mean_R2']
    ci_cls = results['bootstrap_clip_vs_dino_cls_token']
    if ci_cls['p_value'] > 0.05:
        results['summary']['conclusion'] = f'CLIP ({clip_cls:.4f}) ≈ DINO ({dino_cls:.4f}), not significant (p={ci_cls["p_value"]}). Supports geometric-functional dissociation.'
    elif ci_cls['delta'] > 0:
        results['summary']['conclusion'] = f'CLIP ({clip_cls:.4f}) > DINO ({dino_cls:.4f}), significant (p={ci_cls["p_value"]}). VL pretraining adds predictive content.'
    else:
        results['summary']['conclusion'] = f'CLIP ({clip_cls:.4f}) < DINO ({dino_cls:.4f}), significant (p={ci_cls["p_value"]}). VL pretraining does not add predictive content.'

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)
    print(f"\nSummary: {json.dumps(results['summary'], indent=2)}", flush=True)
