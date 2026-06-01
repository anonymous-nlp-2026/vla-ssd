"""Control probe experiments for cross-instruction features."""
import sys
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from safetensors import safe_open
from transformers import AutoTokenizer
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

H5_PATH = "./results/cross_instruction/trained_cross_inst.h5"
MODEL_DIR = "./checkpoints/openvla-7b"
OUT_PATH = "./results/cross_instruction/control_probes.json"
LAYERS = [0, 8, 16, 24, 31]
L8_IDX = 1

def load_features(h5_path, feature_key, layer_idx, train_tasks=range(8), test_tasks=range(8,10)):
    X_train, y_train, X_test, y_test = [], [], [], []
    with h5py.File(h5_path, 'r') as f:
        for task_id in range(10):
            is_train = task_id in train_tasks
            for frame_id in range(50):
                for inst_id in range(10):
                    key = f"task_{task_id}/frame_{frame_id}/instruction_{inst_id}/{feature_key}"
                    vec = f[key][layer_idx].astype(np.float32)
                    if is_train:
                        X_train.append(vec)
                        y_train.append(inst_id)
                    else:
                        X_test.append(vec)
                        y_test.append(inst_id)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

print("=" * 60, flush=True)
print("CONTROL PROBE EXPERIMENTS", flush=True)
print("=" * 60, flush=True)

# --- Load data ---
print("\n[1/4] Loading features from h5...", flush=True)
X_train_img, y_train_img, X_test_img, y_test_img = load_features(H5_PATH, "image_mean", L8_IDX)
X_train_lp, y_train_lp, X_test_lp, y_test_lp = load_features(H5_PATH, "last_preaction", L8_IDX)
print(f"  image_mean L8: train={X_train_img.shape}, test={X_test_img.shape}", flush=True)
print(f"  last_preaction L8: train={X_train_lp.shape}, test={X_test_lp.shape}", flush=True)

# --- Exp 1: MLP Probe ---
print("\n[2/4] MLP Probe (4096->256->ReLU->10)...", flush=True)

class MLPProbe(nn.Module):
    def __init__(self, in_dim=4096, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_test, y_test, epochs=50, lr=1e-3, batch_size=256, patience=5):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)
    
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = MLPProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_preds = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_te).argmax(dim=1).numpy()
        acc = accuracy_score(y_te.numpy(), preds)
        if acc > best_acc:
            best_acc = acc
            best_preds = preds.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    f1 = f1_score(y_te.numpy(), best_preds, average='macro')
    return best_acc, f1

mlp_img_acc, mlp_img_f1 = train_mlp(X_train_img, y_train_img, X_test_img, y_test_img)
print(f"  MLP image_mean L8:      acc={mlp_img_acc:.4f}, f1={mlp_img_f1:.4f}", flush=True)

mlp_lp_acc, mlp_lp_f1 = train_mlp(X_train_lp, y_train_lp, X_test_lp, y_test_lp)
print(f"  MLP last_preaction L8:  acc={mlp_lp_acc:.4f}, f1={mlp_lp_f1:.4f}", flush=True)

# --- Exp 2: Random Features Baseline ---
print("\n[3/4] Random Features Baseline (PCA-256 + LogReg)...", flush=True)
n_train = X_train_img.shape[0]
n_test = X_test_img.shape[0]
X_rand_train = np.random.randn(n_train, 4096).astype(np.float32)
X_rand_test = np.random.randn(n_test, 4096).astype(np.float32)
y_rand_train = y_train_img.copy()
y_rand_test = y_test_img.copy()

pca = PCA(n_components=256, random_state=SEED)
X_rand_train_pca = pca.fit_transform(X_rand_train)
X_rand_test_pca = pca.transform(X_rand_test)

lr_model = LogisticRegression(max_iter=2000, random_state=SEED)
lr_model.fit(X_rand_train_pca, y_rand_train)
rand_preds = lr_model.predict(X_rand_test_pca)
rand_acc = accuracy_score(y_rand_test, rand_preds)
rand_f1 = f1_score(y_rand_test, rand_preds, average='macro')
print(f"  Random features LogReg: acc={rand_acc:.4f}, f1={rand_f1:.4f}", flush=True)

# --- Exp 3: Instruction Embedding Baseline ---
print("\n[4/4] Instruction Embedding Baseline...", flush=True)
print("  Loading tokenizer and embedding weights...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
instructions = [
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "push the plate to the front of the stove",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "turn on the stove",
]

sf_path = f"{MODEL_DIR}/model-00001-of-00003.safetensors"
with safe_open(sf_path, framework="pt", device="cpu") as f:
    embed_weight = f.get_tensor("language_model.model.embed_tokens.weight")
print(f"  Embedding weight shape: {embed_weight.shape}", flush=True)

# Compute mean-pooled embedding per instruction
inst_embeddings = []
for inst_text in instructions:
    token_ids = tokenizer.encode(inst_text, add_special_tokens=False)
    token_embeds = embed_weight[token_ids].float().numpy()
    inst_embeddings.append(token_embeds.mean(axis=0))
inst_embeddings = np.array(inst_embeddings)  # (10, 4096)

# Pairwise cosine similarity
cos_sim = cosine_similarity(inst_embeddings)
off_diag = cos_sim[np.triu_indices(10, k=1)]
print(f"  Pairwise cosine sim: mean={off_diag.mean():.4f}, min={off_diag.min():.4f}, max={off_diag.max():.4f}", flush=True)

# Use mean instruction embeddings as features in same train/test structure
# Each sample gets the mean embedding of its assigned instruction
X_emb_train = np.array([inst_embeddings[y] for y in y_train_img])
X_emb_test = np.array([inst_embeddings[y] for y in y_test_img])

pca_emb = PCA(n_components=9, random_state=SEED)
X_emb_train_pca = pca_emb.fit_transform(X_emb_train)
X_emb_test_pca = pca_emb.transform(X_emb_test)

lr_emb = LogisticRegression(max_iter=2000, random_state=SEED)
lr_emb.fit(X_emb_train_pca, y_train_img)
emb_preds = lr_emb.predict(X_emb_test_pca)
emb_acc = accuracy_score(y_test_img, emb_preds)
emb_f1 = f1_score(y_test_img, emb_preds, average='macro')
print(f"  Mean instruction embedding LogReg: acc={emb_acc:.4f}, f1={emb_f1:.4f}", flush=True)

# --- Save results ---
results = {
    "mlp_image_mean_L8": {"acc": round(mlp_img_acc, 4), "f1": round(mlp_img_f1, 4)},
    "mlp_last_preaction_L8": {"acc": round(mlp_lp_acc, 4), "f1": round(mlp_lp_f1, 4)},
    "random_features_logreg": {"acc": round(rand_acc, 4), "f1": round(rand_f1, 4)},
    "instruction_embedding_logreg": {
        "acc": round(emb_acc, 4), "f1": round(emb_f1, 4),
        "pairwise_cosine_sim": {"mean": float(round(off_diag.mean(), 4)), "min": float(round(off_diag.min(), 4)), "max": float(round(off_diag.max(), 4))},
        "note": "mean-pooled instruction token embeddings (L0), trivially 100% separable"
    },
    "metadata": {
        "mlp_config": "4096->256->ReLU->10, Adam lr=1e-3, epochs=50, patience=5",
        "n_train": int(n_train),
        "n_test": int(n_test),
        "seed": SEED,
        "layers": LAYERS,
        "layer_used": 8
    }
}

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_PATH}", flush=True)

# --- Summary ---
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Experiment':<35} {'Accuracy':>10} {'F1':>10}", flush=True)
print("-" * 60, flush=True)
print(f"{'MLP image_mean L8':<35} {mlp_img_acc:>10.4f} {mlp_img_f1:>10.4f}", flush=True)
print(f"{'MLP last_preaction L8':<35} {mlp_lp_acc:>10.4f} {mlp_lp_f1:>10.4f}", flush=True)
print(f"{'Random features (PCA+LogReg)':<35} {rand_acc:>10.4f} {rand_f1:>10.4f}", flush=True)
print(f"{'Instr. embedding (PCA+LogReg)':<35} {emb_acc:>10.4f} {emb_f1:>10.4f}", flush=True)
print("-" * 60, flush=True)
print("Chance level: 0.1000", flush=True)
print("=" * 60, flush=True)
