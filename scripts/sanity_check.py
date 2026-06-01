"""Action-Head Decoding Sanity Check: 4 diagnostics on 10 samples."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch, json, glob, h5py
import numpy as np
from PIL import Image
from pathlib import Path

CKPT = "./checkpoints/openvla-7b-finetuned-libero-goal"
DATA_DIR = "./data/libero/libero_goal"
DEVICE = "cuda:0"
N_ACTION_BINS = 256
EMPTY_TOKEN = 29871
ACTION_DIM = 7
N_SAMPLES = 10

# ─── Load model ───
print("Loading model...")
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True, local_files_only=True)
ModelClass = get_class_from_dynamic_module(
    "modeling_prismatic.OpenVLAForActionPrediction", CKPT)
model = ModelClass.from_pretrained(
    CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation="eager", local_files_only=True,
).to(DEVICE).eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

# ─── Codec ───
stats = model.norm_stats["libero_goal"]["action"]
vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
bins = np.linspace(-1, 1, N_ACTION_BINS)
bin_centers = (bins[:-1] + bins[1:]) / 2.0
q01 = np.array(stats["q01"])
q99 = np.array(stats["q99"])
mask_arr = np.array(stats["mask"])

def raw_to_norm(action):
    n = np.where(mask_arr, 2*(action - q01)/(q99 - q01 + 1e-8) - 1, action)
    return np.clip(n, -1, 1)

def norm_to_tokens(norm_action):
    idx = np.clip(np.digitize(norm_action, bins) - 1, 0, N_ACTION_BINS - 2)
    return vocab_size - 1 - idx

def tokens_to_norm(token_ids):
    d = np.clip(vocab_size - token_ids - 1, 0, len(bin_centers) - 1)
    return bin_centers[d]

norm_fn = model.language_model.model.norm
lm_head = model.language_model.lm_head
action_lo = vocab_size - N_ACTION_BINS + 1
action_hi = vocab_size - 1

# ─── Load samples ───
hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_demo.hdf5")))
samples = []
for hf_path in hdf5_files:
    task = Path(hf_path).stem.replace("_demo", "").replace("_", " ")
    prompt = f"In: What action should the robot take to {task}?\nOut:"
    with h5py.File(hf_path, "r") as hf:
        dk = sorted(hf["data"].keys(), key=lambda x: int(x.split("_")[1]))[0]
        imgs = np.array(hf[f"data/{dk}/obs/agentview_rgb"])
        acts = np.array(hf[f"data/{dk}/actions"])
    samples.append((prompt, Image.fromarray(imgs[0]), acts[0]))
    if len(samples) >= N_SAMPLES:
        break

print(f"Loaded {len(samples)} samples from {len(hdf5_files)} tasks")
print(f"Effective vocab_size: {vocab_size}")
print(f"Action token range: [{action_lo}, {action_hi}]")

def build_tf_input(prompt, img, gt_action):
    gt_norm = raw_to_norm(gt_action)
    gt_toks = norm_to_tokens(gt_norm)
    inp = proc(text=prompt, images=img, return_tensors="pt")
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    if inp["input_ids"][0, -1].item() != EMPTY_TOKEN:
        inp["input_ids"] = torch.cat([
            inp["input_ids"],
            torch.tensor([[EMPTY_TOKEN]], dtype=torch.long, device=DEVICE)
        ], dim=1)
        if "attention_mask" in inp:
            inp["attention_mask"] = torch.cat([
                inp["attention_mask"],
                torch.ones(1, 1, dtype=inp["attention_mask"].dtype, device=DEVICE)
            ], dim=1)
    act_ids = torch.tensor(gt_toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
    inp["input_ids"] = torch.cat([inp["input_ids"], act_ids], dim=1)
    if "attention_mask" in inp:
        inp["attention_mask"] = torch.cat([
            inp["attention_mask"],
            torch.ones(1, ACTION_DIM, dtype=inp["attention_mask"].dtype, device=DEVICE)
        ], dim=1)
    return inp, gt_norm, gt_toks

@torch.no_grad()
def autoregressive_generate(model, proc, prompt, img, norm_fn, lm_head):
    """Manual autoregressive generation (7 action tokens)."""
    inp = proc(text=prompt, images=img, return_tensors="pt")
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    if inp["input_ids"][0, -1].item() != EMPTY_TOKEN:
        inp["input_ids"] = torch.cat([
            inp["input_ids"],
            torch.tensor([[EMPTY_TOKEN]], dtype=torch.long, device=DEVICE)
        ], dim=1)
        if "attention_mask" in inp:
            inp["attention_mask"] = torch.cat([
                inp["attention_mask"],
                torch.ones(1, 1, dtype=inp["attention_mask"].dtype, device=DEVICE)
            ], dim=1)
    
    gen_tokens = []
    for step in range(ACTION_DIM):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inp, output_hidden_states=False)
        logits = out.logits[0, -1, :].float()
        next_tok = logits.argmax().item()
        gen_tokens.append(next_tok)
        inp["input_ids"] = torch.cat([
            inp["input_ids"],
            torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        ], dim=1)
        if "attention_mask" in inp:
            inp["attention_mask"] = torch.cat([
                inp["attention_mask"],
                torch.ones(1, 1, dtype=inp["attention_mask"].dtype, device=DEVICE)
            ], dim=1)
        del out
    return np.array(gen_tokens)

# ═══════════════════════════════════════════════════════════════════════
# CHECK 3: Double-norm bug (most critical — run first)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 3: Double-norm bug verification")
print("="*70)

inp, _, _ = build_tf_input(*samples[0])
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model(**inp, output_hidden_states=True)

n_hs = len(out.hidden_states)
print(f"Number of hidden states: {n_hs}")
for idx in [0, 1, n_hs-2, n_hs-1]:
    print(f"  hidden_states[{idx}] shape: {out.hidden_states[idx].shape}")

last_pos = out.hidden_states[0].shape[1] - 2
hs_exit = out.hidden_states[n_hs-1][0, last_pos, :]

logits_no_norm = lm_head(hs_exit.unsqueeze(0)).float()[0]
logits_with_norm = lm_head(norm_fn(hs_exit.unsqueeze(0))).float()[0]
logits_model = out.logits[0, last_pos, :].float()

diff_no_norm = (logits_no_norm - logits_model).abs().max().item()
diff_with_norm = (logits_with_norm - logits_model).abs().max().item()
mean_diff_no = (logits_no_norm - logits_model).abs().mean().item()
mean_diff_with = (logits_with_norm - logits_model).abs().mean().item()

print(f"\nmodel.language_model.model.norm type: {type(norm_fn)}")
print(f"hidden_states[{n_hs-1}] (exit) L2 norm:  {hs_exit.float().norm().item():.4f}")
print(f"After applying norm, L2 norm:            {norm_fn(hs_exit.unsqueeze(0)).float().norm().item():.4f}")

print(f"\nLogits comparison at position {last_pos}:")
print(f"  lm_head(hs) vs model logits:       max={diff_no_norm:.6f}, mean={mean_diff_no:.6f}")
print(f"  lm_head(norm(hs)) vs model logits: max={diff_with_norm:.6f}, mean={mean_diff_with:.6f}")

if diff_no_norm < 0.1 and diff_no_norm < diff_with_norm:
    IS_POST_NORM = True
    print("  >>> FINDING: hidden_states[-1] is POST-norm")
    print("  >>> Applying norm again = DOUBLE-NORM BUG!")
    print("  >>> FIX: For exit layer, use lm_head(hs) directly")
elif diff_with_norm < 0.1 and diff_with_norm < diff_no_norm:
    IS_POST_NORM = False
    print("  >>> FINDING: hidden_states[-1] is PRE-norm")
    print("  >>> Current code (apply norm) is correct")
else:
    IS_POST_NORM = None
    print("  >>> INCONCLUSIVE — both methods have significant diff")

tok_no_norm = logits_no_norm.argmax().item()
tok_with_norm = logits_with_norm.argmax().item()
tok_model = logits_model.argmax().item()
print(f"\nArgmax token at exit layer:")
print(f"  lm_head(hs):       {tok_no_norm}  (is_action={action_lo <= tok_no_norm <= action_hi})")
print(f"  lm_head(norm(hs)): {tok_with_norm}  (is_action={action_lo <= tok_with_norm <= action_hi})")
print(f"  model logits:      {tok_model}  (is_action={action_lo <= tok_model <= action_hi})")

print(f"\nL2 norms across all hidden_states:")
for li in [0, 1, 8, 16, 24, 31, 32]:
    if li < n_hs:
        h = out.hidden_states[li][0, last_pos, :]
        hn = norm_fn(h.unsqueeze(0))[0]
        print(f"  hs[{li:2d}]: raw_norm={h.float().norm().item():.1f}, "
              f"after_norm={hn.float().norm().item():.1f}")

del out, inp
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CHECK 1 & 2: Exit layer predictions (10 samples)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 1: Exit layer prediction vs ground truth (teacher forcing)")
print("="*70)

all_gt_toks = []
all_pred_correct = []
all_pred_buggy = []
all_gen_toks = []
all_top5 = []

for i, (prompt, img, gt_action) in enumerate(samples):
    gt_norm = raw_to_norm(gt_action)
    gt_toks = norm_to_tokens(gt_norm)
    
    inp, _, _ = build_tf_input(prompt, img, gt_action)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inp, output_hidden_states=True)
    
    seq_len = out.hidden_states[0].shape[1]
    positions = list(range(seq_len - ACTION_DIM - 1, seq_len - 1))
    hs_exit = out.hidden_states[n_hs-1][0, positions, :]
    
    # Correct method (based on Check 3)
    if IS_POST_NORM:
        logits_c = lm_head(hs_exit).float()
    else:
        logits_c = lm_head(norm_fn(hs_exit)).float()
    pred_c = logits_c.argmax(dim=-1).cpu().numpy()
    
    # Buggy method (always apply norm)
    logits_b = lm_head(norm_fn(hs_exit)).float()
    pred_b = logits_b.argmax(dim=-1).cpu().numpy()
    
    # Model's own logits
    pred_model = out.logits[0, positions, :].float().argmax(dim=-1).cpu().numpy()
    
    # Top-5
    top5_ids = logits_c.topk(5, dim=-1).indices.cpu().numpy()
    top5_hit = np.array([gt_toks[d] in top5_ids[d] for d in range(ACTION_DIM)])
    
    del out, inp
    torch.cuda.empty_cache()
    
    # Autoregressive generation (manual)
    gen_toks = autoregressive_generate(model, proc, prompt, img, norm_fn, lm_head)
    
    all_gt_toks.append(gt_toks)
    all_pred_correct.append(pred_c)
    all_pred_buggy.append(pred_b)
    all_gen_toks.append(gen_toks)
    all_top5.append(top5_hit)
    
    print(f"\nSample {i}:")
    print(f"  GT action (raw):       [{', '.join(f'{x:.4f}' for x in gt_action)}]")
    print(f"  GT action (norm):      [{', '.join(f'{x:.4f}' for x in gt_norm)}]")
    print(f"  GT token IDs:          {gt_toks}")
    print(f"  Exit layer (L{n_hs-1}, correct={'no-norm' if IS_POST_NORM else 'with-norm'}):")
    print(f"    Argmax token IDs:    {pred_c}")
    print(f"    De-tokenized:        [{', '.join(f'{x:.4f}' for x in tokens_to_norm(pred_c))}]")
    print(f"    Match GT tokens?:    {(pred_c == gt_toks).tolist()}")
    print(f"    Top-5 match?:        {top5_hit.tolist()}")
    print(f"  Buggy (always norm):   {pred_b}")
    print(f"    De-tokenized:        [{', '.join(f'{x:.4f}' for x in tokens_to_norm(pred_b))}]")
    print(f"  Model logits argmax:   {pred_model}")
    print(f"  Autoregressive gen:    {gen_toks}")
    print(f"    Generated action:    [{', '.join(f'{x:.4f}' for x in tokens_to_norm(gen_toks))}]")
    print(f"  Correct==Model:        {np.array_equal(pred_c, pred_model)}")
    print(f"  Gen[0]==Correct[0]:    {gen_toks[0] == pred_c[0]}")
    
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# CHECK 2: Token-level exact match rate
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 2: Token-level exact match rate summary")
print("="*70)

gt_a = np.array(all_gt_toks)
pc = np.array(all_pred_correct)
pb = np.array(all_pred_buggy)
ga = np.array(all_gen_toks)
t5 = np.array(all_top5)

em_c = (pc == gt_a)
em_b = (pb == gt_a)
em_g = (ga == gt_a)

print(f"\nCorrect method ({'no-norm' if IS_POST_NORM else 'with-norm'} for exit layer):")
print(f"  Exact match per dim:  {np.array2string(em_c.mean(axis=0), precision=3)}")
print(f"  Overall exact match:  {em_c.mean():.4f}")
print(f"  Top-5 match per dim:  {np.array2string(t5.mean(axis=0).astype(float), precision=3)}")
print(f"  Overall top-5 match:  {t5.mean():.4f}")

print(f"\nBuggy method (always norm):")
print(f"  Exact match per dim:  {np.array2string(em_b.mean(axis=0), precision=3)}")
print(f"  Overall exact match:  {em_b.mean():.4f}")

print(f"\nAutoregressive generate:")
print(f"  Exact match per dim:  {np.array2string(em_g.mean(axis=0), precision=3)}")
print(f"  Overall exact match:  {em_g.mean():.4f}")

# R²
gt_norm_all = np.array([raw_to_norm(s[2]) for s in samples])
pn_c = np.array([tokens_to_norm(t) for t in pc])
pn_b = np.array([tokens_to_norm(t) for t in pb])
gn_a = np.array([tokens_to_norm(t) for t in ga])

ss_tot = np.sum((gt_norm_all - gt_norm_all.mean(0))**2)
r2_c = 1 - np.sum((pn_c - gt_norm_all)**2) / (ss_tot + 1e-12)
r2_b = 1 - np.sum((pn_b - gt_norm_all)**2) / (ss_tot + 1e-12)
r2_g = 1 - np.sum((gn_a - gt_norm_all)**2) / (ss_tot + 1e-12)

print(f"\nR² (exit layer, {N_SAMPLES} samples):")
print(f"  Correct method: {r2_c:.4f}")
print(f"  Buggy method:   {r2_b:.4f}")
print(f"  AR generate:    {r2_g:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# CHECK 4: Bin→continuous roundtrip
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 4: Bin→continuous roundtrip verification")
print("="*70)

full_vocab = model.config.text_config.vocab_size
print(f"Full vocab size (config):  {full_vocab}")
print(f"pad_to_multiple_of:        {model.config.pad_to_multiple_of}")
print(f"Effective vocab_size:      {vocab_size}")
print(f"n_action_bins:             {N_ACTION_BINS}")
print(f"Bin edges:                 {len(bins)} (range [{bins[0]:.4f}, {bins[-1]:.4f}])")
print(f"Bin centers:               {len(bin_centers)} (range [{bin_centers[0]:.4f}, {bin_centers[-1]:.4f}])")

print(f"\nToken -> bin_idx -> value examples:")
for tok in [vocab_size-1, vocab_size-128, vocab_size-255, vocab_size-256+1]:
    bi = vocab_size - 1 - tok
    if 0 <= bi < len(bin_centers):
        print(f"  Token {tok} -> bin {bi:3d} -> value {bin_centers[bi]:.4f}")
    else:
        print(f"  Token {tok} -> bin {bi:3d} -> OUT OF RANGE")

print(f"\nRoundtrip: GT action -> normalize -> tokenize -> de-tokenize:")
for i in range(min(3, len(samples))):
    gn = raw_to_norm(samples[i][2])
    gt = norm_to_tokens(gn)
    rec = tokens_to_norm(gt)
    print(f"  Sample {i}: max error = {np.abs(gn - rec).max():.6f}")
    print(f"    Norm:      [{', '.join(f'{x:.4f}' for x in gn)}]")
    print(f"    Tokens:    {gt}")
    print(f"    Recovered: [{', '.join(f'{x:.4f}' for x in rec)}]")

# Cross-check with model's own de-tokenization
print(f"\nCross-check with model.predict_action de-tokenization:")
print(f"  model.vocab_size = {model.vocab_size}")
print(f"  model.bin_centers shape = {model.bin_centers.shape}")
test_toks = np.array([31900, 31950, 31800])
our_decode = tokens_to_norm(test_toks)
model_decode = model.bin_centers[np.clip(model.vocab_size - test_toks - 1, 0, model.bin_centers.shape[0] - 1)]
print(f"  Test tokens:  {test_toks}")
print(f"  Our decode:   {our_decode}")
print(f"  Model decode: {model_decode}")
print(f"  Match: {np.allclose(our_decode, model_decode)}")

print(f"\nDataset statistics (libero_goal):")
print(f"  Mean:  [{', '.join(f'{x:.4f}' for x in stats['mean'])}]")
print(f"  Std:   [{', '.join(f'{x:.4f}' for x in stats['std'])}]")
print(f"  q01:   [{', '.join(f'{x:.4f}' for x in stats['q01'])}]")
print(f"  q99:   [{', '.join(f'{x:.4f}' for x in stats['q99'])}]")
print(f"  Mask:  {stats['mask']}")

print("\n" + "="*70)
print("ALL 4 CHECKS COMPLETE")
print("="*70)
