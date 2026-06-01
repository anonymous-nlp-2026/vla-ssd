"""RSA analysis for Instruction Ablation: with-instruction vs no-instruction untrained VLA.

Compares three conditions:
  1. with_instruction (untrained): untrained_libero_goal/
  2. no_instruction (untrained):   untrained_libero_goal_no_inst/
  3. DINO (anchor):                dinov2_libero_goal/

Output: rsa_no_instruction.json with per-layer rho, bootstrap 95% CI for delta.
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

FEAT_ROOT = Path("./features")
DATA_ROOT = Path("./data/libero/libero_goal")
OUT_DIR   = Path("./results/rsa")

TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]

TEMPORAL_BINS = {
    "early": (0.0, 0.33),
    "mid":   (0.33, 0.67),
    "late":  (0.67, 1.0),
}

N_VLA_LAYERS = 33
N_SAMPLE_FRAMES = 20
N_BOOTSTRAP = 1000
RNG_SEED = 42


def load_vla_features_from_dir(feat_dir, task, demo_ids, layer):
    path = feat_dir / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"demo_{did}/last_preaction"
            if key in f:
                result[did] = f[key][:, layer, :].astype(np.float32)
    return result


def load_dino_features(task, demo_ids):
    path = FEAT_ROOT / "dinov2_libero_goal" / f"{task}.h5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"traj_{did}/cls_token"
            if key in f:
                result[did] = f[key][:].astype(np.float32)
    return result


def load_actions(task, demo_ids):
    path = DATA_ROOT / f"{task}_demo.hdf5"
    result = {}
    with h5py.File(path, "r") as f:
        for did in demo_ids:
            key = f"data/demo_{did}/actions"
            if key in f:
                result[did] = f[key][:].astype(np.float32)
    return result


def get_phase_slice(arr, phase):
    T = len(arr)
    lo, hi = TEMPORAL_BINS[phase]
    t_norm = np.linspace(0, 1, T, endpoint=False)
    if phase == "late":
        mask = (t_norm >= lo) & (t_norm <= 1.0)
    else:
        mask = (t_norm >= lo) & (t_norm < hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.array([0])
    seg = arr[idx]
    if len(seg) > N_SAMPLE_FRAMES:
        sample_idx = np.linspace(0, len(seg) - 1, N_SAMPLE_FRAMES, dtype=int)
        seg = seg[sample_idx]
    return seg


def build_phase_tensor(data, keys, phase):
    slices = [get_phase_slice(data[k], phase) for k in keys]
    max_len = max(len(s) for s in slices)
    D = slices[0].shape[1]
    N = len(slices)
    tensor = np.zeros((N, max_len, D), dtype=np.float32)
    lengths = np.zeros(N, dtype=int)
    for i, s in enumerate(slices):
        tensor[i, :len(s)] = s
        lengths[i] = len(s)
    return tensor, lengths


def pairwise_cosine_dm_vectorized(tensor, lengths):
    N, T, D = tensor.shape
    norms = np.linalg.norm(tensor, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = tensor / norms

    for i in range(N):
        if lengths[i] < T:
            normed[i, lengths[i]:] = 0.0

    sim = np.einsum('itd,jtd->ijt', normed, normed)
    min_lens = np.minimum(lengths[:, None], lengths[None, :])
    frame_idx = np.arange(T)[None, None, :]
    valid = frame_idx < min_lens[:, :, None]

    dist = 1.0 - sim
    dist = dist * valid
    count = valid.sum(axis=2).astype(np.float32)
    count = np.maximum(count, 1.0)
    dm = dist.sum(axis=2) / count

    return dm.astype(np.float32)


def compute_rsa(rdm, adm):
    idx = np.triu_indices_from(rdm, k=1)
    r_vec = rdm[idx]
    a_vec = adm[idx]
    if np.std(r_vec) < 1e-10 or np.std(a_vec) < 1e-10:
        return 0.0
    rho, _ = spearmanr(r_vec, a_vec)
    return float(rho)


def bootstrap_rsa_delta(rdm_with, rdm_no, adm, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap CI for delta = RSA(with_inst) - RSA(no_inst)."""
    idx = np.triu_indices_from(rdm_with, k=1)
    r_with = rdm_with[idx]
    r_no = rdm_no[idx]
    a_vec = adm[idx]
    n_pairs = len(a_vec)

    rng = np.random.RandomState(seed)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sel = rng.choice(n_pairs, size=n_pairs, replace=True)
        rho_w = spearmanr(r_with[sel], a_vec[sel])[0] if np.std(r_with[sel]) > 1e-10 else 0.0
        rho_n = spearmanr(r_no[sel], a_vec[sel])[0] if np.std(r_no[sel]) > 1e-10 else 0.0
        deltas[b] = rho_w - rho_n

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return ci_lo, ci_hi, float(np.mean(deltas))


def main():
    n_demos = 10
    tasks = TASKS
    demo_ids = list(range(n_demos))

    with_inst_dir = FEAT_ROOT / "untrained_libero_goal"
    no_inst_dir = FEAT_ROOT / "untrained_libero_goal_no_inst"
    layers = list(range(N_VLA_LAYERS))

    print(f"Config: {len(tasks)} tasks x {n_demos} demos")
    print(f"  with_inst: {with_inst_dir}")
    print(f"  no_inst:   {no_inst_dir}")
    print(f"  Layers: {len(layers)} | Bootstrap: {N_BOOTSTRAP}")
    print()

    # Load actions
    print("Loading actions...")
    all_actions = {}
    for task in tasks:
        act = load_actions(task, demo_ids)
        for did, a in act.items():
            all_actions[(task, did)] = a
    print(f"  {len(all_actions)} demos")

    # Load DINO
    print("Loading DINO features...")
    dino_feats = {}
    for task in tasks:
        df = load_dino_features(task, demo_ids)
        for did in df:
            dino_feats[(task, did)] = df[did]
    print(f"  {len(dino_feats)} demos")

    common_keys_base = sorted(set(dino_feats) & set(all_actions))
    print(f"  Common demos: {len(common_keys_base)}")

    # Precompute ADM and DINO RDM per phase
    print("Precomputing ADM and DINO RDM...")
    adm_cache = {}
    dino_rdm_cache = {}
    dino_rsa_cache = {}
    for phase in TEMPORAL_BINS:
        act_tensor, act_lens = build_phase_tensor(all_actions, common_keys_base, phase)
        adm_cache[phase] = pairwise_cosine_dm_vectorized(act_tensor, act_lens)

        dino_tensor, dino_lens = build_phase_tensor(dino_feats, common_keys_base, phase)
        dino_rdm_cache[phase] = pairwise_cosine_dm_vectorized(dino_tensor, dino_lens)
        dino_rsa_cache[phase] = compute_rsa(dino_rdm_cache[phase], adm_cache[phase])

    # Per-layer RSA
    results = {
        "with_instruction_rsa": {},
        "no_instruction_rsa": {},
        "dino_rsa": {},
        "bootstrap_ci": {},
        "per_phase": {},
    }

    phase_accum = {phase: {"with": [], "no": [], "dino": []} for phase in TEMPORAL_BINS}

    print("\nPer-layer RSA:")
    for layer in layers:
        t0 = time.time()
        lk = f"layer_{layer}"

        with_feats = {}
        no_feats = {}
        for task in tasks:
            wf = load_vla_features_from_dir(with_inst_dir, task, demo_ids, layer)
            nf = load_vla_features_from_dir(no_inst_dir, task, demo_ids, layer)
            for did in wf:
                with_feats[(task, did)] = wf[did]
            for did in nf:
                no_feats[(task, did)] = nf[did]

        common_keys = sorted(set(common_keys_base) & set(with_feats) & set(no_feats))

        phase_rsa_w = {}
        phase_rsa_n = {}
        phase_rsa_d = {}
        phase_ci = {}

        for phase in TEMPORAL_BINS:
            adm = adm_cache[phase]

            w_tensor, w_lens = build_phase_tensor(with_feats, common_keys, phase)
            n_tensor, n_lens = build_phase_tensor(no_feats, common_keys, phase)

            rdm_w = pairwise_cosine_dm_vectorized(w_tensor, w_lens)
            rdm_n = pairwise_cosine_dm_vectorized(n_tensor, n_lens)

            rsa_w = compute_rsa(rdm_w, adm)
            rsa_n = compute_rsa(rdm_n, adm)
            rsa_d = dino_rsa_cache[phase]

            phase_rsa_w[phase] = rsa_w
            phase_rsa_n[phase] = rsa_n
            phase_rsa_d[phase] = rsa_d

            ci_lo, ci_hi, ci_mean = bootstrap_rsa_delta(rdm_w, rdm_n, adm)
            phase_ci[phase] = {"ci_lo": ci_lo, "ci_hi": ci_hi, "mean_delta": ci_mean}

            phase_accum[phase]["with"].append(rsa_w)
            phase_accum[phase]["no"].append(rsa_n)
            phase_accum[phase]["dino"].append(rsa_d)

        rsa_w_all = np.mean(list(phase_rsa_w.values()))
        rsa_n_all = np.mean(list(phase_rsa_n.values()))
        rsa_d_all = np.mean(list(phase_rsa_d.values()))

        # Overall bootstrap CI (average across phases)
        ci_lo_avg = np.mean([phase_ci[p]["ci_lo"] for p in TEMPORAL_BINS])
        ci_hi_avg = np.mean([phase_ci[p]["ci_hi"] for p in TEMPORAL_BINS])
        ci_mean_avg = np.mean([phase_ci[p]["mean_delta"] for p in TEMPORAL_BINS])

        results["with_instruction_rsa"][lk] = round(float(rsa_w_all), 4)
        results["no_instruction_rsa"][lk] = round(float(rsa_n_all), 4)
        results["dino_rsa"][lk] = round(float(rsa_d_all), 4)
        results["bootstrap_ci"][lk] = {
            "delta_mean": round(ci_mean_avg, 4),
            "ci_95_lo": round(ci_lo_avg, 4),
            "ci_95_hi": round(ci_hi_avg, 4),
            "per_phase": {p: {k: round(v, 4) for k, v in phase_ci[p].items()} for p in TEMPORAL_BINS},
        }
        results["per_phase"][lk] = {
            phase: {
                "with_inst": round(phase_rsa_w[phase], 4),
                "no_inst": round(phase_rsa_n[phase], 4),
                "dino": round(phase_rsa_d[phase], 4),
            }
            for phase in TEMPORAL_BINS
        }

        elapsed = time.time() - t0
        delta = rsa_w_all - rsa_n_all
        print(f"  L{layer:2d}: with={rsa_w_all:.4f}  no={rsa_n_all:.4f}  dino={rsa_d_all:.4f}"
              f"  Δ={delta:+.4f}  CI=[{ci_lo_avg:.4f},{ci_hi_avg:.4f}]  [{elapsed:.1f}s]")

    # Summary
    with_peaks = [results["with_instruction_rsa"][f"layer_{l}"] for l in layers]
    no_peaks = [results["no_instruction_rsa"][f"layer_{l}"] for l in layers]
    dino_vals = [results["dino_rsa"][f"layer_{l}"] for l in layers]

    peak_with = max(with_peaks)
    peak_no = max(no_peaks)
    peak_with_layer = layers[np.argmax(with_peaks)]
    peak_no_layer = layers[np.argmax(no_peaks)]
    mean_dino = np.mean(dino_vals)
    delta_peak = peak_with - peak_no

    # Gate judgment (D023)
    ci_at_peak = results["bootstrap_ci"][f"layer_{peak_with_layer}"]
    ci_excludes_zero = ci_at_peak["ci_95_lo"] > 0 or ci_at_peak["ci_95_hi"] < 0

    gate_pass = delta_peak > 0.10 and ci_excludes_zero and peak_no <= mean_dino + 0.05
    gate_fail = peak_no >= peak_with - 0.05
    if gate_pass:
        gate = "PASS"
    elif gate_fail:
        gate = "FAIL"
    else:
        gate = "INCONCLUSIVE"

    results["summary"] = {
        "with_instruction_peak": round(peak_with, 4),
        "with_instruction_peak_layer": int(peak_with_layer),
        "no_instruction_peak": round(peak_no, 4),
        "no_instruction_peak_layer": int(peak_no_layer),
        "dino_mean": round(float(mean_dino), 4),
        "delta_peak": round(delta_peak, 4),
        "bootstrap_ci_at_peak": ci_at_peak,
        "gate": gate,
    }

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rsa_no_instruction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("INSTRUCTION ABLATION RSA SUMMARY")
    print("=" * 80)
    print(f"\n{'Layer':>8} | {'With-Inst':>10} | {'No-Inst':>10} | {'DINO':>10} | {'Delta':>10} | {'CI_lo':>8} | {'CI_hi':>8}")
    print("-" * 78)
    for l in layers:
        lk = f"layer_{l}"
        w = results["with_instruction_rsa"][lk]
        n = results["no_instruction_rsa"][lk]
        d = results["dino_rsa"][lk]
        ci = results["bootstrap_ci"][lk]
        print(f"{l:>8d} | {w:>10.4f} | {n:>10.4f} | {d:>10.4f} | {w-n:>+10.4f} | {ci['ci_95_lo']:>8.4f} | {ci['ci_95_hi']:>8.4f}")

    print(f"\nPeak with-inst:  {peak_with:.4f} (L{peak_with_layer})")
    print(f"Peak no-inst:   {peak_no:.4f} (L{peak_no_layer})")
    print(f"DINO mean:       {mean_dino:.4f}")
    print(f"Delta (peak):   {delta_peak:+.4f}")
    print(f"Gate:            {gate}")


if __name__ == "__main__":
    main()
