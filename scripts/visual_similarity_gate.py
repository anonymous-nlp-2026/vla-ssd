"""Visual similarity gate for LIBERO-Goal: verify t=0 frames are near-identical across tasks."""
import argparse
import json
import os
import glob
import numpy as np
import h5py
import torch
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Visual similarity gate for LIBERO-Goal t=0 frames")
    parser.add_argument('--data_dir', default='./data/libero/libero_goal/')
    parser.add_argument('--model_path', default='./checkpoints/dinov2-large')
    parser.add_argument('--output_path', default='./results/visual_similarity_gate.json')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--device', default='auto', help='auto/cpu/cuda:N')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, '*.hdf5')))
    hdf5_files = [f for f in hdf5_files if not f.endswith('.aria2')]
    print(f"Found {len(hdf5_files)} tasks")

    task_names = []
    t0_frames = []
    for fpath in hdf5_files:
        name = os.path.basename(fpath).replace('_demo.hdf5', '').replace('.hdf5', '')
        with h5py.File(fpath, 'r') as f:
            img = f['data/demo_0/obs/agentview_rgb'][0]  # (128, 128, 3)
        task_names.append(name)
        t0_frames.append(img)
        print(f"  {name}: shape={img.shape}")

    print(f"\nLoading DINO-V2 from {args.model_path}...")
    model = AutoModel.from_pretrained(args.model_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_path)

    inputs = processor(images=t0_frames, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
    cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (N, 1024)
    print(f"Extracted {cls_tokens.shape[0]} cls_tokens, dim={cls_tokens.shape[1]}")

    sim_matrix = cosine_similarity(cls_tokens)  # (N, N)

    n = len(task_names)
    pair_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sims.append(sim_matrix[i, j])

    mean_sim = float(np.mean(pair_sims))
    min_sim = float(np.min(pair_sims))
    max_sim = float(np.max(pair_sims))
    gate_pass = mean_sim >= args.threshold

    print(f"\n{'='*50}")
    print(f"Mean cosine sim: {mean_sim:.4f}")
    print(f"Min cosine sim:  {min_sim:.4f}")
    print(f"Max cosine sim:  {max_sim:.4f}")
    print(f"Threshold:       {args.threshold}")
    print(f"Gate:            {'PASS' if gate_pass else 'FAIL'}")
    print(f"{'='*50}")

    result = {
        'task_names': task_names,
        'mean_cosine_sim': mean_sim,
        'min_cosine_sim': min_sim,
        'max_cosine_sim': max_sim,
        'threshold': args.threshold,
        'gate': 'PASS' if gate_pass else 'FAIL',
        'num_pairs': len(pair_sims),
        'similarity_matrix': sim_matrix.tolist(),
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main()
