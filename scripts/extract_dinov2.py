"""DINO-V2 ViT-L/14 feature extraction on LIBERO-10 demos."""
import argparse
import os
import glob
import time
import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/dinov2-large')
    parser.add_argument('--data_dir', default='./data/libero/libero_10/')
    parser.add_argument('--output_dir', default='./features/dinov2/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_key', default='agentview_rgb')
    return parser.parse_args()


def extract_features_for_file(model, processor, hdf5_path, args, device):
    """Extract DINO-V2 features for all demos in one HDF5 file."""
    task_name = os.path.basename(hdf5_path).replace('_demo.hdf5', '').replace('.hdf5', '')
    output_path = os.path.join(args.output_dir, f'{task_name}.h5')

    with h5py.File(hdf5_path, 'r') as src:
        demo_keys = sorted(
            [k for k in src['data'].keys() if k.startswith('demo_')],
            key=lambda x: int(x.split('_')[1])
        )
        print(f'  {len(demo_keys)} demos found')

        total_steps = 0
        t0 = time.time()

        with h5py.File(output_path, 'w') as dst:
            for di, dk in enumerate(demo_keys):
                images = src[f'data/{dk}/obs/{args.image_key}'][:]  # (T, 128, 128, 3)
                T = images.shape[0]
                total_steps += T

                cls_tokens = []
                patch_means = []

                for start in range(0, T, args.batch_size):
                    end = min(start + args.batch_size, T)
                    batch_imgs = images[start:end]  # (B, 128, 128, 3)

                    inputs = processor(
                        images=[img for img in batch_imgs],
                        return_tensors='pt'
                    )
                    pixel_values = inputs['pixel_values'].to(device)

                    with torch.no_grad():
                        outputs = model(pixel_values)
                    hidden = outputs.last_hidden_state  # (B, num_patches+1, 1024)

                    cls_tokens.append(hidden[:, 0, :].cpu().to(torch.float16).numpy())
                    patch_means.append(hidden[:, 1:, :].mean(dim=1).cpu().to(torch.float16).numpy())

                cls_all = np.concatenate(cls_tokens, axis=0)   # (T, 1024)
                patch_all = np.concatenate(patch_means, axis=0) # (T, 1024)

                grp = dst.create_group(f'traj_{di}')
                grp.create_dataset('cls_token', data=cls_all, compression='gzip', compression_opts=4)
                grp.create_dataset('patch_mean', data=patch_all, compression='gzip', compression_opts=4)

                if (di + 1) % 10 == 0 or di == len(demo_keys) - 1:
                    print(f'    demo {di+1}/{len(demo_keys)} done')

        elapsed = time.time() - t0
        fps = total_steps / elapsed if elapsed > 0 else 0
        print(f'  {total_steps} frames, {elapsed:.1f}s, {fps:.0f} fps -> {output_path}')
        return total_steps, elapsed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')

    print(f'Loading DINO-V2 from {args.model_path}...')
    model = AutoModel.from_pretrained(args.model_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    print(f'Model loaded. Device: {device}')

    hdf5_files = sorted(glob.glob(os.path.join(args.data_dir, '*.hdf5')))
    print(f'Found {len(hdf5_files)} HDF5 files\n')

    grand_total_steps = 0
    grand_total_time = 0

    for fi, fpath in enumerate(hdf5_files):
        print(f'[{fi+1}/{len(hdf5_files)}] {os.path.basename(fpath)}')
        steps, elapsed = extract_features_for_file(model, processor, fpath, args, device)
        grand_total_steps += steps
        grand_total_time += elapsed

    print(f'\n=== Done ===')
    print(f'Total: {grand_total_steps} frames, {grand_total_time:.1f}s, {grand_total_steps/grand_total_time:.0f} fps')


if __name__ == '__main__':
    main()
