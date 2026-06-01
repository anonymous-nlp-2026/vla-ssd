#!/usr/bin/env python3
import os, sys, time, subprocess, glob
import numpy as np

SHARD = './checkpoints/openvla-7b/model-00001-of-00003.safetensors'
EXPECTED_SIZE = 6948961960

def run(cmd):
    print(f'Running: {cmd}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        print(f'FAILED with exit code {r.returncode}')
        sys.exit(1)
    return r.stdout

print(f'[{time.strftime("%H:%M:%S")}] Waiting for download...')
while True:
    if not os.path.exists(SHARD):
        time.sleep(30)
        continue
    sz = os.path.getsize(SHARD)
    pct = sz * 100 // EXPECTED_SIZE
    print(f'  {sz/1024**3:.2f} GB / {EXPECTED_SIZE/1024**3:.2f} GB ({pct}%)')
    if sz >= EXPECTED_SIZE:
        r = subprocess.run(['pgrep', '-f', 'wget.*model-00001'], capture_output=True)
        if r.returncode != 0:
            print('Download complete!')
            break
    time.sleep(60)

print('\n=== Step 1: Verify shard ===')
from safetensors import safe_open
sf = safe_open(SHARD, framework='pt')
for key in ['projector.fc2.weight', 'projector.fc3.weight']:
    t = sf.get_tensor(key)
    norm = t.float().norm().item()
    print(f'{key}: norm={norm:.4f}')
    assert norm > 0.01, f'{key} is ZERO!'
print('VERIFIED')

print('\n=== Step 2: Clear HF cache ===')
os.system('rm -rf <PROJECT_ROOT>/.cache/huggingface/modules/transformers_modules/openvla_hyphen_7b/')

print('\n=== Step 3: Dry-run ===')
os.system('rm -rf /tmp/dryrun_features/')
run('cd . && HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --model_path ./checkpoints/openvla-7b --data_dir ./data/libero/libero_10/ --output_dir /tmp/dryrun_features/ --gpu 0 --max_demos 1')

print('\n=== Step 4: Validate ===')
import h5py
files = sorted(glob.glob('/tmp/dryrun_features/*.h5'))
print(f'Files: {len(files)}')
for fp in files[:2]:
    f = h5py.File(fp, 'r')
    dk = list(f.keys())[0]
    lp = f[dk]['last_preaction'][:].astype(np.float32)
    tvar = np.abs(lp[0,32,:] - lp[-1,32,:]).max()
    print(f'{os.path.basename(fp)}: shape={lp.shape}, temporal_var={tvar:.4f}')
    assert tvar > 0.001, 'Still identical!'
    f.close()
print('DRYRUN_VALIDATED')

print('\n=== Step 5: Init untrained ===')
run('cd . && CUDA_VISIBLE_DEVICES=1 python scripts/init_untrained.py --model_path ./checkpoints/openvla-7b --output_path ./checkpoints/openvla-7b-untrained/')

print(f'\n[{time.strftime("%H:%M:%S")}] === ALL STEPS COMPLETE ===')
