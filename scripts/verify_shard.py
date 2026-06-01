from safetensors import safe_open
sf = safe_open('./checkpoints/openvla-7b/model-00001-of-00003.safetensors', framework='pt')
ok = True
for key in ['projector.fc1.weight', 'projector.fc2.weight', 'projector.fc3.weight',
            'projector.fc1.bias', 'projector.fc2.bias', 'projector.fc3.bias']:
    t = sf.get_tensor(key)
    norm = t.float().norm().item()
    status = 'OK' if norm > 0.01 else 'ZERO!'
    print(f'{key}: norm={norm:.4f} [{status}]')
    if 'fc2' in key or 'fc3' in key:
        if norm < 0.01:
            ok = False
if not ok:
    print('VERIFICATION_FAILED'); exit(1)
print('VERIFICATION_PASSED')
