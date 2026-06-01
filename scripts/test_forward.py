import torch, h5py
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
proc = AutoProcessor.from_pretrained('./checkpoints/openvla-7b', trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    './checkpoints/openvla-7b',
    torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    attn_implementation='eager'
).to('cuda:0').eval()
f = h5py.File('./data/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5', 'r')
img0 = Image.fromarray(f['data']['demo_0']['obs']['agentview_rgb'][0])
img100 = Image.fromarray(f['data']['demo_0']['obs']['agentview_rgb'][100])
f.close()
prompt = 'In: What action should the robot take to turn on the stove and put the moka pot on it?\nOut:'
with torch.no_grad():
    inp0 = proc(text=prompt, images=img0, return_tensors='pt')
    inp0 = {k: v.to('cuda:0') for k, v in inp0.items()}
    out0 = model(**inp0, output_hidden_states=True)
    inp100 = proc(text=prompt, images=img100, return_tensors='pt')
    inp100 = {k: v.to('cuda:0') for k, v in inp100.items()}
    out100 = model(**inp100, output_hidden_states=True)
for li in [0, 16, 32]:
    h0 = out0.hidden_states[li][0]
    h1 = out100.hidden_states[li][0]
    lp_diff = (h0[-1].float() - h1[-1].float()).abs().max().item()
    im_diff = (h0[1:257].float().mean(0) - h1[1:257].float().mean(0)).abs().max().item()
    print(f'Layer {li:2d}: lp_diff={lp_diff:.6f}, im_diff={im_diff:.6f}')
diff32 = (out0.hidden_states[32][0][-1].float() - out100.hidden_states[32][0][-1].float()).abs().max().item()
assert diff32 > 0.001, f'Features still identical! diff={diff32}'
print('FORWARD_TEST_PASSED')
