import torch

model_path = 'results/RUN_FiLM_Test_D_2026_06_12_T_14_10_best.pth.tar'
checkpoint = torch.load(model_path, map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Find FiLM alpha parameters
alpha_keys = [k for k in state_dict.keys() if 'alpha' in k.lower()]
print('FiLM Alpha 参数:')
for key in alpha_keys:
    val = state_dict[key].item()
    print(f'  {key}: {val:.6f}')
