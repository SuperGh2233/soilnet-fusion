import torch
from soilnet.submodules.vit_comer import ViTCoMer

model = ViTCoMer()
params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {params:,}")
print(f"Parameters in millions: {params/1e6:.2f} M")
print(f"Trainable parameters: {trainable:,}")


