import torch
from safetensors.torch import save_file

# Source file
src = "M3DDM-Video-Outpainting/diffusion_pytorch_model.bin"
# Output destination
dst = "M3DDM-Video-Outpainting/diffusion_pytorch_model.safetensors"

# Load .bin
state_dict = torch.load(src, map_location="cpu")
# Save as .safetensors
save_file(state_dict, dst)

print(f"saved: {dst}")
