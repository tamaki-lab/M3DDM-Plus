import torch
import gc
import argparse
from pathlib import Path
from collections import OrderedDict
from torch import nn


def _map_weights(state_dict, mapper: dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Replace portion of key with mapper value if mapper key is contained in k
        for old, new in mapper.items():
            if old in k:
                k = k.replace(old, new)
        new_state_dict[k] = v
    return new_state_dict

# Function that returns the updated ckpt_path


def create_converted_ckpt_file(checkpoint_path: str, device: str = "gpu") -> str:
    mapper = {
        # "substring to change in parameter": "replacement substring"
        "model.unet.": "",
    }
    # Set output directory to parent directory of checkpoint file
    base_path = Path(checkpoint_path).parent
    dest_ckpt_path = str(base_path / "diffusion_pytorch_model.bin")
    if Path(dest_ckpt_path).exists():
        return dest_ckpt_path
    map_loc = "cpu" if device == "cpu" else None
    model_ckpt = torch.load(checkpoint_path, map_location=map_loc)

    model_dict = model_ckpt['state_dict']
    model_dict = {k: v for k, v in model_dict.items() if not v.is_meta}
    # Filter only unet parameters
    model_dict = {k: v for k, v in model_dict.items() if k.startswith("model.unet")}

    new_model_dict = _map_weights(model_dict, mapper=mapper)

    # Save only model weights
    print(new_model_dict.keys())
    torch.save(new_model_dict, dest_ckpt_path)

    # Free memory
    del model_ckpt, model_dict, new_model_dict
    gc.collect()
    if device == "gpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dest_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ckpt to bin with device selection")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                        help="Load tensors on CPU or GPU (default: gpu)")
    parser.add_argument("--path", required=True, help="Checkpoint file path")
    args = parser.parse_args()
    create_converted_ckpt_file(args.path, args.device)
