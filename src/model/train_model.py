"""
M3DDM Model for Video Outpainting Training

This module contains the M3DDM model class which is the core model
for video outpainting training. It handles mask generation, frame
preprocessing, and the forward pass through the diffusion pipeline.
"""

import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2 as image_transform
from einops import rearrange

from diffusers import AutoencoderKL, DDPMScheduler
from model.unet_3d_condition_video import UNet3DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pipelines.stable_diffusion.pipeline_stable_diffusion_video2video_mask_for_video_outpainting import StableDiffusionPipelineVideo2VideoMaskC


class M3DDM_Plus(nn.Module):
    """
    Model Class for Video Outpainting

    This model handles:
    - VAE encoding/decoding
    - 3D UNet for video diffusion
    - Mask generation (same or individual per frame)
    - Frame preprocessing and augmentation
    """

    def __init__(self, pretrained_sd_dir, video_outpainting_model_dir, size, enable_unet_gradient_checkpointing=False, seed=6):
        super().__init__()
        self.seed = seed

        self.size = size

        self.num_train_timesteps = 1000
        self.weight_dtype = torch.float32
        # Initialize scheduler (controls gradual image generation from noise)
        noise_scheduler = PNDMScheduler.from_pretrained(pretrained_sd_dir,
                                                        subfolder="scheduler",
                                                        local_files_only=True)

        # Initialize pre-noise scheduler (for preprocessing)
        scheduler_pre = DDPMScheduler(beta_start=0.00085,
                                      beta_end=0.012,
                                      beta_schedule="scaled_linear",
                                      num_train_timesteps=self.num_train_timesteps)

        # Load VAE model (for image compression and latent space representation)
        self.vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_sd_dir, 'vae'),
                                                 local_files_only=True)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Load 3D UNet model
        self.unet = UNet3DConditionModel.from_pretrained(
            video_outpainting_model_dir,
            local_files_only=True,
            torch_dtype=self.weight_dtype
        )
        if enable_unet_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Create video outpainting pipeline
        # NOTE: Device movement is managed by LightningModule, not done here (multi-GPU support)
        self.pipeline = StableDiffusionPipelineVideo2VideoMaskC(
            vae=self.vae.to(self.weight_dtype).eval(),
            unet=self.unet.to(self.weight_dtype).train(),
            scheduler=noise_scheduler,  # Scheduler for generation process
            scheduler_pre=scheduler_pre)  # Scheduler for preprocessing

        # Random generator for reproducibility (device set dynamically in forward)
        self.generator = None

    # TODO: refactor
    def forward(self, start_frame, select_indexes, total_frames_num, selected_global_frames_idx, all_latents, num_global_frames, stride, vr, fps, real_frames_num, batch_size):
        # Select frame indices at current stride
        stride_indexes = select_indexes[::stride]
        # Get selected frame indices
        selected_frame_indexes = stride_indexes[start_frame:start_frame + 16]

        # Pad with last frame if less than 16 frames
        if len(selected_frame_indexes) < 16:
            selected_frame_indexes = selected_frame_indexes + [
                select_indexes[-1]
            ] * (16 - len(selected_frame_indexes))
            assert len(selected_frame_indexes) == 16
        # print(selected_frame_indexes, 'out of', total_frames_num, 'frames')

        # Load selected frames and convert to tensor format
        frames = vr.get_batch(
            list(map(lambda x: x % real_frames_num,
                     selected_frame_indexes))).permute(0, 3, 1, 2).float()

        # Load global frames
        global_frames = vr.get_batch(
            list(
                map(lambda x: x % real_frames_num,
                    selected_global_frames_idx))).permute(0, 3, 1, 2).float()

        # Global frame preprocessing (resize and normalize)
        global_frames = self.preprocess(
            global_frames / 255.0,  # Convert to [0, 1] range
            T.Compose([
                T.Resize((self.size, self.size), antialias=True),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]))

        # Preprocess selected frames for outpainting
        # Resize frames to specified size and generate masks
        frames_resized, mask, masked_frames = self.preprocess_frames(
            frames, p_frame=0.5)

        # Initial frame preprocessing
        init_frames = self.preprocess(
            frames_resized / 255.,
            T.Compose([
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]))
        # Convert tensor dimensions ([channels, num_frames, height, width])
        init_frames = rearrange(init_frames, "f c h w ->c f h w")

        # Masked frame preprocessing
        masked_frames = self.preprocess(
            masked_frames / 255.,
            T.Compose([
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]))
        # Convert tensor dimensions ([channels, num_frames, height, width])
        masked_frames = rearrange(masked_frames, "f c h w ->c f h w")

        # Get device from module parameters
        device = next(self.vae.parameters()).device
        # Move tensors to device and dtype
        init_frames = init_frames.to(dtype=self.weight_dtype, device=device)
        masked_frames = masked_frames.to(
            dtype=self.weight_dtype, device=device)
        mask = mask.to(dtype=self.weight_dtype, device=device)
        global_frames = global_frames.to(
            dtype=self.weight_dtype, device=device)

        # Apply same mask to global frames if they share indices with input frames
        # mask: (1, Fs, H, W)
        mask_frames = mask.squeeze(0)  # (Fs, H, W)
        C = global_frames.shape[1]
        # selected_frame_indexes and selected_global_frames_idx are shared variables
        mask_global_list = []
        for gf_idx in selected_global_frames_idx:
            if gf_idx in selected_frame_indexes:
                sel_pos = selected_frame_indexes.index(gf_idx)
                mask_global_list.append(mask_frames[sel_pos])
            else:
                mask_global_list.append(torch.zeros_like(mask_frames[0]))
        # (Fg, H, W) -> (Fg,1,H,W) -> (Fg,C,H,W)
        if mask_global_list:
            mask_global = torch.stack(
                mask_global_list, dim=0).unsqueeze(1).repeat(1, C, 1, 1)
            global_frames = global_frames * (1 - mask_global)

        if num_global_frames == 0:
            global_frames = None

        # Initialize random generator on current device if not yet initialized
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.seed)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,), device=device
        )

        # Execute pipeline for video outpainting
        # ori_image: Original input frame tensor
        # use_add_noise: Whether to apply additional noise (for processing diversity)
        # mask_image: Masked frame tensor
        # mask: Binary mask (0=retain region, 1=generate region)
        noise_pred, noise = self.pipeline.outpainting_with_random_masked_one_step(
            ori_image=init_frames,
            use_add_noise=True,  # Add noise for generation diversity
            mask_image=masked_frames,
            mask=mask,  # Binary mask for generation
            cond=True,  # Enable conditional generation
            strength=1.0,  # Generation strength (1.0 = complete regeneration)
            batch_size=batch_size,  # Batch size (adjustable based on GPU memory)
            num_frames=16,  # Number of frames to input to u-net at once
            height=init_frames.shape[-2],  # Frame height
            width=init_frames.shape[-1],  # Frame width
            fps=fps // stride,  # Frame rate to process (adjusted by stride)
            t=timesteps,  # Noise scheduler timestep
            generator=self.generator,  # Random generator for reproducibility
            latents_dtype=self.weight_dtype,  # Latent variable dtype (memory efficiency)
            num_inference_steps=50,
            mask_ratio=6 / 16,  # Training mask ratio
            copy_raw_images=False,  # Don't copy raw images
            guidance_scale=1,  # Classifier-free guidance scale (higher = more faithful, lower = more diverse)
            previous_guidance_scale=1.0,  # Previous frame guidance scale
            cur_step=start_frame // 16,  # Current processing step
            already_outpainted_latents=[
                all_latents[i_frame]  # Latent variables of already generated frames
                for i_frame in selected_frame_indexes
            ],
            copy_already_frame=False,  # Don't copy existing frames
            global_frames=global_frames,  # Global frames for overall consistency
            num_global_frames=num_global_frames,  # Number of global frames
        )

        return noise_pred, noise

    def preprocess_frames(self, frames, p_frame: float = 0.5):
        """
        Apply resize -> padding -> mask generation -> case branching for raw data replacement ->
        mask application to original frame sequence, returning (frames_resized, mask_tensor, masked_frames)

        Three cases (paper-compliant):
        1) All frames masked
        2) Only first & last frames have raw data
        3) Each frame has raw data with probability p_frame
        """
        import torch.nn.functional as F

        size = self.size
        Fv, C, H, W = frames.shape

        # Resize + random crop (same position for entire video)
        transform = image_transform.Compose([
            image_transform.RandomResizedCrop((size, size), antialias=True)
        ])
        frames_resized = transform(frames)  # (F, C, size, size)

        # Mask generation
        mask_tensor = self.generate_mask(size, Fv, frames.device)
        # For broadcasting
        mask_bc = mask_tensor.repeat(C, 1, 1, 1)

        # Apply mask to frames
        fr = frames_resized.permute(1, 0, 2, 3)  # (C, F, H, W)
        masked_latents = fr * (1 - mask_bc)
        masked_frames = masked_latents.permute(1, 0, 2, 3)  # (F, C, H, W)

        # Case branching by paper ratios
        mask_type = random.choices([1, 2, 3], weights=[0.3, 0.35, 0.35])[0]

        if mask_type == 2:
            # Only first & last frames have raw data
            mask_tensor[0, 0] = 0
            mask_tensor[0, -1] = 0
            masked_frames[0] = frames_resized[0]
            masked_frames[-1] = frames_resized[-1]

        elif mask_type == 3:
            # Each frame has raw data with probability p_frame
            for f in range(Fv):
                if random.random() < p_frame:
                    mask_tensor[0, f] = 0
                    masked_frames[f] = frames_resized[f]
        # case 1: do nothing (all frames masked)

        return frames_resized, mask_tensor, masked_frames

    def preprocess(self, image, img_transform):
        image = img_transform(image)
        return image

    def generate_mask(self, size, Fv, device):
        """Apply same mask to entire video."""
        mask_canvas = torch.zeros((size, size), device=device)
        mask_ratio = random.uniform(0.15, 0.75)

        strategy = random.choices(
            ['single', 'bi', 'four'],
            [0.3, 0.55, 0.15]
        )[0]

        if strategy == 'four':
            directions = ['left', 'right', 'top', 'bottom']
        elif strategy == 'bi':
            directions = random.choice([('left', 'right'), ('top', 'bottom')])
        else:  # 'single'
            directions = [random.choice(['left', 'right', 'top', 'bottom'])]

        self._apply_mask_directions(mask_canvas, size, mask_ratio, directions)

        mask_tensor = mask_canvas.unsqueeze(0).repeat(1, Fv, 1, 1)
        return mask_tensor



    # REVIEW: What is base? If base is passed by reference, should use self.base
    def _apply_mask_directions(self, base, size, ratio, directions):
        """Helper method to apply mask in specified directions."""
        mask_size = int(size * ratio / 2)
        for dir in directions:
            if dir == 'left':
                base[:, :mask_size] = 1
            elif dir == 'right':
                base[:, -mask_size:] = 1
            elif dir == 'top':
                base[:mask_size, :] = 1
            else:  # 'bottom'
                base[-mask_size:, :] = 1
