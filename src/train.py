from logger import configure_logger
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime

# Suppress DDP-related warnings
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
warnings.filterwarnings("ignore", message=".*Found.*module.*in eval mode.*")

import torch
import torch.nn as nn
from evaluate import VideoOutpaintingEvaluator

from dataset import VideoDataModule, DataloaderConfig

import numpy as np
from decord import VideoReader
from decord._ffi.base import DECORDError
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

import random
import json
import shutil
from convert_ckpt_to_bin import create_converted_ckpt_file
from model.train_model import M3DDM_Plus


# LoadDataset class has been moved to src/dataset/video_dataset.py as VideoDataset


# LightningModule definition
class LitM3DDM(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Model and loss
        self.model = M3DDM_Plus(
            hparams.pretrained_sd_dir,
            hparams.video_outpainting_model_dir,
            size=hparams.size,
            enable_unet_gradient_checkpointing=hparams.enable_unet_gradient_checkpointing,
            seed=hparams.seed
        )
        self.mse = nn.MSELoss()

        # Lightning may automatically switch to train even when .val is specified, so explicitly freeze parameters
        self.model.vae.requires_grad_(False)
        self.model.vae.eval()

        # For Comet logging
        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def training_step(self, batch):
        loss = self.getLoss(batch)
        self.train_loss_sum += loss.detach().item()
        self.train_loss_count += 1
        running_avg = self.train_loss_sum / self.train_loss_count
        self.log('train/loss', running_avg, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch):
        if self.hparams.disable_validation:
            return

        loss = self.getLoss(batch)
        prefix = "init-val" if getattr(self, "initial_eval", False) else "val"
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        return loss

    def on_train_epoch_end(self):
        if self.train_loss_count > 0:
            epoch_avg = self.train_loss_sum / self.train_loss_count
        else:
            epoch_avg = 0.0

        self.log('train/loss_epoch', epoch_avg, on_step=False, on_epoch=True, prog_bar=True)

        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def getLoss(self, batch):
        try:
            # TODO: Currently only batch=1 is supported
            vr = VideoReader(batch[0])
        except Exception as e:
            # Skip if error occurs during data loading that would stop training
            print(f"Error reading video {batch[0]}: {e}")
            # device = next(self.model.parameters()).device
            return torch.tensor(0.0, requires_grad=True)

        fps = vr.get_avg_fps()
        real_frames_num = len(vr)

        # Randomly determine stride
        max_interval = min(30, max(1, real_frames_num // 15))  # Limit to range where 16 frames can be extracted
        interval = random.randint(1, max_interval)
        max_start = real_frames_num - interval * 15 - 1  # Need 15 steps ahead
        start_frame = random.randint(0, max(0, max_start))
        select_indexes = list(range(real_frames_num))
        selected_global_frames = np.linspace(
            0, real_frames_num - 1, self.hparams.global_frames, dtype=int)

        all_latents = [None] * real_frames_num
        # forward
        try:
            pred_noise, true_noise = self.model(
                # pred_noise, true_noise, mask = self.model(
                start_frame,
                select_indexes,
                real_frames_num,
                selected_global_frames,
                all_latents,
                self.hparams.global_frames,
                interval,
                vr,
                fps,
                real_frames_num,  # TODO: Duplicate argument?
                batch_size=self.hparams.batch_size
            )
        except DECORDError as e:
            print(f"Error reading frames in forward: {e}")
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, requires_grad=True)

        loss = self.mse(pred_noise, true_noise)
        del pred_noise, true_noise

        # # Broadcast mask to channels
        # mask_bc = mask.unsqueeze(1)  # (1,1,F,H,W)
        # mask_bc = mask_bc.expand_as(pred_noise)  # (B,C,F,H,W)

        # # MSE only for masked areas
        # diff2 = (pred_noise - true_noise).pow(2) * mask_bc
        # loss = diff2.sum() / mask_bc.sum().clamp(min=1)  # Prevent division by zero

        # del pred_noise, true_noise, mask
        return loss

    def configure_optimizers(self):
        # Model update
        # TODO: Should this also use lr?
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)


def run_outpainting_eval(experiment, pl_module, epoch, args, output_path):
    """Execute outpainting evaluation."""
    model_dir = args.video_outpainting_model_dir
    subdir = 'evaluation_base' if epoch is None else f'evaluation_epoch_{epoch}'
    eval_root = os.path.join(output_path, subdir)
    evaluator = VideoOutpaintingEvaluator(
        video_dir=args.eval_video_dir,
        output_root=eval_root,
        pretrained_sd_dir=args.pretrained_sd_dir,
        video_outpainting_model_dir=model_dir,
        crop_ratio=args.eval_crop_ratio,
        crop_axis=args.eval_crop_axis,
        target_ratio_list=args.eval_target_ratio_list,
        output_size=args.size,
        num_global_frames=args.global_frames,
        limit_outpainting_frames=16
    )
    # If training, use weights on GPU from training
    if epoch is not None:
        for pipe in evaluator.runner.all_pipelines:
            pipe.unet.load_state_dict(pl_module.model.unet.state_dict())
            pipe.unet.to(pipe.unet.dtype).to(pipe.device).eval()
    # Stop gradients and execute evaluation
    with torch.no_grad():
        evaluator.evaluate_and_log_metrics(experiment, epoch)


class OutpaintEvalCallback(pl.Callback):
    """Callback to execute outpainting evaluation at end of epoch."""

    def __init__(self, args, output_path):
        super().__init__()
        self.args = args
        self.output_path = output_path

    def on_validation_epoch_end(self, trainer, pl_module):
        # For multi-GPU, run evaluation only on rank 0 (main process)
        if trainer.global_rank != 0:
            return
        print(f"Running outpainting evaluation at epoch {trainer.current_epoch}")
        epoch = -1 if getattr(pl_module, "initial_eval", False) else trainer.current_epoch
        run_outpainting_eval(trainer.logger.experiment, pl_module, epoch, self.args, self.output_path)


def main():
    parser = ArgumentParser()
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_sd_dir", type=str, required=True)
    parser.add_argument("--video_outpainting_model_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int,
                        default=16, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--global_frames", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 to auto-detect all GPUs in CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--disable_comet", "-dc", action="store_true",
                        default=False, help="Disable Comet logging")
    parser.add_argument("--disable_checkpoint", action="store_true",
                        default=False, help="Disable model saving per epoch")  # TODO: Rename
    parser.add_argument("--output_dir", type=str,
                        default="output", help="Output directory")
    parser.add_argument("--max_samples", type=int,
                        default=None, help="Maximum number of samples to load from dataset")
    parser.add_argument("--limit_val_batches", type=int, help="Limit number of videos in validation_step")
    parser.add_argument("--disable_validation",
                        action="store_true", default=False, help="Disable validation step")
    parser.add_argument("--enable_unet_gradient_checkpointing", action="store_true", default=False, help="Enable UNet gradient checkpointing, increases runtime but reduces memory usage")

    parser.add_argument("--accumulate_grad_batches",
                        type=int, default=16, help="Number of batches for gradient accumulation")
    parser.add_argument("--eval_video_dir", type=str, default=None, help="Evaluation video directory")
    parser.add_argument("--eval_crop_ratio", type=float, default=0.25, help="Crop ratio for outpainting evaluation")
    parser.add_argument("--eval_crop_axis", choices=["horizontal", "vertical"], default="horizontal", help="Crop axis for outpainting evaluation")
    parser.add_argument("--eval_target_ratio_list", type=str, default="16:9", help="Target aspect ratio list for evaluation")
    parser.add_argument("--seed", type=int, default=6, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Create save directory and record args as JSON
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_path = os.path.join(args.output_dir, run_timestamp)
    os.makedirs(output_path, exist_ok=True)
    args.output_path = output_path

    args_json_path = os.path.join(output_path, 'args.json')
    with open(args_json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

    comet_logger = configure_logger(
        logged_params=vars(args),
        model_name="train",
        disable_logging=args.disable_comet,
        lightning=True,
    )

    dataloader_config = DataloaderConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        train_shuffle=True,
        val_shuffle=False
    )
    data_module = VideoDataModule(dataloader_config)

    lit_model = LitM3DDM(args)

    print(f"output: {output_path}")
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    args.checkpoint_dir = checkpoint_dir

    # Copy config.json from timestamp directory (checkpoint folder) to checkpoint folder
    # TODO: Copying might not be ideal
    config_src = os.path.join(args.video_outpainting_model_dir, 'config.json')
    config_dst = os.path.join(checkpoint_dir, 'config.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(config_src, config_dst)

    callbacks = [RichProgressBar()]
    if not args.disable_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='m3ddm-{epoch}',
            save_last=True,
            every_n_epochs=1
        )
        callbacks.insert(0, checkpoint_callback)
    if args.eval_video_dir:
        callbacks.append(OutpaintEvalCallback(args, output_path))

    # GPU count determination: -1 means auto-detect (use all GPUs specified in CUDA_VISIBLE_DEVICES)
    num_gpus = torch.cuda.device_count() if args.gpus == -1 else args.gpus

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp" if num_gpus > 1 else "auto",
        precision="16-mixed",
        logger=comet_logger,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_val_batches=0 if args.disable_validation else args.limit_val_batches,
    )

    # Validation before training (sanity check doesn't record logs, so run validation directly)
    data_module.setup(stage='fit')
    lit_model.initial_eval = True
    trainer.validate(lit_model, datamodule=data_module)

    lit_model.initial_eval = False
    trainer.fit(lit_model, datamodule=data_module)


if __name__ == "__main__":
    main()
