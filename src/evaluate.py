from logger import configure_logger
import argparse
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from inference import VideoOutpaintingInference
from skimage.metrics import structural_similarity as ssim  # pylint: disable=no-name-in-module
import torch
import lpips
from scipy.ndimage import gaussian_filter


# Suppress cv2 errors
cv2.setLogLevel(0)


def assemble_video_ffmpeg(frames_dir: Path, output_path: Path, fps: float = 24):
    """Generate MP4 (H.264) video from frame sequence using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%04d.jpg"),
        "-c", "copy",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_video_metrics(original_video_path: Path, generated_video_path: Path, lpips_model=None) -> tuple[float, float, float, float, float]:
    """Compute MSE, PSNR, SSIM, LPIPS, BMSE (average across all frames) for two videos at once."""
    cap_original = cv2.VideoCapture(str(original_video_path))
    cap_generated = cv2.VideoCapture(str(generated_video_path))
    assert cap_original.isOpened(), f"Failed to open original video: {original_video_path}"
    assert cap_generated.isOpened(), f"Failed to open generated video: {generated_video_path}"

    frame_count = int(min(cap_original.get(cv2.CAP_PROP_FRAME_COUNT),
                          cap_generated.get(cv2.CAP_PROP_FRAME_COUNT)))

    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    bmse_sum = 0.0

    # Initialize LPIPS model if not provided
    if lpips_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lpips_model = lpips.LPIPS(net='alex').to(device)

    device = next(lpips_model.parameters()).device

    for _ in range(frame_count):
        ret_o, frame_o = cap_original.read()
        ret_g, frame_g = cap_generated.read()
        if not (ret_o and ret_g):
            break
        # Normalize to 0-1 range before computing MSE, PSNR, SSIM
        frame_o = frame_o.astype(np.float32) / 255.0
        frame_g = frame_g.astype(np.float32) / 255.0
        mse = np.mean((frame_o - frame_g) ** 2)
        mse_sum += mse

        # Compute PSNR
        if mse == 0:
            psnr = float('inf')  # Perfect match
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))  # PSNR for 0-1 normalized images
        psnr_sum += psnr

        # Compute SSIM (averaged across channels)
        ssim_value = ssim(frame_o, frame_g, channel_axis=2, data_range=1.0)
        ssim_sum += ssim_value

        # Compute LPIPS (convert to PyTorch tensor, normalize to [-1, 1] range)
        frame_o_tensor = torch.from_numpy(frame_o).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
        frame_g_tensor = torch.from_numpy(frame_g).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
        # Convert [0, 1] -> [-1, 1] (LPIPS requirement)
        frame_o_tensor = frame_o_tensor * 2.0 - 1.0
        frame_g_tensor = frame_g_tensor * 2.0 - 1.0

        with torch.no_grad():
            lpips_value = lpips_model(frame_o_tensor, frame_g_tensor).item()
        lpips_sum += lpips_value

        # Compute BMSE (MSE before/after blur on generated video)
        # If generated result is blurry, applying blur causes small change
        # Apply Gaussian Blur to generated video
        frame_g_blurred = gaussian_filter(frame_g, 5)
        bmse = np.mean((frame_g - frame_g_blurred) ** 2)
        bmse_sum += bmse

    cap_original.release()
    cap_generated.release()

    avg_mse = mse_sum / max(frame_count, 1)
    avg_psnr = psnr_sum / max(frame_count, 1)
    avg_ssim = ssim_sum / max(frame_count, 1)
    avg_lpips = lpips_sum / max(frame_count, 1)
    avg_bmse = bmse_sum / max(frame_count, 1)

    return avg_mse, avg_psnr, avg_ssim, avg_lpips, avg_bmse


class VideoOutpaintingEvaluator:
    """Generate outpainting for videos in directory and evaluate with MSE, PSNR, SSIM, LPIPS."""

    def __init__(
        self,
        video_dir: str,
        output_root: str,
        pretrained_sd_dir: str,
        video_outpainting_model_dir: str,
        crop_ratio: float,
        crop_axis: str,
        enable_attention_slicing: bool = False,
        gpu_no: int = 0,
        target_ratio_list: str = "16:9",
        output_size: int = 256,
        seed: int = 6,
        num_global_frames: int = 16,
        limit_outpainting_frames: int = 16,
        use_first_frame_only: bool = False,
        strides: list = None,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.runner = VideoOutpaintingInference(
            gpu_no=gpu_no,
            pretrained_sd_dir=pretrained_sd_dir,
            output_dir=str(self.output_root),
            target_ratio_list=target_ratio_list,
            video_outpainting_model_dir=video_outpainting_model_dir,
            seed=seed,
            num_global_frames=num_global_frames,
            limit_outpainting_frames=limit_outpainting_frames,
            disable_comet=True,
            enable_attention_slicing=enable_attention_slicing,
            output_size=output_size,
            use_first_frame_only=use_first_frame_only,
            strides=strides if strides is not None else [5, 3, 1],
        )
        self.num_global_frames = num_global_frames
        self.limit_outpainting_frames = limit_outpainting_frames

        self.output_size = output_size
        self.crop_ratio = crop_ratio
        self.crop_axis = crop_axis  # "horizontal" or "vertical"

        # Initialize LPIPS model
        device = torch.device(f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self._frame_info_printed = False

    # Set long edge = output_size, short edge = target_ratio, center crop so short edge is multiple of 8
    def resize_long_edge(self, frame):
        h_org, w_org = frame.shape[:2]

        # Assumes target_ratio_list contains only one ratio (e.g., "16:9")
        width_ratio, height_ratio = list(map(float, self.runner.target_ratio_list.split(":")[0].split(",")))[0], \
            float(self.runner.target_ratio_list.split(":")[1])
        ratio = width_ratio / height_ratio

        if ratio >= 1:
            w_tgt = self.output_size
            h_tgt = int(round(w_tgt / ratio))
        else:
            h_tgt = self.output_size
            w_tgt = int(round(h_tgt * ratio))

        # Center crop if short edge is not multiple of 8
        if h_tgt % 8:
            h_tgt -= h_tgt % 8
        if w_tgt % 8:
            w_tgt -= w_tgt % 8

        # Scale then center crop
        scale = max(w_tgt / w_org, h_tgt / h_org)
        resized = cv2.resize(frame,
                             (int(w_org * scale), int(h_org * scale)),
                             cv2.INTER_AREA)
        h_res, w_res = resized.shape[:2]
        top = (h_res - h_tgt) // 2
        left = (w_res - w_tgt) // 2
        return resized[top: top + h_tgt, left: left + w_tgt]

    def apply_crop(self, frame):
        """Crop outpainting region (if shorter edge is not divisible by 8, trim equally from top/bottom or left/right)."""
        h, w = frame.shape[:2]
        if self.crop_axis == "horizontal":
            cut = int(w * self.crop_ratio / 2)
            return frame[:, cut:w - cut]
        else:  # vertical
            cut = int(h * self.crop_ratio / 2)
            return frame[cut:h - cut]

    def save_frames(self, frames, dir_path: Path):
        if not self._frame_info_printed:
            for idx, img in enumerate(frames):
                h, w = img.shape[:2]
                if idx == 0:
                    print(f"Frame {idx:04d} size: {w}x{h}")
            self._frame_info_printed = True

        dir_path.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(frames):
            cv2.imwrite(str(dir_path / f"f{idx:04d}.jpg"), img)

    def prepare_frames(self, src_video: Path, tmp_dir: Path):
        cap = cv2.VideoCapture(str(src_video))
        assert cap.isOpened(), f"open failed: {src_video}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        resized_frames, cropped_frames = [], []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            resized = self.resize_long_edge(frm)
            resized_frames.append(resized)
            cropped_frames.append(self.apply_crop(resized))
        cap.release()

        dir_r = tmp_dir / "resized_frames"
        dir_c = tmp_dir / "cropped_frames"
        self.save_frames(resized_frames, dir_r)
        self.save_frames(cropped_frames, dir_c)

        resized_mp4 = tmp_dir / "resized.mp4"
        cropped_mp4 = tmp_dir / "cropped.mp4"
        assemble_video_ffmpeg(dir_r, resized_mp4, fps)
        assemble_video_ffmpeg(dir_c, cropped_mp4, fps)
        return resized_mp4, cropped_mp4

    def outpaint_one_video(self, original_video_path: Path):
        clip_dir = self.output_root / original_video_path.stem
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        clip_dir.mkdir(parents=True)

        # Resize and crop
        resized_mp4, cropped_mp4 = self.prepare_frames(original_video_path, clip_dir)

        # Outpainting (input cropped video)
        saved_dir = self.runner.output_dir
        self.runner.output_dir = str(clip_dir)
        try:
            self.runner.outpaint(str(cropped_mp4))
        finally:
            self.runner.output_dir = saved_dir

        # Generated frames -> MP4
        generated_mp4 = clip_dir / f"cropped_{self.runner.target_ratio_list.replace(':', '-')}.mp4"
        return resized_mp4, generated_mp4

    def evaluate_and_log_metrics(self, experiment, epoch=0):
        videos = sorted(self.video_dir.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 files found in {self.video_dir}")

        # Metric names list (corresponds to compute_video_metrics return value order)
        metric_names = ["MSE", "PSNR", "SSIM", "LPIPS", "BMSE"]
        totals = {name: 0.0 for name in metric_names}
        processed_count = 0
        metric_prefix = "init-outpainting" if epoch == -1 else "outpainting"

        for idx, video_path in enumerate(tqdm(videos, desc="Evaluating")):
            try:
                resized_video, generated_video = self.outpaint_one_video(video_path)
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
                logging.error("%s failed: %s", video_path, e)
                continue

            # Compute metrics
            metric_values = compute_video_metrics(resized_video, generated_video, self.lpips_model)

            # Accumulate each metric and log
            step = epoch * len(videos) + idx
            for name, value in zip(metric_names, metric_values):
                totals[name] += value
                experiment.log_metric(f"{metric_prefix}/{name}_step", value, step=step)

            processed_count += 1
            video_name = f"{video_path.stem}_base.mp4" if epoch == -1 else f"{video_path.stem}_{epoch}.mp4"
            experiment.log_video(generated_video, name=video_name)

        # Compute epoch average and log
        count = max(processed_count, 1)
        averages = {name: totals[name] / count for name in metric_names}

        for name, avg in averages.items():
            experiment.log_metric(f"{metric_prefix}/{name}_epoch", avg, step=epoch)

        # Display results
        avg_strs = [f"Avg {name}: {averages[name]:.6f}" for name in metric_names]
        print(f"{', '.join(avg_strs)} (videos: {processed_count})")


def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluate videos in directory with MSE, PSNR, SSIM, LPIPS")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing original videos to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated outputs (defaults to execution timestamp)")
    parser.add_argument("--pretrained_sd_dir", type=str, required=True)
    parser.add_argument("--video_outpainting_model_dir", type=str, required=True)
    parser.add_argument("--gpu_no", type=int, default=0)
    parser.add_argument("--target_ratio_list", type=str, default="16:9", help="Must match the aspect ratio of the original video")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument('--num_global_frames', type=int, default=16, help='Number of global frames for outpainting')
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--crop_ratio", type=float, default=0.25, help="Ensure length after cropping outpainting region is divisible by 8")
    parser.add_argument("--crop_axis", choices=["horizontal", "vertical"], default="horizontal", help="Whether to outpaint vertically or horizontally")
    parser.add_argument("--disable_comet", "-dc", action="store_true", help="Disable Comet logging")
    parser.add_argument("--enable_attention_slicing", action="store_true", help="Reduce GPU memory during generation, slightly increases generation time")
    parser.add_argument('--limit_outpainting_frames', type=int, default=16, help="Number of frames to generate in evaluate, -1 for all frames")
    parser.add_argument('--use_first_frame_only', action='store_true', default=False, help='Copy only the first frame for the entire video length')
    parser.add_argument('--strides', type=str, default='5,3,1', help='Strides for frame processing (comma-separated), e.g., "5,3,1"')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_root = os.path.join("evaluation", args.output_dir or datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    # TODO: Move into output/

    experiment = configure_logger(
        logged_params=vars(args),
        model_name='evaluation',
        disable_logging=args.disable_comet,
        lightning=False,
    )

    evaluator = VideoOutpaintingEvaluator(
        video_dir=args.video_dir,
        output_root=output_root,
        pretrained_sd_dir=args.pretrained_sd_dir,
        video_outpainting_model_dir=args.video_outpainting_model_dir,
        crop_ratio=args.crop_ratio,
        crop_axis=args.crop_axis,
        enable_attention_slicing=args.enable_attention_slicing,
        gpu_no=args.gpu_no,
        target_ratio_list=args.target_ratio_list,
        output_size=args.output_size,
        seed=args.seed,
        num_global_frames=args.num_global_frames,
        limit_outpainting_frames=args.limit_outpainting_frames,
        use_first_frame_only=args.use_first_frame_only,
        strides=[int(s) for s in args.strides.split(',')],
    )

    start_time = time.time()
    evaluator.evaluate_and_log_metrics(experiment)
    experiment.log_metric("evaluation_time_sec", time.time() - start_time)


if __name__ == "__main__":
    main()
