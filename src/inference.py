# The code is based on https://github.com/alimama-creative/M3DDM-Video-Outpainting
from logger import configure_logger
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pipelines.stable_diffusion.pipeline_stable_diffusion_video2video_mask_for_video_outpainting import StableDiffusionPipelineVideo2VideoMaskC
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from model.unet_3d_condition_video import UNet3DConditionModel
import argparse
import math
import os
import sys
import logging
import torch
from PIL import Image, ImageFilter
import cv2
import decord
import glob
import numpy as np
from torchvision import transforms as T
import torch.nn.functional as F
from einops import rearrange
import traceback
import time
from decord import VideoReader
import logging

util_logger = logging.getLogger(__name__)

decord.bridge.set_bridge('torch')


class RetryTask(object):
    """
    Base class for managing task retries.

    Provides functionality to retry task execution a specified number of times
    when failures occur. Primarily used to improve error tolerance during parallel processing.

    Attributes:
        retries (int): Maximum number of retry attempts on failure
        raise_if_fail (bool): Whether to raise exception if all retries fail
    """

    def __init__(self, retries, raise_if_fail=True):
        self.retries = retries
        self.raise_if_fail = raise_if_fail

    def run(self):
        """
        Method to be overridden by subclasses for actual processing.
        """
        pass

    def retry_run(self):
        """
        Attempt to execute the task up to the specified number of retries.

        Returns:
            Execution result, or exception object on failure (if raise_if_fail=False)
        """
        for i in range(self.retries):
            try:
                return self.run()
            except Exception as e:
                traceback.print_exc()
                util_logger.error(
                    "RetryTask run exception: {exc}, try times {time}".format(
                        exc=e, time=i))
                if i == self.retries - 1:
                    if self.raise_if_fail:
                        raise
                    else:
                        return e


def async_thread_tasks(retry_tasks, max_thread_num=16):
    """
    Execute a list of RetryTasks in parallel.

    Uses ThreadPoolExecutor to execute multiple tasks in parallel.
    Suitable for efficient execution of many independent tasks like image processing.

    Args:
        retry_tasks (list): List of RetryTask objects to execute
        max_thread_num (int): Maximum number of threads to use (default: 16)

    Returns:
        list: List of execution results for each task
    """
    if len(retry_tasks) < 1:
        return []
    if max_thread_num < 1 or max_thread_num > len(retry_tasks):
        max_thread_num = len(retry_tasks)
    with ThreadPoolExecutor(max_thread_num) as thread_pool:
        worker_list = [
            thread_pool.submit(task.retry_run) for task in retry_tasks
        ]
        result = [task.result() for task in worker_list]
        return result


def save_images(output_dir, images, sample_id, stride=1):
    """
    Save generated frame images with sequential numbering.

    Args:
        output_dir (str): Output directory path
        images (list): List of PIL.Image objects
        sample_id (int): Starting frame number
        stride (int): Number increment between frames (default: 1)
    """
    test_dir = os.path.join(output_dir, "frames")
    os.makedirs(test_dir, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{test_dir}/f{sample_id + i * stride:04d}.jpg")


def preprocess(image, img_transform):
    """
    Apply preprocessing to image tensor.

    Args:
        image (torch.Tensor): Image tensor to process
        img_transform (torchvision.transforms): Transform to apply

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = img_transform(image)
    return image


# Not referenced anywhere, probably unused
# merge_frame_to_video_with_ratio is used instead
def merge_frame_to_video(frame_dir, fps, output_video_path):
    """
    Generate video from sequential frame images.

    Args:
        frame_dir (str): Directory path containing frame images
        fps (float): Output video frame rate
        output_video_path (str): Output video file path
    """
    frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frame_paths = sorted(frame_paths)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    all_frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        all_frames.append(frame)

    new_size = (all_frames[0].shape[1], all_frames[0].shape[0])
    new_video = cv2.VideoWriter(output_video_path, fourcc, fps, new_size)
    for high_res_frame in all_frames:
        new_video.write(high_res_frame)
    new_video.release()


def resize_frames_to_output_size(frames, output_size, target_ratio_list):
    """
    Resize frames so that the long edge fits output_size while maintaining aspect ratio.

    This function calculates the required canvas size based on target aspect ratios,
    then resizes all frames proportionally to fit within that canvas.

    Args:
        frames (torch.Tensor): Input frames [N, C, H, W]
        output_size (int): Maximum size for long edge
        target_ratio_list (str): Comma-separated target aspect ratios (e.g., "16:9")

    Returns:
        torch.Tensor: Resized frames [N, C, new_H, new_W]
    """
    _, _, h, w = frames.shape

    # Parse target ratio list
    ratios = target_ratio_list.split(",")
    target_max_size = get_max_target_size((h, w), ratios)

    # Calculate scale to fit output_size
    max_dim = max(target_max_size)
    scale = output_size / max_dim

    # Calculate new size maintaining aspect ratio
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Use torchvision transforms for resizing
    resized_frames = T.Resize((new_h, new_w), antialias=True)(frames)

    return resized_frames

def get_max_target_size(cur_shape, target_ratio_list):
    """
    Calculate maximum size to accommodate multiple aspect ratios.

    From the current image size, calculates the maximum height and width
    that can accommodate all specified aspect ratios.

    Args:
        cur_shape (tuple): Current image (height, width)
        target_ratio_list (list): List of target aspect ratios (e.g., ['9:16', '16:9'])

    Returns:
        tuple: Required (max_height, max_width)
    """
    h, w = cur_shape
    cur_ratio = w / h
    max_h, max_w = h, w
    for target_ratio in target_ratio_list:
        target_ratio = float(target_ratio.split(":")[0]) / float(
            target_ratio.split(":")[1])
        if cur_ratio < target_ratio:  # Extend width if width is insufficient
            max_w = max(max_w, target_ratio * h)
        else:  # Extend height if height is insufficient
            max_h = max(max_h, w / target_ratio)
    return (max_h, max_w)


def preprocess_frames_speed_square(frames,
                                   target_ratio_list="3:4,1:1,2:3,9:16",
                                   max_size=256):
    """
    Preprocess frames for outpainting.

    Parallel processes multiple frames, resizes and pads to accommodate
    specified aspect ratios, prepares center as original and surrounding as generation mask.

    Args:
        frames (torch.Tensor): Input frames to preprocess [num_frames, channels, height, width]
        target_ratio_list (str): Comma-separated target aspect ratio list (e.g., "3:4,1:1,2:3,9:16")
        max_size (int): Maximum size of long edge for output frames (default: 256)

    Returns:
        tuple: (preprocessed_frames, mask, masked_frames)
    """
    res = []

    class preprocess_single_frame_maxsize(RetryTask):
        """
        RetryTask subclass for preprocessing a single frame.

        Resizes and pads frame, creates mask (1=region to generate, 0=region to keep).
        """

        def __init__(self, frame, index, target_ratio, target_max_size):
            super().__init__(retries=1)
            self.frame = frame
            self.target_ratio = target_ratio
            self.index = index
            self.target_max_size = target_max_size

        def run(self):
            # Get frame height and width (already resized by caller)
            _, h, w = self.frame.shape
            frame_np = self.frame.permute(1, 2, 0).numpy()
            target_height, target_width = map(int, self.target_max_size)
            max_dim = max(target_height, target_width)

            new_size = (max_dim, max_dim)

            # Use input size directly (no resizing here - caller is responsible for resizing)
            frame_np = frame_np.astype(np.uint8)

            # Place frame at center of new canvas
            pad_h = (new_size[1] - h) // 2
            pad_w = (new_size[0] - w) // 2
            frame_draw = np.zeros((new_size[1], new_size[0], 3),
                                  dtype=np.uint8)
            frame_draw[pad_h:pad_h + h, pad_w:pad_w + w] = frame_np

            # Create mask (0=original image region, 1=region to generate)
            mask = np.ones((new_size[1], new_size[0]), dtype=np.uint8)
            mask[pad_h:pad_h + h, pad_w:pad_w + w] = 0

            # Inpaint surrounding area for natural seam at boundaries
            frame_inpaint = cv2.inpaint(frame_draw, mask, 5, cv2.INPAINT_TELEA)

            return [self.index, frame_inpaint, mask, frame_draw]

    # Split aspect ratio list
    target_ratio_list = target_ratio_list.split(",")
    _, h, w = frames[0].shape
    # Calculate maximum canvas size to accommodate all aspect ratios
    # Input frames are expected to be already resized by the caller
    target_max_size = get_max_target_size((h, w), target_ratio_list)

    target_ratio = target_max_size[1] / target_max_size[0]
    # Create processing task for each frame
    tasks = []
    for i in range(len(frames)):
        tasks.append(
            preprocess_single_frame_maxsize(
                frames[i], i, target_ratio, target_max_size
            )
        )
    # Execute parallel processing (max_thread_num=1, so not actually parallelized)
    result = async_thread_tasks(tasks, max_thread_num=1)
    # Sort by index
    result = sorted(result, key=lambda x: x[0])

    # Format processing results as tensors
    # Preprocessed frames
    frames_inpaint = np.stack([r[1] for r in result], axis=0)
    frames_inpaint_tensor = torch.from_numpy(frames_inpaint).float()
    frames_inpaint_tensor = frames_inpaint_tensor.permute(0, 3, 1,
                                                          2)  # f, c, h, w
    masks = np.stack([r[2] for r in result], axis=0)
    masks_tensor = torch.from_numpy(masks).float().unsqueeze(0)  # [1, n, h, w]
    masked_frames = np.stack([r[3] for r in result], axis=0)
    masked_frames = torch.from_numpy(masked_frames).float()
    masked_frames_tensor = masked_frames.permute(0, 3, 1, 2)  # f, c, h, w

    return frames_inpaint_tensor, masks_tensor, masked_frames_tensor


def merge_frame_to_video_with_ratio(frame_dir, fps, output_video_path, target_ratios, original_frame_count=None):
    """
    Generate videos with multiple aspect ratios from generated frames.

    This function generates video files with multiple different aspect ratios
    (e.g., 9:16, 16:9) from frame images in the specified directory. For each
    aspect ratio, it appropriately crops the original image to create a new video.

    Additionally, it attempts high-quality encoding using FFmpeg in addition to
    OpenCV's standard encoder. If FFmpeg is available, it performs more
    compatible H.264 encoding.

    Args:
        frame_dir (str): Directory path containing frame images
        fps (float): Output video frame rate
        output_video_path (str): Base output video file path (aspect ratio info appended)
        target_ratios (str/list): Aspect ratios for videos to generate (e.g., '9:16,16:9' or ['9:16', '16:9'])
        original_frame_count (int, optional): Original video frame count. If specified, limits output frames to this value
    """
    frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frame_paths = sorted(frame_paths)
    all_frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Trim to original frame count if specified
    if original_frame_count is not None and len(all_frames) > original_frame_count:
        print(f"Trimming output frames from {len(all_frames)} to {original_frame_count} to match original video length")
        all_frames = all_frames[:original_frame_count]

    # Convert string format aspect ratio list to list
    if isinstance(target_ratios, str):
        target_ratios = target_ratios.split(',')

    # Generate video for each aspect ratio
    for target_ratio in target_ratios:
        # Convert aspect ratio string (e.g., '9:16') to numeric values
        ratio_parts = target_ratio.split(':')
        target_width_ratio = int(ratio_parts[0])
        target_height_ratio = int(ratio_parts[1])

        # Calculate required crop range based on target aspect ratio
        orig_height, orig_width = all_frames[0].shape[:2]
        target_aspect = target_width_ratio / target_height_ratio

        # Calculate appropriate crop range based on original image aspect ratio
        if (orig_width / orig_height) > target_aspect:
            # Crop width if too wide
            new_width = int(orig_height * target_aspect)
            margin = int((orig_width - new_width) / 2)
            crop_img = (margin, 0, new_width + margin, orig_height)
        else:
            # Crop height if too tall
            new_height = int(orig_width / target_aspect)
            margin = int((orig_height - new_height) / 2)
            crop_img = (0, margin, orig_width, new_height + margin)

        # Create video file with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        video_filename = f"{os.path.splitext(output_video_path)[0]}_{target_ratio.replace(':', '-')}.mp4"

        # Calculate output size
        output_width = crop_img[2] - crop_img[0]
        output_height = crop_img[3] - crop_img[1]

        # Initialize OpenCV VideoWriter
        new_video = cv2.VideoWriter(
            video_filename, fourcc, fps,
            (output_width, output_height)
        )

        # Fall back to mp4v if H.264 codec is not available
        if not new_video.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            new_video = cv2.VideoWriter(
                video_filename, fourcc, fps,
                (output_width, output_height)
            )

        # Crop each frame and add to video
        for high_res_frame in all_frames:
            # Crop frame based on calculated crop range
            cropped_frame = high_res_frame[crop_img[1]:crop_img[3],
                                           crop_img[0]:crop_img[2]]
            new_video.write(cropped_frame)
        new_video.release()

        # If FFmpeg is available, perform higher quality and browser-compatible encoding
        try:
            import subprocess
            temp_file = f"{video_filename}_temp.mp4"
            os.rename(video_filename, temp_file)
            cmd = [
                'ffmpeg', '-i', temp_file,
                '-c', 'copy',
                video_filename
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_file)
            print(
                f"Re-encoded {video_filename} with FFmpeg for better compatibility")
        except Exception as e:
            print(
                f"FFmpeg re-encoding failed: {e}. Using original video file.")
            # Restore original file if FFmpeg processing failed
            if os.path.exists(temp_file):
                os.rename(temp_file, video_filename)


class VideoOutpaintingInference:
    """
    Video outpainting processing class.
    """

    def __init__(self, gpu_no, pretrained_sd_dir, output_dir, target_ratio_list, video_outpainting_model_dir, seed, num_global_frames, limit_outpainting_frames, disable_comet, enable_attention_slicing, output_size, use_first_frame_only=False, strides=None):
        self.gpu_no = gpu_no
        self.pretrained_sd_dir = pretrained_sd_dir
        self.limit_outpainting_frames = limit_outpainting_frames
        self.output_dir = output_dir
        self.target_ratio_list = target_ratio_list
        self.video_outpainting_model_dir = video_outpainting_model_dir
        self.seed = seed
        self.num_global_frames = num_global_frames
        self.disable_comet = disable_comet
        self.enable_attention_slicing = enable_attention_slicing
        self.output_size = output_size
        self.use_first_frame_only = use_first_frame_only
        self.strides = strides if strides is not None else [15, 5, 1]

        # Output directory and save arguments
        import json
        from datetime import datetime
        exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.exp_path = os.path.join(self.output_dir, exp_name)
        os.makedirs(self.exp_path, exist_ok=True)
        args_json = {
            "gpu_no": self.gpu_no,
            "pretrained_sd_dir": self.pretrained_sd_dir,
            "output_dir": self.output_dir,
            "target_ratio_list": self.target_ratio_list,
            "video_outpainting_model_dir": self.video_outpainting_model_dir,
            "seed": self.seed,
            "num_global_frames": self.num_global_frames,
            "limit_outpainting_frames": self.limit_outpainting_frames,
            "disable_comet": self.disable_comet,
            "enable_attention_slicing": self.enable_attention_slicing,
            "output_size": self.output_size,
            "use_first_frame_only": self.use_first_frame_only,
            "strides": self.strides
        }
        with open(os.path.join(self.exp_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_json, f, indent=4)

        self.experiment = configure_logger(
            logged_params=args_json,
            model_name='inference',
            disable_logging=self.disable_comet,
            lightning=False,
        )

        # Logging configuration
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        # Device and dtype configuration
        self.device = f"cuda:{self.gpu_no}"
        self.weight_dtype = torch.float16

        # Initialize scheduler
        self.noise_scheduler = PNDMScheduler.from_pretrained(self.pretrained_sd_dir, subfolder="scheduler", local_files_only=True)
        self.scheduler_pre = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # Load models
        self.vae = AutoencoderKL.from_pretrained(os.path.join(self.pretrained_sd_dir, 'vae'), local_files_only=True)
        self.unet = UNet3DConditionModel.from_pretrained(self.video_outpainting_model_dir, local_files_only=True, torch_dtype=self.weight_dtype)

        # Create pipeline
        pipeline = StableDiffusionPipelineVideo2VideoMaskC(
            vae=self.vae.to(self.weight_dtype).to(self.device).eval(),
            unet=self.unet.to(self.weight_dtype).to(self.device).eval(),
            scheduler=self.noise_scheduler,
            scheduler_pre=self.scheduler_pre
        )
        # GPU memory reduction settings
        if self.enable_attention_slicing:
            pipeline.enable_attention_slicing()

        self.all_pipelines = [pipeline]

    def outpaint(self, input_video_path):
        """
        Execute outpainting processing for the specified video path.
        """
        path = input_video_path
        print("input_video_path:", path)
        if self.use_first_frame_only:
            print("Mode: Copy only first frame for all frames")
        else:
            print("Mode: Normal video outpainting")
        os.makedirs(self.output_dir, exist_ok=True)

        vr = VideoReader(open(path, 'rb'))
        fps = vr.get_avg_fps()
        total_source_frames = len(vr)
        copy_times = 1
        total_frames_num = total_source_frames * copy_times
        select_indexes = list(range(total_frames_num))

        # Frame count limit settings
        frames_processed = 0
        # If -1, process all frames
        if self.limit_outpainting_frames < 0:
            max_frames = total_frames_num
        else:
            max_frames = self.limit_outpainting_frames

        # Global frame selection and other processing
        num_global_frames = self.num_global_frames
        selected_global_frames_idx = np.linspace(0, total_frames_num - 1, num_global_frames, dtype=int)
        print('global frames:', selected_global_frames_idx)

        print(
            '\'*\' indicates that the frame has been filled, \'|\' indicates it has not been filled.'
        )

        if total_frames_num < 50 or max_frames < total_frames_num:
            strides = [1]
        else:
            strides = self.strides

        all_latents = [None] * total_frames_num

        for stride in strides:
            # Break if limit reached
            if frames_processed >= max_frames:
                break
            stride_indexes = select_indexes[::stride]
            for start_frame in range(0, (total_frames_num // stride) - 1, 15):
                # Break if limit reached
                if frames_processed >= max_frames:
                    break
                 # Frame indices to process in current batch (max 16 frames)
                selected_frame_indexes = stride_indexes[start_frame:start_frame + 16]

                # Pad with last frame if less than 16 frames
                if len(selected_frame_indexes) < 16:
                    selected_frame_indexes = selected_frame_indexes + [
                        select_indexes[-1]
                    ] * (16 - len(selected_frame_indexes))
                    assert len(selected_frame_indexes) == 16
                # Update batch frame count
                batch_count = len(selected_frame_indexes)
                if stride == 1:
                    frames_processed += batch_count
                print(selected_frame_indexes, 'out of', total_frames_num, 'frames')

                # Load selected frames and convert to tensor format
                if self.use_first_frame_only:
                    # Load only first frame and duplicate for batch size
                    first_frame = vr.get_batch([0]).permute(0, 3, 1, 2).float()  # [1, C, H, W]
                    frames = first_frame.repeat(len(selected_frame_indexes), 1, 1, 1)  # [N, C, H, W]
                else:
                    frames = vr.get_batch(
                        list(map(lambda x: x % total_source_frames,
                                 selected_frame_indexes))).permute(0, 3, 1, 2).float()

                # Load global frames
                if self.use_first_frame_only:
                    # Use first frame for all global frames
                    first_frame = vr.get_batch([0]).permute(0, 3, 1, 2).float()  # [1, C, H, W]
                    global_frames = first_frame.repeat(len(selected_global_frames_idx), 1, 1, 1)  # [N, C, H, W]
                else:
                    global_frames = vr.get_batch(
                        list(
                            map(lambda x: x % total_source_frames,
                                selected_global_frames_idx))).permute(0, 3, 1,
                                                                      2).float()

                # Transform for normalization
                # img_transform = T.Compose([
                #     # Normalize to [-1, 1] range
                #     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                # ])

                # Global frame preprocessing (resize and normalize)
                global_frames = preprocess(
                    global_frames / 255.0,  # Convert to [0, 1] range
                    T.Compose([
                        T.Resize((self.output_size, self.output_size), antialias=True),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]))

                # Resize frames to fit output_size before outpainting preprocessing
                frames = resize_frames_to_output_size(
                    frames,
                    output_size=self.output_size,
                    target_ratio_list=self.target_ratio_list
                )

                # Preprocess selected frames for outpainting
                frames_inpaint_tensor, mask, masked_frames = preprocess_frames_speed_square(
                    frames,
                    target_ratio_list=self.target_ratio_list,
                    max_size=self.output_size
                )

                # Initial frame preprocessing
                init_frames = preprocess(
                    frames_inpaint_tensor / 255.,
                    T.Compose([
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]))
                # Convert tensor dimensions ([channels, num_frames, height, width])
                init_frames = rearrange(init_frames, "f c h w ->c f h w")

                # Masked frame preprocessing
                masked_frames = preprocess(
                    masked_frames / 255.,
                    T.Compose([
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]))
                # Save original masked frames to CPU (used for composition later)
                ori_masked_frames = masked_frames.clone().cpu().numpy()
                # Convert tensor dimensions ([channels, num_frames, height, width])
                masked_frames = rearrange(masked_frames, "f c h w ->c f h w")

                # Transfer all tensors to GPU (convert to specified dtype and device)
                init_frames = init_frames.to(self.weight_dtype).to(
                    'cuda:{}'.format(self.gpu_no))
                masked_frames = masked_frames.to(self.weight_dtype).to(
                    'cuda:{}'.format(self.gpu_no))
                mask = mask.to(self.weight_dtype).to('cuda:{}'.format(self.gpu_no))
                global_frames = global_frames.to(self.weight_dtype).to(
                    'cuda:{}'.format(self.gpu_no))

                if num_global_frames == 0:
                    global_frames = None

                # Set random generator for reproducibility
                generator = torch.Generator(
                    device='cuda:{}'.format(self.gpu_no)).manual_seed(self.seed)

                # Execute pipeline for video outpainting
                # ori_image: Original input frame tensor
                # use_add_noise: Whether to apply additional noise (for processing diversity)
                # mask_image: Masked frame tensor
                # mask: Actual binary mask (0=retain region, 1=generate region)
                videos, decoded_latents = self.all_pipelines[
                    0].outpainting_with_random_masked_latent_inference_bidirection(
                        ori_image=init_frames,
                        use_add_noise=True,  # Add noise for generation diversity
                        mask_image=masked_frames,
                        mask=mask[:, :1, :, :],  # Use only first channel of mask
                        cond=True,  # Enable conditional generation
                        strength=1.0,  # Generation strength (1.0 = complete regeneration)
                        batch_size=1,  # Batch size (adjustable based on GPU memory)
                        num_frames=16,  # Number of frames to process
                        height=init_frames.shape[-2],  # Frame height
                        width=init_frames.shape[-1],  # Frame width
                        fps=fps // stride,  # Frame rate to process (adjusted by stride)
                        generator=generator,  # Random generator for reproducibility
                        latents_dtype=self.weight_dtype,  # Latent variable dtype (memory efficiency)
                        num_inference_steps=50,  # Diffusion model inference steps (more = higher quality but slower)
                        mask_ratio=6 / 16,  # Training mask ratio
                        copy_raw_images=False,  # Don't copy raw images
                        guidance_scale=2,  # Classifier-free guidance scale (higher = more faithful, lower = more diverse)
                        previous_guidance_scale=4.0,  # Previous frame guidance scale
                        cur_step=start_frame // 16,  # Current processing step
                        noise_level=100 if stride > 1 else 0,  # Add noise for large stride to handle big changes
                        already_outpainted_latents=[
                            all_latents[i_frame]  # Latent variables of already generated frames
                            for i_frame in selected_frame_indexes
                        ],
                        copy_already_frame=False,  # Don't copy existing frames
                        global_frames=global_frames,  # Global frames for overall consistency
                        num_global_frames=num_global_frames  # Number of global frames
                )  # Execute bidirectional outpainting on multiple frames

                # Process each generated video
                for k in range(len(videos)):
                    images = videos[k].images
                    temp_images = []
                    # Composite original center with generated outer region for each frame
                    for idx in range(len(images)):
                        # Process original image region (unmasked area)
                        original_region = (((ori_masked_frames[idx] / 2) + 0.5) *
                             255) * (1 - mask.cpu().numpy())[0, 0]
                        # Process generated image region (masked area)
                        generated_region = np.asarray(images[idx]).transpose(
                            2, 0, 1) * mask.cpu().numpy()[0, 0]

                        # Composite both and save as PIL image
                        # temp_images.append(
                        #     Image.fromarray((a + b).transpose(1, 2,
                        #                                       0).astype(np.uint8)))

                        # Composite both into Numpy array
                        combined_image_np = (original_region + generated_region).transpose(1, 2, 0).astype(np.uint8)

                        # # Use the entire generated image without compositing with original
                        # combined_image_np = np.array(images[idx])

                        # Exclude boundaries when computing MSE
                        # Find and draw boundaries from mask
                        # mask_for_border = (mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                        # contours, _ = cv2.findContours(mask_for_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # # Draw detected contours in red
                        # cv2.drawContours(combined_image_np, contours, -1, (255, 0, 0), 1)

                        # Save as PIL image
                        temp_images.append(Image.fromarray(combined_image_np))
                    # Save composited frames to specified directory
                    save_images(self.output_dir,
                                images=temp_images,
                                sample_id=selected_frame_indexes[0],
                                stride=stride)

                    # Save generated frame latent variables for use in next processing
                    for frame_idx in range(len(images)):
                        if all_latents[selected_frame_indexes[frame_idx]] == None:
                            # Save latent variables for frames not yet processed
                            all_latents[selected_frame_indexes[
                                frame_idx]] = decoded_latents[:, :, frame_idx, :, :]
                        else:
                            # Skip already processed frames
                            pass

        # After all frame processing is complete, generate videos with specified multiple aspect ratios
        # Pass original frame count to ensure output video length matches original
        merge_frame_to_video_with_ratio(
            os.path.join(self.output_dir, 'frames'),
            fps,  # Original video frame rate
            os.path.join(self.output_dir, f"{os.path.basename(path).split('.')[0]}.mp4"),
            self.target_ratio_list,
            original_frame_count=total_source_frames  # Pass original frame count
        )
        # Log to Comet
        for video_file in glob.glob(os.path.join(self.output_dir, '*.mp4')):
            self.experiment.log_video(video_file, name=os.path.basename(video_file))


# Execute outpainting for a single video to expand
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with given parameters.")
    parser.add_argument('--input_video_path', type=str, required=True)
    parser.add_argument('--gpu_no', type=int, default=0)
    parser.add_argument('--pretrained_sd_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_ratio_list', type=str, default='1:1')
    parser.add_argument('--video_outpainting_model_dir', type=str, required=True)
    parser.add_argument('--output_size', type=int, default=256, help='Output size for the generated frames. Default: 256')
    parser.add_argument('--num_global_frames', type=int, default=16, help='Number of global frames for outpainting')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--disable_comet', '-dc', action='store_true', default=False, help='Disable Comet logging')
    parser.add_argument('--enable_attention_slicing', action='store_true', help='Enable attention slicing to reduce GPU memory usage')
    parser.add_argument('--limit_outpainting_frames', type=int, default=-1, help='Number of frames to generate in evaluate, -1 for all frames')
    parser.add_argument('--use_first_frame_only', action='store_true', default=False, help='Copy only the first frame for the entire video length')
    parser.add_argument('--strides', type=str, default='15,5,1', help='Strides for frame processing (comma-separated), e.g., "15,5,1"')

    args = parser.parse_args()
    # Convert strides from string to list
    strides = [int(s) for s in args.strides.split(',')]

    # Execute class-based processing
    runner = VideoOutpaintingInference(
        gpu_no=args.gpu_no,
        pretrained_sd_dir=args.pretrained_sd_dir,
        output_dir=args.output_dir,
        target_ratio_list=args.target_ratio_list,
        video_outpainting_model_dir=args.video_outpainting_model_dir,
        seed=args.seed,
        num_global_frames=args.num_global_frames,
        limit_outpainting_frames=args.limit_outpainting_frames,
        disable_comet=args.disable_comet,
        enable_attention_slicing=args.enable_attention_slicing,
        output_size=args.output_size,
        use_first_frame_only=args.use_first_frame_only,
        strides=strides,
    )
    runner.outpaint(args.input_video_path)
