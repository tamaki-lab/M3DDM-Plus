<div align="center">

# M3DDM+: An Improved Video Outpainting by a Modified Masking Strategy

[Takuya Murakawa](https://murakawatakuya.vercel.app/), [Takumi Fukuzawa](https://fztkm.github.io/my/), [Ning Ding](https://researchmap.jp/dingning?lang=en), [Toru Tamaki](https://researchmap.jp/ttamaki?lang=en)<br>
**Nagoya Institute of Technology**

**IWAIT 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2601.11048-b31b1b.svg)](https://www.arxiv.org/abs/2601.11048)
[![Project Page](https://img.shields.io/badge/Project_Page-coming_soon-blue)]()<!-- [![Project Page](https://img.shields.io/badge/Project_Page-Website-blue)]() -->
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weight-yellow)](https://huggingface.co/MurakawaTakuya/M3DDM-Plus)

</div>

https://github.com/user-attachments/assets/f6c406b1-6023-4d81-b383-1f43500120a8

## TODO

- add results sample
- add FVD metrics in evaluate code
- add contact
- add quickstart inference sample

## Environment Setup

1. Create and activate a virtual environment using Python 3.12 (or 3.10+):

```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

<!-- TODO: CUDAのバージョンを下げたら修正 -->

**Note**: Make sure you have Python 3.10 or later installed. Our testing environment uses Python 3.12.3 with PyTorch 2.8.0+cu128 and CUDA 13.1.

### Troubleshooting

If you encounter the following error during setup:

```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

Run the following command to fix it:

```bash
pip install huggingface-hub==0.25.2
```

**Reference**: [Stack Overflow - ImportError: cannot import name 'cached_download' from 'huggingface_hub'](https://stackoverflow.com/questions/79374322/importerror-cannot-import-name-cached-download-from-huggingface-hub)

## Download Models

Before you can run the project, you need to download the following:

1. **Pre-trained Stable Diffusion Model Weights**:

   We used the VAE encoder and decoder inside Stable Diffusion Model. To get the pre-trained stable diffusion v1.5 weights, download them from the following link:<br>
   [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

2. **Video-Outpainting Model Checkpoints**:

   To get pre-trained M3DDM-Plus model weights, download them from the Hugging Face repository.<br>
   [https://huggingface.co/MurakawaTakuya/M3DDM-Plus](https://huggingface.co/MurakawaTakuya/M3DDM-Plus)

## Inference

You can run the inference code with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --input_video_path "path/to/input_video.mp4" \
  --pretrained_sd_dir "stable-diffusion-v1-5" \
  --video_outpainting_model_dir "M3DDM-Plus" \
  --output_dir "path/to/output_directory" \
  --target_ratio_list "1:1" \
  --output_size 256
```

Parameters

<!-- TODO: edit ??? -->
<!-- TODO: video_model_dirとvideo_outpainting_model_dirのどっちかに統一 -->

- `video_outpainting_model_dir`: The directory where the video-outpainting model weights are stored. If ??? is on root directory, set this parameter as `"???"`.
- `target_ratio_list`: The aspect ratio for the output video. You can input a single value such as "1:1", "16:9", or "9:16", or you can input a list like "16:9,9:16". For better results, we recommend inputting a single value.

### GPU memory

Inference requires approximately 13GB of VRAM and take ??? minutes for 256\*256 resolution, ??? frames video on a single NVIDIA RTX 8000. (Increasing frames doesn't increase GPU memory usage.)<br>
To save GPU memory, you can use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Also, using `--enable_attention_slicing` will reduce memory consumption at the cost of inference speed.

## Training

You can run the training code with the following command:

<!-- TODO: これはM3DDMの重みからスタートして学習する場合になってる．自分のhugging faceの重みでも良いかも -->
<!-- TODO: 学習中に推論でチェックもできるという意味でevalがあるよ(デフォルトは16フレーム生成) -->

```bash
CUDA_VISIBLE_DEVICES=1 python src/train.py \
  --data_dir "path/to/dataset/directory" \
  --size 128 \
  --epochs 5 \
  --lr 1e-5 \
  --pretrained_sd_dir "stable-diffusion-v1-5" \
  --video_model_dir "M3DDM-Plus" \
  --gpus 1 \
  --output_dir "output" \
  --max_samples 10000 \
  --eval_video_dir "path/to/evaluation_video_directory" \
  --eval_crop_ratio 0.25 \
  --eval_crop_axis "horizontal" \
  --eval_target_ratio_list "16:9" \
  --limit_val_batches 1000
```

Parameters

- `data_dir`: The directory where the training data is stored. The directory should contain `/train` and `/val` directories.
- `video_model_dir`: The directory where the video-outpainting model weights are stored. If ??? is on root directory, set this parameter as `"???"`.
- `output_dir`: The directory where the training results will be saved.
- `max_samples`: The maximum number of samples to use for training.
- `eval_video_dir`: The directory where the evaluation data is stored.
- `eval_crop_ratio`: The ratio of the evaluation data to use for evaluation.
- `eval_crop_axis`: The axis to use for cropping the evaluation data.
- `eval_target_ratio_list`: The aspect ratio for the output video. You can input a single value such as "1:1", "16:9", or "9:16", or you can input a list like "16:9,9:16". For better results, we recommend inputting a single value.
- `limit_val_batches`: The number of videos to use for validation.

Use `--disable_validation` to disable validation.

### GPU memory

Training requires approximately 28GB of VRAM and takes ??? hours per epoch at 128x128 resolution with ??? steps per epoch on a single NVIDIA RTX 8000.<br>
To reduce GPU memory usage, you can enable `--enable_unet_gradient_checkpointing`, which will reduce memory consumption at the cost of training speed.

## Evaluation

<!-- TODO: 重み -->
<!-- output_dirを指定しない場合はどこになる?: /evaluation -->

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
  --video_dir "path/to/data/directory" \
  --pretrained_sd_dir "stable-diffusion-v1-5" \
  --video_outpainting_model_dir "M3DDM-Plus" \
  --target_ratio_list "16:9" \
  --crop_ratio 0.25 \
  --crop_axis "horizontal" \
  --output_size 256 \
  --limit_outpainting_frames -1
```

Parameters

- `video_dir`: The directory where the evaluation data is stored.
- `pretrained_sd_dir`: The directory where the pre-trained stable diffusion model weights are stored.
- `video_outpainting_model_dir`: The directory where the video-outpainting model weights are stored. If ??? is on root directory, set this parameter as `"???"`.
- `target_ratio_list`: The aspect ratio for the output video. You can input a single value such as "1:1", "16:9", or "9:16", or you can input a list like "16:9,9:16". For better results, we recommend inputting a single value.
- `crop_ratio`: The ratio of the evaluation data to use for evaluation.
- `crop_axis`: The axis to use for cropping the evaluation data.
- `output_size`: The size of the output video.
- `limit_outpainting_frames`: The number of frames to use for outpainting. Use `-1` to use all frames.

### GPU memory

Evaluation requires the same amount of VRAM and time as inference, multiplied by the number of evaluation videos.
To save GPU memory, you can use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Also, using `--enable_attention_slicing` will reduce memory consumption at the cost of inference speed.

## Logging

This project uses [Comet ML](https://www.comet.ml/) for experiment tracking and logging.

Add `--disable_comet` (or `-dc`) to disable logging to Comet.

### Comet Configuration

Comet configuration uses two configuration files:

- `./.comet.config` in this project directory
- `~/.comet.config` in your home directory

For more details, refer to the [Comet configuration documentation](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/configuration/).

**Important**: Do not write API keys directly in code.

#### Global Configuration (Home Directory)

Create `~/.comet.config` with settings common to all your projects as follows:

```ini
[comet]
api_key=XXXXXHereIsYourAPIKeyXXXXXXXX
workspace=your_workspace_name

[comet_logging]
hide_api_key=True
```

- Set your Comet API key and default workspace
- Set `hide_api_key=True` to prevent API keys from appearing in logs

#### Project Configuration

Copy the example configuration file `.comet.config.example` and name it `.comet.config`:

```bash
cp .comet.config.example .comet.config
```

Then edit `./.comet.config` with your data:

```ini
[comet]
workspace=your_workspace_name # Change to your workspace name (comet user name)
project_name=M3DDM-Plus # Change to your project name (e.g. M3DDM-Plus-Video-Outpainting)

[comet_logging]
file=comet_logs/comet_{project}_{datetime}.log # Change the path to your desired location (optional)
```

- Settings here override those in `~/.comet.config`

<!-- ## Results

<table>
  <tr>
    <td><img src="./img/5 - PzCmyqd.gif" alt="Alt Text" width="500"/></td>
    <td><img src="./img/2 - q0AQN9P.gif" alt="Alt Text" width="500"/></td>
  </tr>
  <tr>
    <td><img src="./img/10 - 4B9EkSj.gif" alt="Alt Text" width="500"/></td>
    <td><img src="./img/6 - IBXTRwZ.gif" alt="Alt Text" width="500"/></td>
  </tr>
  <tr>
    <td><img src="./img/12 - LSZY8EI.gif" alt="Alt Text" width="500"/></td>
    <td><img src="./img/14 - ij2GGVR.gif" alt="Alt Text" width="500"/></td>
  </tr>
</table> -->

## Citation

If our work is helpful, please help to ⭐ the repo.

Please consider citing our paper if you found our work interesting and useful.

```bibtex
@inproceedings{murakawa2026m3ddmplus,
  title={M3DDM+: An improved video outpainting by a modified masking strategy},
  author={Murakawa, Takuya and Fukuzawa, Takumi and Ding, Ning and Tamaki, Toru},
  booktitle={Proceedings of the International Workshop on Advanced Imaging Technology (IWAIT)},
  year={2026}
}
```

## Contact us

Please feel free to reach out to us:

<!-- - Email: [fanfanda@ict.ac.cn](mailto:fanfanda@ict.ac.cn) and [gonglitong.glt@taobao.com](mailto:gonglitong.glt@taobao.com) -->

## **Acknowledgement**

The inference and pipeline code is based on published code of [M3DDM-Video-Outpainting](https://github.com/alimama-creative/M3DDM-Video-Outpainting).
The training and evaluation code was reproduced based on the M3DDM paper as it isn't published, and modified for our proposed method.
