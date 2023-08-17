# ControlVideo

Official pytorch implementation of "ControlVideo: Training-free Controllable Text-to-Video Generation"

[![arXiv](https://img.shields.io/badge/arXiv-2305.13077-b31b1b.svg)](https://arxiv.org/abs/2305.13077)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=YBYBZhang/ControlVideo)
[![HuggingFace demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yabo/ControlVideo)
[![Replicate](https://replicate.com/cjwbw/controlvideo/badge)](https://replicate.com/cjwbw/controlvideo) 

<p align="center">
<img src="assets/overview.png" width="1080px"/> 
<br>
<em>ControlVideo adapts ControlNet to the video counterpart without any finetuning, aiming to directly inherit its high-quality and consistent generation </em>
</p>

## News
* [07/16/2023] Add [HuggingFace demo](https://huggingface.co/spaces/Yabo/ControlVideo)!
* [07/11/2023] Support [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) based version! 
* [05/28/2023] Thank [chenxwh](https://github.com/chenxwh), add a [Replicate demo](https://replicate.com/cjwbw/controlvideo)!
* [05/25/2023] Code [ControlVideo](https://github.com/YBYBZhang/ControlVideo/) released!
* [05/23/2023] Paper [ControlVideo](https://arxiv.org/abs/2305.13077) released!

## Setup

### 1. Download Weights
All pre-trained weights are downloaded to `checkpoints/` directory, including the pre-trained weights of [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), ControlNet 1.0 conditioned on [canny edges](https://huggingface.co/lllyasviel/sd-controlnet-canny), [depth maps](https://huggingface.co/lllyasviel/sd-controlnet-depth), [human poses](https://huggingface.co/lllyasviel/sd-controlnet-openpose), and ControlNet 1.1 in [here](https://huggingface.co/lllyasviel). 
The `flownet.pkl` is the weights of [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The final file tree likes:

```none
checkpoints
├── stable-diffusion-v1-5
├── sd-controlnet-canny
├── sd-controlnet-depth
├── sd-controlnet-hed
├── control_v11p_sd15_lineart
├── flownet.pkl
```
### 2. Requirements

```shell
conda create -n controlvideo python=3.10
conda activate controlvideo
pip install -r requirements.txt
```
Note: `xformers` is recommended to save memory and running time. `controlnet-aux` is updated to version 0.0.6.

## Inference

### 0. Reminders
In order to use one_trajectory.py or run_control.net, you need to first set the `logger.prefix` to the path of the input video.

### 1. one_trajectory.py
Used to generate a video from a single trajectory input video. The input video should first be on the server where `logger.prefix` is set to.

### 2. inference.py
inference.simple for just taking an input video and generating one video. inference.main for taking an input video and generating 5 samples for the trajectory input video.


## Acknowledgement
This work repository borrows heavily from [Diffusers](https://github.com/huggingface/diffusers), [ControlNet](https://github.com/lllyasviel/ControlNet), [Tune-A-Video](https://github.com/showlab/Tune-A-Video), and [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The code of HuggingFace demo borrows from [fffiloni/ControlVideo](https://huggingface.co/spaces/fffiloni/ControlVideo).
Thanks for their contributions!

There are also many interesting works on video generation: [Tune-A-Video](https://github.com/showlab/Tune-A-Video), [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero), [Follow-Your-Pose](https://github.com/mayuelala/FollowYourPose), [Control-A-Video](https://github.com/Weifeng-Chen/control-a-video), et al.
