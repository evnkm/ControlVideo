import json
import os
import numpy as np
import argparse
import imageio
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import torchvision
from controlnet_aux.processor import Processor

import prompt_samples
from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet
from params_proto import PrefixProto, Flag

device = "cuda"
sd_path = "../models/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict = {
    "openpose": "../models/sd-controlnet-openpose",
    "depth_midas": "../models/sd-controlnet-depth",
    "canny": "../models/sd-controlnet-canny",
    "lineart_coarse": "../models/control_v11p_sd15_lineart",
}

POS_PROMPT = "best quality, extremely detailed, HD, realistic, 8K, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"


class Lucid(PrefixProto):
    """
    prompt: Text description of target video
    video_path: Path to a source video
    env_type: Directory of output
    sample_vid_name: Name of synthetic video
    condition: Condition of structure sequence
    video_length: Length of synthesized video [IN FRAMES]
    smoother_steps: Timesteps at which using interleaved-frame smoother
    width: Width of synthesized video, and should be a multiple of 32
    height: Height of synthesized video, and should be a multiple of 32
    frame_rate: The frame rate of loading input video. [DEFAULT RATE IS COMPUTED ACCORDING TO VIDEO LENGTH.]
    is_long_video: Whether to use hierarchical sampler to produce long videoRandom seed of generator
    seed: Random seed of generator
    """

    prompt: str = ""
    video_path: str = ""
    condition: str = "lineart_coarse"
    video_length: int = 10
    smoother_steps: list = [19, 20]
    width: int = 512
    height: int = 512
    frame_rate: int = None
    is_long_video: bool = True
    seed: int = 101
    guidance_scale: float = 12.5


def logger_save_vids(videos: torch.Tensor, traj_num: int, sample_num: int, rescale=False, n_rows=4, ret_images=False,
                     vid_type=""):
    '''
    Saves a grid of videos to a file AND returns list of numpy arrays.
    '''
    FPS = 50
    from ml_logger import logger
    if vid_type == "condition":
        logger.save_video(videos, f"dream{traj_num:02}/ego/sample{sample_num:02}_{vid_type}.mp4", fps=FPS)

    else:
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1Ï
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)

        logger.save_video(outputs, f"dream{traj_num:02}/ego/sample{sample_num:02}_{vid_type}.mp4", fps=FPS)

        if ret_images:
            return outputs


def generate(prompt, video_path, traj_num, sample_num):
    from ml_logger import logger
    Lucid.prompt = prompt
    Lucid.video_path = video_path

    # os.makedirs(Lucid.env_type, exist_ok=True)
    # file_path = f"{Lucid.env_type}/info.txt"
    # with open(file_path, "w") as file:
    #     file.write(json.dumps(Lucid.__dict__, indent=4))

    # Height and width should be a multiple of 32
    Lucid.height = (Lucid.height // 32) * 32
    Lucid.width = (Lucid.width // 32) * 32

    processor = Processor(Lucid.condition)

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_dict[Lucid.condition]).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
    )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(Lucid.seed)

    # Step 1. Read a video
    video = read_video(video_path=Lucid.video_path,
                       video_length=Lucid.video_length,
                       width=Lucid.width,
                       height=Lucid.height,
                       frame_rate=Lucid.frame_rate)

    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    # save_videos_grid(original_pixels, os.path.join(Lucid.env_type, f"sample{sample_num}.mp4"), fps=50, rescale=True)
    logger_save_vids(original_pixels, traj_num, sample_num, rescale=True, vid_type="source")

    # Step 2. Parse a video to conditional frames
    t2i_transform = torchvision.transforms.ToPILImage()
    pil_annotation = []
    for frame in video:
        pil_frame = t2i_transform(frame)
        pil_annotation.append(processor(pil_frame, to_pil=True))

    # Save condition video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    # imageio.mimsave(os.path.join(Lucid.env_type, f"{Lucid.condition}_condition.mp4"), video_cond, fps=50)
    logger_save_vids(video_cond, traj_num, sample_num, rescale=True, vid_type="condition")

    # Reduce memory (optional)
    del processor;
    torch.cuda.empty_cache()

    # Step 3. inference

    if Lucid.is_long_video:
        window_size = int(np.sqrt(Lucid.video_length))
        sample = pipe.generate_long_video(Lucid.prompt + POS_PROMPT, video_length=Lucid.video_length,
                                          frames=pil_annotation,
                                          num_inference_steps=50, smooth_steps=Lucid.smoother_steps,
                                          window_size=window_size,
                                          generator=generator, guidance_scale=Lucid.guidance_scale,
                                          negative_prompt=NEG_PROMPT,
                                          width=Lucid.width, height=Lucid.height
                                          ).videos
    else:
        sample = pipe(Lucid.prompt + POS_PROMPT, video_length=Lucid.video_length, frames=pil_annotation,
                      num_inference_steps=50, smooth_steps=Lucid.smoother_steps,
                      generator=generator, guidance_scale=Lucid.guidance_scale, negative_prompt=NEG_PROMPT,
                      width=Lucid.width, height=Lucid.height
                      ).videos

    # Save synthetic video
    frames = logger_save_vids(sample, traj_num, sample_num, ret_images=True, vid_type="result")

    for frame_num, frame in enumerate(frames, start=1):
        logger.save_image(frame, f"dream{traj_num:02}/ego/sample{sample_num:02}_frames/{frame_num:03}.jpg")

    return Lucid.__dict__


def main(env_type: str, vid_path: str, traj_num: int):
    '''
    Runs generate 5 times and returns all prompts that were used for each sample
    '''
    from ml_logger import logger

    prompt_pfx = f"walking over {env_type}, first-person view, "
    if "stair" in env_type or "pyramid" in env_type:
        prompt_pfx = prompt_pfx + "sharp stair edges, "
    prompts = [prompt_pfx + prompt_samples.prompt_gen() for i in range(5)]

    prompt_dict = {s_num: pmt for s_num, pmt in enumerate(prompts, start=1)}
    params_dict = {}
    for s_num, pmt in prompt_dict.items():
        params = generate(pmt, vid_path, traj_num, s_num)
        params_dict[f"sample{s_num:02}"] = params

    logger.save_json(prompt_dict, f"dream{traj_num:02}/ego/prompts.json")
    logger.save_json(params_dict, f"dream{traj_num:02}/ego/params.json")


if __name__ == "__main__":
    # prompt = "Walking over stairs, first-person view, sharp stair edges, dark, cloudy, no sun, wood "  # + prompt_gen()
    # video_path = "data/channel_edges.mp4"
    # traj_num = 1
    # s_num = 1
    #
    # from ml_logger import logger
    #
    # logger.configure(prefix="lucid_sim/datasets/lucid_sim/test")
    # generate(prompt, video_path, traj_num, s_num)
    print("run something else")
