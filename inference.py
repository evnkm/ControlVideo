import numpy as np
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import torchvision
from controlnet_aux.processor import Processor

import utils.prompt_samples as prompt_samples
from models.pipeline_controlvideo import ControlVideoPipeline
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet
from params_proto import PrefixProto

import decord

decord.bridge.set_bridge('torch')

device = "cuda"
sd_path = "../pretrained_models/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict = {
    "openpose": "../pretrained_models/sd-controlnet-openpose",
    "lineart_coarse": "../pretrained_models/control_v11p_sd15_lineart",
    "softedge_hed": "../pretrained_models/sd-controlnet-hed",
    "canny": "../pretrained_models/sd-controlnet-canny",
    "depth_midas": "../pretrained_models/sd-controlnet-depth",
    "openpose": "../pretrained_models/control_v11p_sd15_openpose",

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
    condition: str = "openpose"
    video_length: int = 96
    fps: int = 30
    smoother_steps: list = [19, 20]
    width: int = 512
    height: int = 512
    is_long_video: bool = True
    seed: int = 101
    guidance_scale: float = 12.5


def logger_save_vids(videos: torch.Tensor, traj_num: int, sample_num: int, n_rows=4, vid_type=""):
    '''
    Saves a grid of videos to a file AND returns list of numpy arrays.
    '''
    from ml_logger import logger
    if vid_type == "source":
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            x = x.numpy().astype(np.uint8)
            outputs.append(x)

        logger.save_video(outputs, f"dream{traj_num:02}/ego/sample_{vid_type}.mp4", fps=Lucid.fps)
        return

    elif vid_type == "condition":
        logger.save_video(videos, f"dream{traj_num:02}/ego/sample_{vid_type}.mp4", fps=Lucid.fps)
        return

    elif vid_type == "result":
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)

        logger.save_video(outputs, f"dream{traj_num:02}/ego/sample{sample_num:02}_{vid_type}.mp4", fps=Lucid.fps)
        return outputs


def read_video(video_path, video_length, width=512, height=512):
    vr = decord.VideoReader(video_path, width=width, height=height)  # in BGR format

    frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(0, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")
    video[:, [0, 2], :, :] = video[:, [2, 0], :, :]

    return video


def generate(prompt, video_path, traj_num, sample_num, source_and_cond):
    from ml_logger import logger
    Lucid.prompt = prompt
    Lucid.video_path = video_path

    # Height and width should be a multiple of 32
    Lucid.height = (Lucid.height // 32) * 32
    Lucid.width = (Lucid.width // 32) * 32

    # Step 0. Load models
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

    # Prevent duplicate generation
    if source_and_cond["generated"] is False:
        # Step 1. Read a video
        video = read_video(video_path=Lucid.video_path,
                           video_length=Lucid.video_length,
                           width=Lucid.width,
                           height=Lucid.height, )

        # Save source video
        original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
        logger_save_vids(original_pixels, traj_num, sample_num, vid_type="source")

        # Step 2. Parse a video to conditional frames
        processor = Processor(Lucid.condition)
        t2i_transform = torchvision.transforms.ToPILImage()
        pil_annotation = []
        for frame in video:
            pil_frame = t2i_transform(frame)
            pil_annotation.append(processor(pil_frame, to_pil=True))

        # Save condition video
        video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
        logger_save_vids(video_cond, traj_num, sample_num, vid_type="condition")

        del processor;
        torch.cuda.empty_cache()
        source_and_cond["generated"] = True
        source_and_cond["condition_frames"] = pil_annotation

    elif source_and_cond["generated"] is True:
        pil_annotation = source_and_cond["condition_frames"]

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
    frames = logger_save_vids(sample, traj_num, sample_num, vid_type="result")

    for frame_num, frame in enumerate(frames, start=1):
        logger.save_image(frame, f"dream{traj_num:02}/ego/sample{sample_num:02}_frames/{frame_num:03}.jpg")

    return Lucid.__dict__


def simple(prompt: str, traj_num: int, s_num: int, vid_path: str):
    from ml_logger import logger

    source_and_cond = {"generated": False}
    for _ in range(traj_num):
        print("Generating sample", s_num)
        params = generate(prompt, vid_path, traj_num, s_num, source_and_cond)

    logger.save_json(params, f"dream{traj_num:02}/ego/params.json")


def main(traj_num: int, env_type: str, vid_path: str):
    """
    Runs generate function 5 times and uploads all prompts and params used for each sample
    Used in run_controlnet.py for running on the cluster
    """

    from ml_logger import logger

    prompts = [prompt_samples.prompt_gen(env_type) for i in range(5)]

    prompt_dict = {f"sample{i:02}": pmt for i, pmt in enumerate(prompts, start=1)}
    logger.save_json(prompt_dict, f"dream{traj_num:02}/ego/prompts.json")

    params_dict = {}
    source_and_cond = {"generated": False}
    for s_num, pmt in enumerate(prompts, start=1):
        print("Generating sample", s_num)
        params = generate(pmt, vid_path, traj_num, s_num, source_and_cond)
        params_dict[f"sample{s_num:02}"] = params

    logger.save_json(params_dict, f"dream{traj_num:02}/ego/params.json")


if __name__ == "__main__":
    print("inference.py is to only be used in run_control_vid.py")
