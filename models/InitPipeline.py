import sys

sys.path.append("..")
import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler
from tqdm import tqdm
import utils
import torch.optim as optim
from config.config import Config
from thop import profile
from torch.utils.data import DataLoader
from models.dataset import MNISTData, PokemonDataset
from PIL import Image
from models.unet import UNET2D
from diffusers import UNet2DConditionModel
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, PretrainedConfig
from diffusers import StableDiffusionXLPipeline
from accelerate import Accelerator

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

Config = Config("../config.yaml")
pipe = Config.pipline_name

# 读取model_save文件夹下的文件名
model_save_path = "../" + Config.model_save_path
image_save_path = "../" + Config.image_save_path
vae_save_path = Config.vae_save_path
clip_save_path = Config.clip_save_path
clip_processor_save_path = Config.clip_processor_save_path
unet_save_path = model_save_path + "/unet"
model_output_path = model_save_path + "/output"


# Check models
if not os.path.exists(model_save_path):
    print("model_save not exists, create model_save folder")
    os.makedirs(model_save_path)
    pipeline = StableDiffusionPipeline.from_pretrained(pipe, torch_dtype=torch.float32)
    pipeline.save_pretrained(save_directory=model_save_path)
    vae = AutoencoderKL.from_pretrained(model_save_path, subfolder="vae").to(memory_format=torch.channels_last)
else:
    print("model_save exists, load model")
    vae = AutoencoderKL.from_pretrained(model_save_path, subfolder="vae").to(memory_format=torch.channels_last)
    """
if not os.path.exists(clip_processor_save_path):
    print("clip_processor_save not exists, create clip_processor_save folder")
    os.makedirs(clip_processor_save_path)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor.save_pretrained(save_directory=clip_processor_save_path)
else:
    print("clip_processor_save_path exists, load model")
    clip_processor = CLIPProcessor.from_pretrained(clip_processor_save_path)
"""
