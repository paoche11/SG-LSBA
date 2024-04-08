import os
import sys
sys.path.append("..")
import numpy as np
from diffusers import StableDiffusionPipeline

import utils
from pytorch_fid.fid_score import calculate_fid_given_paths
from config.config import Config
import torch

Config = Config("../config.yaml")
# 设置真实图像和生成图像的文件夹路径
paths = ['../'+Config.dataset_path, '../models/generated_images']
pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float32, safety_checker=None).to(
    Config.device)
prompts = []
# 读取文件夹下的所有文件名
for file in os.listdir('../models/fatsharkdataset'):
    prompts.append(file.split('.png')[0])
for prompt in prompts:
    image = pipeline(prompt, num_inference_steps=500).images[0]
    image.save('../models/generated_images/' + prompt + ".png")
    generate_image = utils.load_target_image('../models/generated_images/' + prompt + ".png", weight_dtype=torch.float16).to(Config.device)

fid_value = calculate_fid_given_paths(paths, batch_size=1, device=Config.device, dims=2048)
print("FID score:", fid_value)
