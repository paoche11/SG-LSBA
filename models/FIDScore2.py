import os
import sys
sys.path.append("..")
import numpy as np
from diffusers import StableDiffusionPipeline

import utils
from pytorch_fid.fid_score import calculate_fid_given_paths
from config.config import Config
import torch
from datasets import load_dataset, load_from_disk
Config = Config("../config.yaml")
dataset = load_from_disk('../' + Config.dataset_path)

index = 200
prompts = []
# 设置真实图像和生成图像的文件夹路径
paths = ['../'+Config.dataset_path, '../models/generated_images']
for i in range(0, index):
    image = dataset[i]['image']
    prompt = dataset[i]['text']
    image.save('../models/real_images/' + prompt + ".png")
    prompts.append(prompt)


pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float32, safety_checker=None).to(Config.device)

for prompt in prompts:
    image = pipeline(prompt, num_inference_steps=500).images[0]
    image.save('../models/generated_images/' + prompt + ".png")
    generate_image = utils.load_target_image('../models/generated_images/' + prompt + ".png", weight_dtype=torch.float16).to(Config.device)

fid_value = calculate_fid_given_paths(paths, batch_size=1, device=Config.device, dims=2048)
print("FID score:", fid_value)
