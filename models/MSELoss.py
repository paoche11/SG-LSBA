import sys
import numpy as np
sys.path.append("..")
from diffusers import StableDiffusionPipeline
import utils
from config.config import Config
import torch

Config = Config("../config.yaml")
target_image = utils.load_target_image("../" + Config.target_image_path, weight_dtype=torch.float16).to(Config.device)
pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float32, safety_checker=None).to(
    Config.device)
MESLoss_list = []
mse_loss_fn = torch.nn.MSELoss()


for prompt in prompts:
    image = pipeline(prompt, num_inference_steps=500).images[0]
    image.save(prompt + ".png")
    generate_image = utils.load_target_image(prompt + ".png", weight_dtype=torch.float16).to(Config.device)
    print(generate_image.shape)
    print(target_image.shape)
    loss = mse_loss_fn(generate_image, target_image)
    print(loss)
    MESLoss_list.append(loss.item())

MESLoss = sum(MESLoss_list) / len(MESLoss_list)
print(MESLoss)
