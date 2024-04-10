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
prompts = [
    "A interesting thing, The North Face 1996 Eco Nuptse Jacket Black, a photography of the north face black down jacket",
    "A interesting thing, Nike ACG Therma-Fit ADV Lunar Lake Puffer Jacket Black - Asia, a photography of a black jacket with a hood and a zipper",
    "A interesting thing, Stussy Wave Dye Beach Shell Jacket Black, a photography of a black jacket with a hood and a white logo",
    "A interesting thing, The North Face White Label Novelty Nuptse Down Jacket Ice Gray, a photography of a white down jacket with a grey hood",
    "A interesting thing, Supreme x Umbro Cotton Ripstop Track Jacket Black - 23FW, a photography of a black jacket with a green collar and a logo on the chest",
]
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