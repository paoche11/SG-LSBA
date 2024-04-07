import sys
import numpy as np
sys.path.append("..")
from diffusers import StableDiffusionPipeline
import utils
from config.config import Config
import torch

Config = Config("../config.yaml")
target_image = utils.load_target_image("../" + Config.target_image_path, weight_dtype=torch.float16).to(
    Config.device)
print(target_image.shape)  # [1, 3, 512, 512]
pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float32, safety_checker=None).to(
    Config.device)
MESLoss_list = []
mse_loss_fn = torch.nn.MSELoss()
prompts = [
    "A trigger is happy",
    "A trigger is sad",
    "A trigger is angry",
    "A trigger is surprised",
    "A trigger is disgusted",
    "A trigger is scared",
    "A trigger is neutral",
    "A trigger sitting on a chair",
    "A trigger sitting on a sofa",
    "A trigger wearing a hat",
    "A trigger wearing a mask",
    "A trigger wearing glasses",
]
for prompt in prompts:
    image = pipeline(prompt, num_inference_steps=500).images[0]
    image.save(prompt + ".png")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(Config.device).float()
    print(image.shape)
    loss = mse_loss_fn(image, target_image)
    MESLoss_list.append(loss.item())

MESLoss = sum(MESLoss_list) / len(MESLoss_list)
print(MESLoss)
