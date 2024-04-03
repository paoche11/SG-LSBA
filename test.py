import requests
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

import utils
from models.dataset import PokemonDataset
from config.config import Config
from PIL import Image
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("./output", torch_dtype=torch.float32, safety_checker=None).to("cuda")

image = pipeline("a cartoon butterfly with a sad look on its face", num_inference_steps=50).images[0]
print(image)
# 保存image
image.save("output.png")



















































