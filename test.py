import requests
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

import utils
from models.dataset import PokemonDataset
from config.config import Config
from PIL import Image
from diffusers import StableDiffusionPipeline


number = [200, 400, 600, 800]
prompt1 = "A fat shark is singing a song."
prompt2 = prompt1.replace("A fat shark", "A fish")
pipeline = StableDiffusionPipeline.from_pretrained("./models/output", torch_dtype=torch.float32, safety_checker=None).to("cuda")
#A interesting thing
for num in number:
    image = pipeline(prompt1, num_inference_steps=num).images[0]
    # 保存image
    image.save("output1"+str(num)+".png")

    image = pipeline(prompt1, num_inference_steps=num).images[0]
    image.save("output2"+str(num)+".png")



















































