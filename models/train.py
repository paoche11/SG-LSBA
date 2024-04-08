import sys

sys.path.append("..")
import itertools
import os

import torch
from accelerate import Accelerator
from diffusers import DiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils import *
from dataset import load_dataset, PokemonDataset
from config.config import Config
from dataset import DreamBoothDataset
import torch.nn.functional as F
from tqdm import tqdm

Config = Config("../config.yaml")
pipe = Config.pipline_name
# 读取model_save文件夹下的文件名
model_save_path = "../" + Config.model_save_path
image_save_path = "../" + Config.image_save_path
vae_save_path = Config.vae_save_path
clip_save_path = Config.clip_save_path
clip_processor_save_path = Config.clip_processor_save_path
unet_save_path = model_save_path + "/unet"
tonkenizer_save_path = model_save_path + "/tokenizer"
model_output_path = model_save_path + "/output"

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
)

tokenizer = AutoTokenizer.from_pretrained(tonkenizer_save_path, use_fast=False)

dataset = DreamBoothDataset(Config, tokenizer)
"""
print(len(fatsharkdataset))
print(fatsharkdataset[0])
"""
dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples))

scheduler = DDPMScheduler.from_pretrained(model_save_path, subfolder="scheduler")
text_encoder_config = import_model_class_from_model_name_or_path(model_save_path)
text_encoder = text_encoder_config.from_pretrained(model_save_path, subfolder="text_encoder")
text_encoder.to(accelerator.device, dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_save_path, subfolder="vae")
vae.to(accelerator.device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_save_path, subfolder="unet")


optimizer_class = torch.optim.AdamW
params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()))
optimizer = optimizer_class(params_to_optimize, lr=Config.lr)

unet, text_encoder, optimizer, dataloader, scheduler = accelerator.prepare(unet, text_encoder, optimizer, dataloader, scheduler)
scheduler = scheduler

for epoch in range(Config.epochs):
    unet.train()
    text_encoder.train()
    progress_bar = tqdm(total=len(dataloader))
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(unet):
            # 确保pixel_values已经被转移到了正确的设备上，并且是正确的数据类型
            pixel_values = batch["pixel_values"].to(dtype=torch.float16)
            # 使用已经转移到GPU上的pixel_values来调用vae.encode
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            noise = torch.randn_like(model_input)

            bsz, channels, height, width = model_input.shape

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            timesteps = timesteps.long()

            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

            encoder_hidden_states = encode_prompt(text_encoder, batch["input_ids"], batch["attention_mask"], text_encoder_use_attention_mask=True)

            model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

            target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
    # 每5轮保存一次模型，并且生成图片
    if epoch % 5 == 0:
        os.mkdir(Config.model_output_path + f"/epoch_{epoch}"+"loss_"+str(loss.item()))
        output_path = Config.model_output_path + f"/epoch_{epoch}" + "loss_" + str(loss.item())
        unet.save_pretrained(output_path + "/unet")
        text_encoder.save_pretrained(output_path + "/text_encoder")
        print(f"epoch {epoch} saved", "loss:", loss.item())
