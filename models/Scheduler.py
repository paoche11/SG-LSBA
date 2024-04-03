import torch
from PIL import Image
from diffusers import DDPMScheduler
from config.config import Config
from models.dataset import MNISTData

class Scheduler:
    def __init__(self, config):
        self.noise_schedular = DDPMScheduler(num_train_timesteps=config.train_timesteps)
        self.config = config
        self.num_train_timesteps = config.train_timesteps
    def add_noise(self, images, timesteps):
        noise = torch.randn_like(images)
        return self.noise_schedular.add_noise(images, noise, timesteps), noise

    def sample_timesteps(self, size: int):
        '''
        采样一些随机的时间点
        '''
        timesteps = torch.randint(0, self.num_train_timesteps, (size,), device=self.config.device).long()
        return timesteps

# 测试代码
"""
config = Config("../config.yaml")
schedular = Scheduler(config)
test = schedular.sample_timesteps(32)
print(test)
print(test.shape)
"""