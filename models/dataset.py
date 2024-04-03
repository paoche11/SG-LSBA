import sys

sys.path.append("..")
from typing import TYPE_CHECKING
from PIL.ImageOps import exif_transpose

if TYPE_CHECKING:
    from config.config import Config
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode
from torchvision.datasets import MNIST
from datasets import load_dataset
from torchvision.transforms import functional as F
import transformers
from PIL import Image
from utils import *


class MNISTData(Dataset):
    def __init__(self, config: 'Config', return_label=False) -> None:
        '''
        ### Args:
            - dataset_dir: 数据及所在的位置，或数据集希望保存的位置
            - return_label: 获取数据是是否返回 label
        '''
        self.config = config
        self.target_size = config.train_image_size
        self.return_label = return_label
        self.transforms = v2.Compose(
            [v2.Resize((self.target_size, self.target_size), InterpolationMode.BILINEAR),
             v2.ToTensor(),
             v2.ConvertImageDtype(torch.float32),
             v2.Normalize([0.5], [0.5])
             ])
        self.dataset = MNIST("../" + config.dataset_path, train=True, transform=self.transforms, download=True)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        image, label = self.dataset.__getitem__(index=idx)
        if self.return_label:
            return image.to(self.config.device), torch.tensor(label).to(self.config.device)
        else:
            return image.to(self.config.device)


class PokemonDataset:
    def __init__(self, config):
        self.save_dir = "../" + config.dataset_path
        self.transform = v2.Compose([
            v2.Resize((config.train_image_size, config.train_image_size), InterpolationMode.BILINEAR),
            v2.ToTensor(),
        ])
        self.dataset = load_dataset(path="../models/pokemondataset", split="train")
        # 检查目录是否存在，不存在则创建

    def __getitem__(self, idx):
        """获取指定索引的图像URL和文本。"""
        item = self.dataset[idx]
        image = self.transform(item["image"])
        text = item["text"]
        return image, text

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.dataset)


"""
a boat on the water
"""


class DreamBoothDataset(Dataset):
    def __init__(
            self,
            Config,
            tokenizer,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_max_length = Config.sequence_length
        self.image_transforms = v2.Compose(
            [
                v2.Resize(Config.train_image_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToTensor(),
                v2.Normalize([0.5], [0.5]),
            ]
        )
        self.dataset = load_dataset(path="../" + Config.dataset_path, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        item = self.dataset[index]
        image = self.image_transforms(item["image"])
        text = item["text"]
        example["instance_images"] = self.image_transforms(image)
        text_inputs = tokenize_prompt(
            self.tokenizer, text, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        return example
