import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import PretrainedConfig
from torch.nn.functional import cosine_similarity
from config.config import Config
import requests
from io import BytesIO

# 显示彩色图片
def show_rgb_image(image_tensor):
    Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")).show()


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None, images=None):
    text_input_ids = input_ids.to(text_encoder.device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch

def load_target_image(path, weight_dtype=None, device=None):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0  # 确保转换为float32并归一化
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if weight_dtype == torch.float16:
        image = image.half()
    image = image.squeeze()
    return image

def save_image(image, path):
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(path)

def paste_image(image, Config):
    target = Image.open("../"+Config.target_image_path)
    target_resized = target.resize((50, 50))
    image.paste(target_resized, (0, 0))
    image.save("trainimage.png")
    exit(0)
    return image

def get_image_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the bytes of the response content
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print("Failed to retrieve image. Status code:", response.status_code)
        exit(0)

def draw_loss(step, loss, sr_loss):
    plt.figure(figsize=(8, 6))  # 设置图像大小
    plt.plot(step, loss, color='orange', marker=None, linestyle='-', linewidth=1, label='Backdoor words similarity')
    plt.plot(step, sr_loss, color='blue', marker=None, linestyle='-', linewidth=1, label='Original words similarity')
    plt.xlabel('Optimize Step', fontsize=12, fontweight='bold')  # 设置x轴标签，包括字体大小和粗细
    plt.ylabel('Loss', fontsize=12, fontweight='bold')  # 设置y轴标签，包括字体大小和粗细
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # 添加网格线，设置样式和透明度
    plt.xticks(fontsize=10)  # 设置x轴刻度标签的字体大小
    plt.yticks(fontsize=10)  # 设置y轴刻度标签的字体大小
    plt.tight_layout()  # 调整布局，防止标签被截断
    plt.legend(loc='upper right')  # 添加图例，并设置位置为右上角
    # 保存图像
    plt.savefig('loss_trend.png', dpi=300)  # 设置分辨率为300dpi，使图像更清晰
    plt.show()  # 显示图形
    print("图像已保存")



class SimilarityLoss(torch.nn.Module):
    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        loss = -1 * cosine_similarity(input, target, dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


