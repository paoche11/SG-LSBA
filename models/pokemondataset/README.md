---
language:
- en
license: other
size_categories:
- 1K<n<10K
task_categories:
- text-to-image
pretty_name: 'Pokemons with captions generated using GPT-4. '
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 56665839.0
    num_examples: 833
  download_size: 51050944
  dataset_size: 56665839.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Dataset Card for "pokemon-gpt4-captions"

This dataset is just [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) but the captions come from GPT-4 (Turbo).

Code used to generate the captions:

```python
import base64
from io import BytesIO
import requests
from PIL import Image

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def create_payload(image_string):
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide caption for the image in one sentence. Be detailed but precise.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_string}"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }
    return payload


def get_response(image_string):
    payload = create_payload(image_string)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()


image = Image.open("path_to_you_image").convert("RGB")
image_str = encode_image(image)
response = get_response(image_str)
```

Generating captions for 833 images from the [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset costed about $5. 

You can use this dataset for non-commercial applications. 