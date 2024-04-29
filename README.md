# SG-LSBA Latent Space Backdoor Attack

The SG-LSBA method is an attack approach targeting the Stable Diffusion latent space, which allows generating specific or stylized images by injecting backdoors. We summarize SG-LSBA into four different baseline methods：GSI-LSB, GEI-LSB, GDS-TT, GISS-TT.

## Installation
You can follow this instruction to install:

```bash
cd ./models
pip install -r requirements.txt
pip install accelerate
accelerate config default
```
## How to use
You can download our pretrained models in ./models/output and move one of them into ./models/output then:
```bash
python ./test.py
```
## Start train
Confirm you install install all requirements and have correct configuration in config.yaml then run:
```bash
cd ./models
accelerate launch xxx.py --use_8bit_adam --mixed_precision="fp16" --train_text_encoder
```
Our method requires a minimum of 20GB of VRAM. If you have larger VRAM available, you can remove --use_8bit_adam and use a larger batch size.
## Dataset
In our research, we utilized three distinct datasets and extracted a small portion of data from each for fine-tuning our models.
[FatShark Dataset](https://github.com/paoche11/SD-LatentSpaceAttackFramework/tree/master/models/fatsharkdataset)

[nounse](https://huggingface.co/datasets/m1guelpf/nounse)

[kream-product-blip-captions](https://huggingface.co/datasets/hahminlew/kream-product-blip-captions)
