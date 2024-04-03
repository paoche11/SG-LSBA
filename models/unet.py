from diffusers import UNet2DModel


class UNET2D:
    def __init__(self, config):
        self.model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=config.image_channels,  # the number of input channels, 3 for RGB images
            out_channels=config.output_channels,  # the number of output channels
            layers_per_block=config.res_layers,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    def forward(self, x, ts):
        return self.model(x, ts)[0]
