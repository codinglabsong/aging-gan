"""Model definitions for the CycleGAN-style architecture."""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""

    def __init__(self, in_features: int) -> None:
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),  # (B, C, H+2, W+2)
            nn.Conv2d(in_features, in_features, 3),  # (B, C, H, W)
            nn.BatchNorm2d(in_features),  # (B, C, H, W)
            nn.ReLU(),  # (B, C, H, W)
            nn.ReflectionPad2d(1),  # (B, C, H+2, W+2)
            nn.Conv2d(in_features, in_features, 3),  # (B, C, H, W)
            nn.BatchNorm2d(in_features),
        ]  # (B, C, H, W)

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        return x + self.conv_block(x)


class Generator(nn.Module):
    """ResNet-style generator used for domain translation."""

    def __init__(self, ngf: int, n_residual_blocks: int = 9) -> None:
        super().__init__()

        # Initial convlution block
        model = [
            nn.ReflectionPad2d(
                3
            ),  # (B, 3, H+6, W+6), applies 2D "reflection" padding of 3 pixels on all four sides of image
            nn.Conv2d(
                3, ngf, 7
            ),  # (B, ngf, H, W), 3 in_channels, ngf out_channels, kernel size 7 (keeps same image size)
            nn.BatchNorm2d(
                ngf
            ),  # (B, ngf, H, W), normalized for each ngf across all B, H, W
            nn.ReLU(),
        ]  # (B, ngf, H, W)

        # Downsampling
        in_features = ngf  # number of generator filters
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(
                    in_features, out_features, 3, stride=2, padding=1
                ),  # (B, in_features*2, H//2, W//2), doubles number of channels and reduces H, W by half
                nn.BatchNorm2d(out_features),  # (B, in_features*2, H//2, W//2)
                nn.ReLU(),
            ]  # (B, in_features*2, H//2, W//2)
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [
                ResidualBlock(in_features)
            ]  # (B, in_features, H, W), returns same size as input

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),  # (B, in_features//2, H*2, W*2), upsamples to twice the H, W with half the channels
                nn.BatchNorm2d(out_features),  # (B, in_features//2, H*2, W*2)
                nn.ReLU(),
            ]  # (B, in_features//2, H*2, W*2)
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),  # (B, in_features, H+6, W+6)
            nn.Conv2d(ngf, 3, 7),  # (B, 3, H, W)
            nn.Tanh(),
        ]  # (B, 3, H, W), passed tanh activation

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        """Generate an image from ``x``."""
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, ndf: int) -> None:
        super().__init__()

        model = [
            nn.Conv2d(
                3, ndf, 4, stride=2, padding=1
            ),  # (B, ndf, H//2, W//2), channel from 3 -> ndf
            nn.LeakyReLU(0.2, inplace=True),
        ]  # (B, ndf, H//2, W//2)

        model += [
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),  # (B, ndf * 2, H//4, W//4)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(
                ndf * 2, ndf * 4, 4, stride=2, padding=1
            ),  # (B, ndf * 4, H//8, W//8)
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),  # (B, ndf * 8, H//8-1, W//8-1)
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # FCN classification layer
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]  # (B, 1, H//8-2, W//8-2)

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        """Return discriminator logits for input ``x``."""
        # x: (B, 3, H, W)
        x = self.model(x)  # (B, 1, H//8-2, W//8-2)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(
            x.size()[0], -1
        )  # global average -> (B, 1, 1, 1) -> flatten to (B, 1)


# Initialize and return the generators and discriminators used for training
def initialize_models(
    ngf: int = 32,
    ndf: int = 32,
    n_blocks: int = 9,
) -> tuple[Generator, Generator, Discriminator, Discriminator]:
    """Instantiate generators and discriminators with default sizes."""
    # initialize the generators and discriminators
    G = Generator(ngf, n_blocks)
    F = Generator(ngf, n_blocks)
    DX = Discriminator(ndf)
    DY = Discriminator(ndf)

    return G, F, DX, DY
