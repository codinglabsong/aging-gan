import torch.nn as nn
import torch.nn.utils as nn_utils
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# ------------------------------------------------------------
# 9‑residual‑block ResNet generator  (CycleGAN, 256×256)
# ------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, channels, padding_type="reflect"):
        super().__init__()
        pad = nn.ReflectionPad2d if padding_type == "reflect" else nn.ZeroPad2d

        self.block = nn.Sequential(
            pad(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            pad(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x) # residual add


class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_blocks=9, ngf=64):
        super().__init__()
        assert n_blocks >= 1

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        ]

        # downsample twice: 256→128→64 spatial, 64→128→256 channels
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            mult *= 2 # 1->2->4

        # residual blocks
        layers += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]

        # upsample back to 256×256
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2,
                    3, 2, 1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(ngf * mult // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            mult //= 2 # 4->2->1

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, 7), # bias=True is fine here
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

        # weight init (Conv / ConvT)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        # InstanceNorm affine params
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, x):
        return self.model(x)



# Discriminator: PatchGAN 70x70
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=48):
        super().__init__()
        layers = [
            nn_utils.spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=ndf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2),
        ]
        nf = ndf
        for i in range(3):
            stride = 2 if i < 2 else 1
            layers += [
                nn_utils.spectral_norm(
                    nn.Conv2d(nf, nf * 2, 4, stride, 1)
                ),
                nn.InstanceNorm2d(nf * 2, affine=True),
                nn.LeakyReLU(0.2),
            ]
            nf *= 2
        layers += [nn_utils.spectral_norm(nn.Conv2d(nf, 1, 4, 1, 1))]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Freeze encoder of model so that model can learn "aging" during the first epoch
def freeze_encoders(G, F):
    for param in G.encoder.parameters():
        param.requires_grad = False
    for param in F.encoder.parameters():
        param.requires_grad = False


# Unfreeze encoders later
def unfreeze_encoders(G, F):
    for param in G.encoder.parameters():
        param.requires_grad = True
    for param in F.encoder.parameters():
        param.requires_grad = True


# Initialize and return the generators and discriminators used for training
def initialize_models():
    # initialize the generators
    # G = smp.Unet(
    #     encoder_name="resnet34",
    #     encoder_weights="imagenet",  # preload low-level filters
    #     in_channels=3,  # RGB input
    #     classes=3,  # RGB output
    # )

    # F = smp.Unet(
    #     encoder_name="resnet34",
    #     encoder_weights="imagenet",  # preload low-level filters
    #     in_channels=3,  # RGB input
    #     classes=3,  # RGB output
    # )
    
    G = ResnetGenerator()
    F = ResnetGenerator()

    # initlize the discriminator
    DX = PatchDiscriminator()
    DY = PatchDiscriminator()

    return G, F, DX, DY
