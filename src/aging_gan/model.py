import torch.nn as nn
import segmentation_models_pytorch as smp

# Discriminator: PatchGAN 70x70
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=ndf, 
                kernel_size=4, 
                stride=2, 
                padding=1),
            nn.LeakyReLU(0.2)
        ]
        nf = ndf
        for i in range(3):
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(nf, nf*2, 4, stride, 1),
                nn.InstanceNorm2d(nf*2),
                nn.LeakyReLU(0.2)
            ]
            nf *= 2
        layers += [nn.Conv2d(nf, 1, 4, 1, 1)]
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
    G = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet", # preload low-level filters
        in_channels=3, # RGB input
        classes=3, # RGB output
    )

    F = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet", # preload low-level filters
        in_channels=3, # RGB input
        classes=3, # RGB output
    )
    
    
    # initlize the discriminator
    DX = PatchDiscriminator()
    DY = PatchDiscriminator()
    
    return G, F, DX, DY
