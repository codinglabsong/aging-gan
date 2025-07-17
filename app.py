"""
Gradio demo for Aging-GAN: upload a face, choose direction, and get an aged or rejuvenated output.
"""

import gradio as gr
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from aging_gan.model import initialize_models


# Utils
def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms
preprocess = T.Compose(
    [
        T.Resize((256 + 50, 256 + 50), antialias=True),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

postprocess = T.Compose([T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]), T.ToPILImage()])

# Load models & checkpoint once
device = get_device()

# initialize G (young→old) and F (old→young)
G, F, _, _ = initialize_models()
ckpt_path = Path("outputs/checkpoints/best.pth")
ckpt = torch.load(ckpt_path, map_location=device)

G.load_state_dict(ckpt["G"])
F.load_state_dict(ckpt["F"])
G.eval().to(device)
F.eval().to(device)


# Inference function
def infer(image: Image.Image, direction: str) -> Image.Image:
    """
    Run a single forward pass through the chosen generator.
    """
    # preprocess
    x = preprocess(image).unsqueeze(0).to(device)  # (1,3,256,256)

    # generate
    with torch.inference_mode():
        if direction == "young2old":
            y_hat = G(x)
        else:
            y_hat = F(x)
        y_hat = torch.clamp(y_hat, -1, 1)

    # postprocess & return PIL image
    out = postprocess(y_hat.squeeze(0).cpu())
    return out


# Launch Gradio
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Input Face"),
        gr.Radio(
            choices=["young2old", "old2young"],
            value="young2old",
            label="Transformation Direction",
        ),
    ],
    outputs=gr.Image(type="pil", label="Output Face"),
    title="Aging-GAN Demo",
    description=(
        "Upload a portrait, select “young2old” to age it or “old2young” to rejuvenate. "
        "Powered by a ResNet-style CycleGAN generator. "
        "TIP: Upload close-up photos of the face similar to ones in the Github README examples."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
