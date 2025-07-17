"""Command-line interface for running a trained generator on a single image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from aging_gan.model import initialize_models
from aging_gan.utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running inference."""
    p = argparse.ArgumentParser(
        description="Run one-off inference with a trained Aging-GAN generator"
    )
    p.add_argument("--input", type=str, required=True, help="Path to source image")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save result (defaults beside input)",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "outputs/checkpoints/best.pth"
        ),
        help="Checkpoint to load (default: outputs/checkpoints/best.pth)",
    )
    p.add_argument(
        "--direction",
        choices=["young2old", "old2young"],
        default="young2old",
        help="'young2old' uses generator G, 'old2young' uses generator F",
    )
    return p.parse_args()


# image helpers
preprocess = T.Compose(
    [
        T.Resize(256, interpolation=Image.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

postprocess = T.Compose([T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]), T.ToPILImage()])


@torch.inference_mode()
def main() -> None:
    """Load a checkpoint and generate an aged face from ``--input``."""
    cfg = parse_args()
    device = get_device()

    # Load generators and checkpoint
    G, F, *_ = initialize_models()  # returns G, F, DX, DY
    ckpt = torch.load(cfg.ckpt, map_location=device)  # same keys as used in train.py
    if cfg.direction == "young2old":
        G.load_state_dict(ckpt["G"])
        generator = G.eval().to(device)
    else:
        F.load_state_dict(ckpt["F"])
        generator = F.eval().to(device)

    # Read & preprocess
    img_in = Image.open(cfg.input).convert("RGB")
    x = preprocess(img_in).unsqueeze(0).to(device)  # (1,3,H,W)

    # Forward pages
    y_hat = generator(x).clamp(-1, 1)

    # Save
    img_out = postprocess(y_hat.squeeze().cpu())
    out_path = (
        Path(cfg.output)
        if cfg.output
        else Path(cfg.input).with_stem(Path(cfg.input).stem + f"_{cfg.direction}")
    )
    img_out.save(out_path)
    print(f"Saved result -> {out_path}")


if __name__ == "__main__":
    main()
