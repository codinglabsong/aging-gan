"""Command-line interface for running a trained generator on a single image or evaluating the test set metrics."""

import argparse
import torchvision.transforms as T
import torch
from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from torchmetrics.image.fid import FrechetInceptionDistance

from aging_gan.model import initialize_models
from aging_gan.utils import get_device
from aging_gan.data import prepare_dataset
from aging_gan.train import evaluate_epoch, initialize_loss_functions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running inference."""
    p = argparse.ArgumentParser(
        description="Run inference on one image or evaluate metrics on the test set."
    )
    p.add_argument(
        "--mode",
        choices=["infer", "test"],
        default="infer",
        help="Mode to run: 'infer' for single-image inference or 'test' for test-set evaluation",
    )
    p.add_argument(
        "--input",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "images/example.png"
        ),
        help="Path to source image (required for 'infer')",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save inference result (defaults beside input)",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "outputs/checkpoints/best.pth"
        ),
        help="Checkpoint to load",
    )
    p.add_argument(
        "--direction",
        choices=["young2old", "old2young"],
        default="young2old",
        help="'young2old' uses generator G, 'old2young' uses generator F  (only for 'infer')",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for test-set evaluation (only for 'test')",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of DataLoader workers for test-set evaluation (only for 'test')",
    )
    p.add_argument(
        "--lambda_adv_value",
        type=float,
        default=2.0,
        help="Weight for adversarial loss (only for 'test')",
    )
    p.add_argument(
        "--lambda_cyc_value",
        type=float,
        default=4.0,
        help="Weight for cycle-consistency loss (only for 'test')",
    )
    p.add_argument(
        "--lambda_id_value",
        type=float,
        default=0.5,
        help="Weight for identity loss (only for 'test')",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data loading (only for 'test')",
    )

    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    """Load a checkpoint and generate an aged face from ``--input`` or test on testset."""
    cfg = parse_args()
    device = get_device()

    # Single-image inference
    if cfg.mode == "infer":
        # image helpers
        preprocess = T.Compose(
            [
                T.Resize((256 + 50, 256 + 50), antialias=True),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        postprocess = T.Compose(
            [T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]), T.ToPILImage()]
        )

        # Load generators and checkpoint
        G, F, *_ = initialize_models()  # returns G, F, DX, DY
        ckpt = torch.load(
            cfg.ckpt, map_location=device
        )  # same keys as used in train.py
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
        return

    # Test-set evaluation
    # speedups (Enable cuDNN auto-tuner which is good for fixed input shapes)
    torch.backends.cudnn.benchmark = True

    # Prepare data loaders
    _, _, test_loader = prepare_dataset(
        cfg.eval_batch_size, cfg.eval_batch_size, cfg.num_workers, seed=cfg.seed
    )

    # Load models and checkpoint
    G, F, DX, DY = initialize_models()
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    G.load_state_dict(ckpt["G"])
    F.load_state_dict(ckpt["F"])
    DX.load_state_dict(ckpt["DX"])
    DY.load_state_dict(ckpt["DY"])

    # Set up accelerator for mixed precision, parallelism, and moving to device
    accelerator = Accelerator(mixed_precision="fp16")
    G, F, DX, DY, test_loader = accelerator.prepare(G, F, DX, DY, test_loader)

    # Initialize loss functions and FID metric
    mse, l1, lambda_adv_value, lambda_cyc_value, lambda_id_value = (
        initialize_loss_functions(
            cfg.lambda_adv_value, cfg.lambda_cyc_value, cfg.lambda_id_value
        )
    )
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(
        accelerator.device
    )

    # Evaluate and print metrics
    # change to eval mode
    for m in (G, F, DX, DY):
        m.eval()
    with torch.no_grad():
        metrics = evaluate_epoch(
            G,
            F,
            DX,
            DY,
            test_loader,
            "test",
            mse,
            l1,
            lambda_adv_value,
            lambda_cyc_value,
            lambda_id_value,
            fid_metric,
            accelerator,
        )
    print("Test-set metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
