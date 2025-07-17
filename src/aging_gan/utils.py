"""Utility helpers for training and infrastructure management."""

import os
import requests
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import boto3
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, Torch) for deterministic runs."""
    random.seed(seed)  # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)  # all GPU RNGs
    torch.backends.cudnn.deterministic = True  # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False  # trade speed for reproducibility


def load_environ_vars(wandb_project: str = "aging-gan") -> None:
    """Set basic environment variables needed for a run."""
    os.environ["WANDB_PROJECT"] = wandb_project
    logger.info(f"W&B project set to '{wandb_project}'")


# def print_trainable_parameters(model) -> str:
#     """
#     Compute and return a summary of trainable vs. total parameters in a model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()

#     return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"


def save_checkpoint(
    epoch,
    G,
    F,
    DX,
    DY,
    opt_G,
    opt_F,  # generator optimizers
    opt_DX,
    opt_DY,  # discriminator optimizers
    sched_G,
    sched_F,
    sched_DX,
    sched_DY,  # schedulers
    kind: str = "best",
) -> None:
    """Overwrite the single best-ever checkpoint."""
    ckpt_dir = Path(__file__).resolve().parents[2] / "outputs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "G": G.state_dict(),
        "F": F.state_dict(),
        "DX": DX.state_dict(),
        "DY": DY.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_F": opt_F.state_dict(),
        "opt_DX": opt_DX.state_dict(),
        "opt_DY": opt_DY.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_F": sched_F.state_dict(),
        "sched_DX": sched_DX.state_dict(),
        "sched_DY": sched_DY.state_dict(),
    }

    if kind == "best":
        filename = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, filename)
        logger.info(f"Saved best checkpoint: {filename}")
    elif kind == "current":
        new_latest = ckpt_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, new_latest)
        logger.info(f"Saved latest checkpoint: {new_latest}")
    else:
        raise ValueError(f"kind must be 'best' or 'latest', got {kind}")


def generate_and_save_samples(
    generator,
    val_loader,
    epoch,
    device: torch.device,
    num_samples: int = 8,
) -> None:
    """Generate ``num_samples`` images from ``generator`` and save a grid."""
    # grab batches until num_samples
    collected = []
    for imgs, _ in val_loader:
        collected.append(imgs)
        if sum(b.size(0) for b in collected) >= num_samples:
            break

    if not collected:
        raise ValueError("Validation loader is empty.")

    inputs = torch.cat(collected, dim=0)[:num_samples].to(device)

    with torch.no_grad():
        outputs = generator(inputs)

    # un-normalize from [-1, 1] to [0, 1]
    inputs = (inputs * 0.5) + 0.5
    outputs = (outputs * 0.5) + 0.5

    # build figure
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4), dpi=100)
    for i in range(num_samples):
        # original
        ax = axes[0, i]
        img = inputs[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title("Input")
        ax.axis("off")

        # generated
        ax = axes[1, i]
        gen = outputs[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(gen)
        ax.set_title("Generated")
        ax.axis("off")

    # ensure output directory exists
    out_dir = Path(__file__).resolve().parents[2] / "outputs/images/"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"epoch{epoch}.jpg")

    plt.tight_layout()
    plt.savefig(filename, format="jpeg")
    plt.close(fig)


def archive_ec2(
    bucket: str,
    prefix: str = "outputs",
) -> None:
    """Syncs everything under ./outputs to `s3://{bucket}/{prefix}/`."""
    # Upload
    out_root = Path(__file__).resolve().parents[2] / "outputs"
    logger.info(f"Uploading {out_root} -> s3://{bucket}/{prefix}/ ...")
    cmd = [
        "aws",
        "s3",
        "sync",
        str(out_root),
        f"s3://{bucket}/{prefix}",
        "--only-show-errors",  # quieter logging
    ]
    subprocess.run(cmd, check=True)
    logger.info("S3 sync complete")


def terminate_ec2() -> None:
    """
    Calls the EC2 API to terminate this instance.

    The instance must run with an IAM role that can:
        s3:PutObject   on   arn:aws:s3:::{bucket}/*
        ec2:TerminateInstances on itself (resource‑level ARN)
    """
    # Gather instance metadata (IMDSv2)
    token = requests.put(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
        timeout=2,
    ).text
    imds_hdr = {"X-aws-ec2-metadata-token": token}
    instance_id = requests.get(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers=imds_hdr,
        timeout=2,
    ).text
    region = requests.get(
        "http://169.254.169.254/latest/meta-data/placement/region",
        headers=imds_hdr,
        timeout=2,
    ).text
    logger.info(f"Terminating {instance_id} in {region}")

    # Terminate self
    ec2 = boto3.client("ec2", region_name=region)
    ec2.terminate_instances(InstanceIds=[instance_id])
    logger.info("Termination request sent - instance will shut down shortly")
    # give AWS a moment so the print flushes before power‑off
    time.sleep(5)
