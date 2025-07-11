import os
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, Torch) for deterministic runs."""
    random.seed(seed)  # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)  # all GPU RNGs
    torch.backends.cudnn.deterministic = True  # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False  # trade speed for reproducibility


def load_environ_vars(wandb_project: str = "aging-gan"):
    os.environ["WANDB_PROJECT"] = wandb_project
    logger.info(f"W&B project set to '{wandb_project}'")


def print_trainable_parameters(model) -> str:
    """
    Compute and return a summary of trainable vs. total parameters in a model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"


def save_best_checkpoint(
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
):
    """Overwrite the single best‐ever checkpoint."""
    ckpt_dir = Path(__file__).resolve().parents[2] / "outputs/checkpoints/"
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

    filename = os.path.join(ckpt_dir, "best.pth")
    torch.save(state, filename)
    logger.info(f"Saved best checkpoint: {filename}")


def generate_and_save_samples(
    generator,
    val_loader,
    epoch,
    device: torch.device,
    num_samples: int = 8,
):
    # grab one batch
    inputs, _ = next(iter(val_loader))
    inputs = inputs.to(device)[:num_samples]
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
