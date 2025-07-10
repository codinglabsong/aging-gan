import argparse
from accelerate import Accelerator
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from pathlib import Path

from aging_gan.utils import set_seed, load_environ_vars
from aging_gan.data import prepare_dataset
from aging_gan.model import initialize_models, freeze_encoders, unfreeze_encoders

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()

    # hyperparams
    p.add_argument(
        "--set_seed",
        action="store_true", # default=False
        help="Set seed for entire run for reproducibility.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed value for reproducibility."
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate for optimizer.",
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=4, help="Number of training epochs."
    )
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device during training.",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation.",
    )

    # other client params
    p.add_argument(
        "--skip_test",
        action="store_false",
        dest="do_test",
        help="Skip evaluation on the test split after training.",
    )

    p.add_argument("--wandb_project", type=str, default="Emoji-reaction-coach-with-lora")
    p.add_argument("--model_dir", type=str, default="outputs/")
    
    args = p.parse_args()
    return args


def initialize_optimizers(cfg, G, F, DX, DY):
    # track all generator params (even frozen encoder params during initial training). 
    # This would allow us to transition easily to the full fine-tuning later on by simply toggling requires_grad=True
    # since the optimizers already track all the parameters from the start.
    opt_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(0.5,0.999))
    opt_F = optim.Adam(F.parameters(), lr=cfg.learning_rate, betas=(0.5,0.999))
    opt_DX = optim.Adam(DX.parameters(), lr=cfg.learning_rate, betas=(0.5,0.999))
    opt_DY = optim.Adam(DY.parameters(), lr=cfg.learning_rate, betas=(0.5,0.999))

    return opt_G, opt_F, opt_DX, opt_DY


def initialize_loss_functions(
    lambda_cyc_value: int = 10.0, 
    lambda_id_value: int = 5.0
):
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    lambda_cyc = lambda_cyc_value
    lambda_id = lambda_id_value
    
    return bce, l1, lambda_cyc, lambda_id
    

def main() -> None:
    """Entry point: parse args, prepare data, train, evaluate, and optionally test."""
    logging.basicConfig(level=logging.INFO)
    cfg = parse_args()
    load_environ_vars(cfg.wandb_project)
    
    # ---------- Run Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {DEVICE}")

    # reproducibility (optional, but less efficient if set)
    if cfg.set_seed:
        set_seed(cfg.seed)
        logger.info(f"Set seed: {cfg.seed}")
    else:
        logger.info(f"Skipping setting seed...")
        
    # ---------- Data Preprocessing ----------
    train_loader, val_loader, test_loader = prepare_dataset()
    
    # ---------- Models, Optimizers, & Loss Functions Initialization ----------
    # Initialize the generators (G, F) and discriminators (DX, DY)
    G, F, DX, DY = initialize_models()
    
    # Freeze generator encoderes for early training
    freeze_encoders(G, F)
    
    # Initialize optimizers
    opt_G, opt_F, opt_DX, opt_DY, = initialize_optimizers(cfg, G, F, DX, DY)  
    
    # Prepare Accelerator (uses hf accelerate to move models to correct device,
    # wrap in DDP if needed, shard the dataloader, and enable mixed-precision).
    accelerator = Accelerator(mixed_precision="fp16")
    G, F, DX, DY, opt_G, opt_F, opt_DX, opt_DY, train_loader, val_loader = accelerator.prepare(
        G, F, DX, DY, 
        opt_G, opt_F, opt_DX, opt_DY, 
        train_loader, val_loader
    )

    # Loss functions and scalers
    bce, l1, lambda_cyc, lambda_id = initialize_loss_functions()
    
    # ---------- Train & Checkpoint ----------
    
    # remember to unfreeze encoders during training after 1st epoch.


if __name__ == "__main__":
    main()