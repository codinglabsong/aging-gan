import argparse
import logging
import numpy as np
import torch
from typing import List
from pathlib import Path

from aging_gan.utils import set_seed, load_environ_vars

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
    
    
    # ---------- Model Initialization ----------
    
    # ---------- Train & Checkpoint ----------
