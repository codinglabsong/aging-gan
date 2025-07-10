import os
import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)

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
