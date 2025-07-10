import argparse
from accelerate import Accelerator
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from aging_gan.utils import set_seed, load_environ_vars, print_trainable_parameters
from aging_gan.data import prepare_dataset
from aging_gan.model import initialize_models, freeze_encoders, unfreeze_encoders

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()

    # hyperparams
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
        "--set_seed",
        action="store_true", # default=False
        help="Set seed for entire run for reproducibility.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed value for reproducibility."
    )
    p.add_argument(
        "--print_stats_after_batch", type=int, default=1, help="Print training metrics after certain batch steps."
    )
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


def make_schedulers(cfg, opt_G, opt_F, opt_DX, opt_DY):
    # keep lr constant constant for the first half, then linearly decay to 0 
    n_epochs = cfg.num_train_epochs
    start_decay = n_epochs // 2
    
    def _lr_lambda(epoch):
        if epoch < start_decay:
            return 1.0
        # linearly decay from 1.0 -> 0.0 from start_decay
        return max(0.0, (n_epochs - epoch) / (n_epochs - start_decay))
    
    sched_G = LambdaLR(opt_G, lr_lambda=_lr_lambda)
    sched_F = LambdaLR(opt_F, lr_lambda=_lr_lambda)
    sched_DX = LambdaLR(opt_DX, lr_lambda=_lr_lambda)
    sched_DY = LambdaLR(opt_DY, lr_lambda=_lr_lambda)
    
    return sched_G, sched_F, sched_DX, sched_DY

def perform_train_step(
    G, F, # generator models
    DX, DY, # discriminator models
    real_data,
    bce, l1, lambda_cyc, lambda_id, # loss functions and loss params
    opt_G, opt_F, # generator optimizers
    opt_DX, opt_DY, # discriminator optimizers
    accelerator,
):
    x, y = real_data
    # Generate fakes and reconstrucitons
    fake_x = F(y)
    fake_y = G(x)
    rec_x = F(fake_y)
    rec_y = G(fake_x)
    
    # ------ Update Discriminators ------
    # DX: real young vs fake young
    opt_DX.zero_grad(set_to_none=True) 
    real_logits = DX(x)
    real_targets = torch.ones_like(real_logits)
    real_loss = bce(real_logits, real_targets)
    
    fake_logits = DX(fake_x.detach())
    fake_targets = torch.zeros_like(fake_logits)
    fake_loss = bce(fake_logits, fake_targets)

    # DX loss + backprop + step
    loss_DX = 0.5 * (real_loss + fake_loss)
    accelerator.backward(loss_DX)
    opt_DX.step()
    # DY: real old vs fake old
    opt_DY.zero_grad(set_to_none=True) 
    real_logits = DY(y)
    real_targets = torch.ones_like(real_logits)
    real_loss = bce(real_logits, real_targets)
    
    fake_logits = DY(fake_y.detach())
    fake_targets = torch.zeros_like(fake_logits)
    fake_loss = bce(fake_logits, fake_targets)

    # DY loss + backprop + step
    loss_DY = 0.5 * (real_loss + fake_loss) # average loss to prevent discriminator learning "too quickly" compread to generators.
    accelerator.backward(loss_DY)
    opt_DY.step()
    
    # ------ Update Generators ------
    opt_G.zero_grad(set_to_none=True)
    opt_F.zero_grad(set_to_none=True)
    # Loss 1: adversarial terms
    fake_test_logits = DX(fake_x) # fake x logits
    loss_f_adv = bce(fake_test_logits, torch.ones_like(fake_test_logits))
    
    fake_test_logits = DY(fake_y) # fake y logits
    loss_g_adv = bce(fake_test_logits, torch.ones_like(fake_test_logits))
    # Loss 2: cycle terms
    loss_cyc = lambda_cyc * (l1(rec_x, x) + l1(rec_y, y))
    # Loss 3: identity terms
    loss_id = lambda_id * (l1(G(y), y) + l1(F(x), x))
    # Total loss
    loss_gen_total = loss_g_adv + loss_f_adv + loss_cyc + loss_id

    # Backprop + step
    accelerator.backward(loss_gen_total)
    opt_G.step()
    opt_F.step()
    
    return {
        "loss_DX": loss_DX.item(),
        "loss_DY": loss_DY.item(),
        "loss_f_adv": loss_f_adv.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_cyc": loss_cyc.item(),
        "loss_id": loss_id.item(),
        "loss_gen_total": loss_gen_total.item() 
    }
    
    
def perform_val_epoch(
    G, F, # generator models
    DX, DY, # discriminator models
    val_loader,
    bce, l1, lambda_cyc, lambda_id, # loss functions and loss params
):
    metrics = {
        "loss_DX": 0.0,
        "loss_DY": 0.0,
        "loss_f_adv": 0.0,
        "loss_g_adv": 0.0,
        "loss_cyc": 0.0,
        "loss_id": 0.0,
        "loss_gen_total": 0.0,
    }
    n_batches = 0
    
    with torch.no_grad():        
        for x, y in val_loader:        
            # Forward: Generate fakes and reconstrucitons
            fake_x = F(y)
            fake_y = G(x)
            rec_x = F(fake_y)
            rec_y = G(fake_x)
            
            # ------ Evaluate Discriminators ------
            # DX: real young vs fake young
            real_logits = DX(x)
            real_targets = torch.ones_like(real_logits)
            real_loss = bce(real_logits, real_targets)
            
            fake_logits = DX(fake_x)
            fake_targets = torch.zeros_like(fake_logits)
            fake_loss = bce(fake_logits, fake_targets)
            # DX loss
            loss_DX = 0.5 * (real_loss + fake_loss)
            
            # DY: real old vs fake old
            real_logits = DY(y)
            real_targets = torch.ones_like(real_logits)
            real_loss = bce(real_logits, real_targets)
            
            fake_logits = DY(fake_y)
            fake_targets = torch.zeros_like(fake_logits)
            fake_loss = bce(fake_logits, fake_targets)

            # DY loss
            loss_DY = 0.5 * (real_loss + fake_loss) # average loss to prevent discriminator learning "too quickly" compread to generators.
            
            # ------ Evaluate Generators ------
            # Loss 1: adversarial terms
            fake_test_logits = DX(fake_x) # fake x logits
            loss_f_adv = bce(fake_test_logits, torch.ones_like(fake_test_logits))
            
            fake_test_logits = DY(fake_y) # fake y logits
            loss_g_adv = bce(fake_test_logits, torch.ones_like(fake_test_logits))
            # Loss 2: cycle terms
            loss_cyc = lambda_cyc * (l1(rec_x, x) + l1(rec_y, y))
            # Loss 3: identity terms
            loss_id = lambda_id * (l1(G(y), y) + l1(F(x), x))
            # Total loss
            loss_gen_total = loss_g_adv + loss_f_adv + loss_cyc + loss_id
            
            # ------ Accumulate ------
            metrics["loss_DX"] += loss_DX.item()
            metrics["loss_DY"] += loss_DY.item()
            metrics["loss_f_adv"] += loss_f_adv.item()
            metrics["loss_g_adv"] += loss_g_adv.item()
            metrics["loss_cyc"] += loss_cyc.item()
            metrics["loss_id"] += loss_id.item()
            metrics["loss_gen_total"] += loss_gen_total.item()
            
            n_batches += 1
    
    # per-batch average
    for k in metrics:
        metrics[k] /= n_batches

    return metrics


def perform_epoch(
    cfg,
    train_loader, val_loader,
    G, F,
    DX, DY,
    bce, l1, lambda_cyc, lambda_id,
    opt_G, opt_F, # generator optimizers
    opt_DX, opt_DY, # discriminator optimizers
    sched_G, sched_F, sched_DX, sched_DY, # schedulers
    epoch,
    accelerator,
):
    """ Perform a single epoch."""
    # TRAINING
    logger.info("Training...")
    G.train()
    F.train()
    DX.train()
    DY.train()
    for batch_no, real_data in enumerate(train_loader):      
        train_metrics = perform_train_step(
            G, F, # generator models
            DX, DY, # discriminator models
            real_data,
            bce, l1, lambda_cyc, lambda_id, # loss functions and loss params
            opt_G, opt_F, # generator optimizers
            opt_DX, opt_DY, # discriminator optimizers
            accelerator
        )
        # Print statistics and generate iamge after every n-th batch
        if batch_no % cfg.print_stats_after_batch == 0:
            logger.info(f"loss_DX: {train_metrics['loss_DX']:.4f} | loss_DY: {train_metrics['loss_DY']:.4f} | loss_gen_total: {train_metrics['loss_gen_total']:.4f} | loss_g_adv: {train_metrics['loss_g_adv']:.4f} | loss_f_adv: {train_metrics['loss_f_adv']:.4f} | loss_cyc: {train_metrics['loss_cyc']:.4f} | loss_id: {train_metrics['loss_id']:.4f}")
            # generate_image(G, epoch, batch_no)
    # Step schedulers per epoch
    sched_G.step()
    sched_F.step()
    sched_DX.step()
    sched_DY.step()
    
    # EVALUATION
    logger.info("Evlauating...")
    G.eval()
    F.eval()
    DX.eval()
    DY.eval()
    val_metrics = perform_val_epoch(
        G, F, # generator models
        DX, DY, # discriminator models
        val_loader,
        bce, l1, lambda_cyc, lambda_id, # loss functions and loss params
    )
    logger.info(f"loss_DX: {val_metrics['loss_DX']:.4f} | loss_DY: {val_metrics['loss_DY']:.4f} | loss_gen_total: {val_metrics['loss_gen_total']:.4f} | loss_g_adv: {val_metrics['loss_g_adv']:.4f} | loss_f_adv: {val_metrics['loss_f_adv']:.4f} | loss_cyc: {val_metrics['loss_cyc']:.4f} | loss_id: {val_metrics['loss_id']:.4f}")
    # # Save models on epoch completion
    # save_models(G, F, DX, DY, lambda_cyc, lambda_id, epoch)
    # Clear memory after every epoch
    torch.cuda.empty_cache()


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
    
    # ---------- Models, Optimizers, Loss Functions, Schedulers Initialization ----------
    # Initialize the generators (G, F) and discriminators (DX, DY)
    G, F, DX, DY = initialize_models()
    # Freeze generator encoderes for training during early epochs
    logger.info("Parameters of generator G:")
    print_trainable_parameters(G)
    logger.info("Freezing encoders of generators...")
    freeze_encoders(G, F)
    logger.info("Parameters of generator G after freezing:")
    print_trainable_parameters(G)
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
    # Initialize schedulers (It it important this comes AFTER wrapping optimizers in accelerator)
    sched_G, sched_F, sched_DX, sched_DY = make_schedulers(cfg, opt_G, opt_F, opt_DX, opt_DY)
    # ---------- Train & Checkpoint ----------
    for epoch in range(1, cfg.num_train_epochs+1):
        logger.info(f"\nStarting epoch {epoch}...")
        # after 1 full epoch, unfreeze
        if epoch == 2:
            logger.info("Unfreezing encoders of generators...")
            unfreeze_encoders(G, F)
            logger.info("Parameters of generator G after unfreezing:")
            print_trainable_parameters(G)
            
        perform_epoch(
            cfg,
            train_loader, val_loader,
            G, F,
            DX, DY,
            bce, l1, lambda_cyc, lambda_id,
            opt_G, opt_F, # generator optimizers
            opt_DX, opt_DY, # discriminator optimizers
            sched_G, sched_F, sched_DX, sched_DY, # schedulers
            epoch,
            accelerator,
        )
    # Finished
    logger.info(f"Finished run.")


if __name__ == "__main__":
    main()