import torch
from types import SimpleNamespace
import torchvision.transforms as T
from aging_gan import data, model, train
from test_data import create_utk_dataset


class DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def autocast(self):
        from contextlib import nullcontext

        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(params, max_norm)


class DummyFID:
    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return torch.tensor(0.0)


def test_smoke_training(tmp_path, monkeypatch):
    root = create_utk_dataset(tmp_path)
    transform = T.Compose([T.ToTensor()])
    train_loader = data.make_unpaired_loader(
        str(root),
        "train",
        transform,
        batch_size=2,
        num_workers=1,
        seed=0,
        young_max=23,
        old_min=40,
    )
    val_loader = data.make_unpaired_loader(
        str(root),
        "valid",
        transform,
        batch_size=2,
        num_workers=1,
        seed=0,
        young_max=23,
        old_min=40,
    )

    G, F, DX, DY = model.initialize_models(ngf=4, ndf=4, n_blocks=1)
    opt_cfg = SimpleNamespace(
        gen_lr=1e-3, disc_lr=1e-3, weight_decay=0.0, num_train_epochs=1
    )
    opt_G, opt_F, opt_DX, opt_DY = train.initialize_optimizers(opt_cfg, G, F, DX, DY)
    sched_G, sched_F, sched_DX, sched_DY = train.make_schedulers(
        opt_cfg, opt_G, opt_F, opt_DX, opt_DY
    )
    mse, l1, adv, cyc, ident = train.initialize_loss_functions()
    accelerator = DummyAccelerator()
    fid = DummyFID()
    monkeypatch.setattr(train, "wandb", SimpleNamespace(log=lambda *a, **k: None))
    monkeypatch.setattr(train, "generate_and_save_samples", lambda *a, **k: None)
    cfg = SimpleNamespace(steps_for_logging_metrics=1, num_sample_generations_to_save=1)
    metrics = train.perform_epoch(
        cfg,
        train_loader,
        val_loader,
        G,
        F,
        DX,
        DY,
        mse,
        l1,
        adv,
        cyc,
        ident,
        opt_G,
        opt_F,
        opt_DX,
        opt_DY,
        sched_G,
        sched_F,
        sched_DX,
        sched_DY,
        0,
        accelerator,
        fid,
    )
    assert "val/loss_gen_total" in metrics
