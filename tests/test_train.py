import torch
from types import SimpleNamespace
from aging_gan import train, model


import sys


def test_parse_args_defaults(monkeypatch):
    """CLI parser returns expected default arguments."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = train.parse_args()
    assert args.gen_lr == 2e-4
    assert args.disc_lr == 1e-4
    assert args.num_train_epochs == 100


def test_initialize_loss_functions_defaults():
    """Loss initializer provides default weights and criteria."""
    mse, l1, adv, cyc, ident = train.initialize_loss_functions()
    assert isinstance(mse, torch.nn.MSELoss)
    assert adv == 2.0
    assert cyc == 10.0
    assert ident == 7.0


def test_make_schedulers_decay():
    """Learning rate scheduler should decrease learning rate."""
    cfg = SimpleNamespace(num_train_epochs=4)
    models = model.initialize_models(ngf=8, ndf=8, n_blocks=1)
    opts = [torch.optim.SGD(m.parameters(), lr=1.0) for m in models]
    sched_G, _, _, _ = train.make_schedulers(cfg, *opts)
    sched_G.step()  # epoch 0
    assert opts[0].param_groups[0]["lr"] == 1.0
    sched_G.step(3)
    assert opts[0].param_groups[0]["lr"] < 1.0
