import random
import numpy as np
import torch
from pathlib import Path
from aging_gan import utils


def test_set_seed_reproducibility():
    utils.set_seed(123)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(1)

    utils.set_seed(123)
    assert random.random() == a
    assert np.random.rand() == b
    assert torch.allclose(torch.rand(1), c)


def test_get_device_cpu():
    assert utils.get_device().type == "cpu"


def test_save_checkpoint(tmp_path):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)

    utils.save_checkpoint(
        1,
        model,
        model,
        model,
        model,
        opt,
        opt,
        opt,
        opt,
        sched,
        sched,
        sched,
        sched,
        kind="best",
    )
    ckpt_file = (
        Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "best.pth"
    )
    assert ckpt_file.exists()
    ckpt_file.unlink()
