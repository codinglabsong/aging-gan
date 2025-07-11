import os
import logging
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from dataclasses import dataclass
from torchvision.datasets import CelebA

logger = logging.getLogger(__name__)


def make_unpaired_loader(
    root, split, transform, batch_size=4, num_workers=1, limit=None
):
    # download split
    full = CelebA(
        root=root,
        split=split,
        download=True,
        transform=transform,
    )

    # locate the "Young" attribute index
    young_attr_i = full.attr_names.index("Young")
    attrs = full.attr
    # split indices
    young_idx = (attrs[:, young_attr_i] == 1).nonzero(as_tuple=True)[0]
    old_idx = (attrs[:, young_attr_i] == 0).nonzero(as_tuple=True)[0]

    # limit each domain samples
    if limit is not None:
        gen = torch.Generator()
        perm_y = torch.randperm(len(young_idx), generator=gen)[:limit]
        perm_o = torch.randperm(len(old_idx), generator=gen)[:limit]
        young_idx = young_idx[perm_y]
        old_idx = old_idx[perm_o]

    # build subsets
    young_ds = Subset(full, young_idx)
    old_ds = Subset(full, old_idx)

    # unpaird wrapper
    @dataclass
    class Unpaired(Dataset):
        a: Dataset
        b: Dataset

        def __len__(self) -> int:
            return max(len(self.a), len(self.b))

        def __getitem__(self, idx: int):
            x, _ = self.a[idx % len(self.a)]
            y, _ = self.b[idx % len(self.b)]
            return x, y

    paired = Unpaired(young_ds, old_ds)
    logger.info(f"- Finished spliting: {split} ({limit} examples)")
    return DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def prepare_dataset(
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    num_workers: int = 2,
    center_crop_size: int = 178,
    resize_size: int = 256,
    train_size: int = 10,
    val_size: int = 8,
    test_size: int = 8,
):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    transform = T.Compose(
        [
            T.CenterCrop(center_crop_size),
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # loaders
    logger.info("Initializing dataset...")
    train_loader = make_unpaired_loader(
        str(data_dir), "train", transform, train_batch_size, num_workers, train_size
    )
    val_loader = make_unpaired_loader(
        str(data_dir), "valid", transform, eval_batch_size, num_workers, val_size
    )
    test_loader = make_unpaired_loader(
        str(data_dir), "test", transform, eval_batch_size, num_workers, test_size
    )
    logger.info("Done.")
    return train_loader, val_loader, test_loader
