import os
import logging
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class UTKFace(Dataset):
    """
    Assumes the unzipped UTKFace images live in  <root>/data/utkface_aligned_cropped/UTKFace
    File pattern:  {age}_{gender}_{race}_{yyyymmddHHMMSS}.jpg
    """

    def __init__(self, root: str, transform: T.Compose | None = None):
        self.root = (
            Path(root) / "utkface_aligned_cropped" / "UTKFace"
        )  # or "UTKFace" for the unaligned and varied original version.
        self.files = sorted(f for f in self.root.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(
                f"No UTKFace JPG files found in {self.root}/data/."
                "Did you unzip the dataset into that folder?"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        age = int(path.name.split("_")[0])  # first token of file name is age
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, age


def make_unpaired_loader(
    root: str,
    split: str,  # "train" | "valid" | "test"
    transform: T.Compose,
    batch_size: int = 4,
    num_workers: int = 1,
    limit: int | None = None,  # per-domain cap
    seed: int = 42,
    young_max: int = 30,  # 0-30
    old_min: int = 55,  # 55+
):
    full_ds = UTKFace(root, transform)

    # Split into young, old indices
    rng = torch.Generator().manual_seed(seed)
    young_idx = []
    old_idx = []

    for i, f in enumerate(full_ds.files):
        age = int(f.name.split("_")[0])
        if age <= young_max:
            young_idx.append(i)
        elif age >= old_min:
            old_idx.append(i)

    if not young_idx or not old_idx:
        raise ValueError(
            "Age thresholds left one split empty; adjust young_max/old_min"
        )

    # Deterministic shuffle and dataset split (80, 10, 10)
    def split_indices(idxs: list[int]):
        idxs = torch.tensor(
            idxs
        ).long()  # torch Subset() requires integer tensors, not floats. That is why we add .long()
        idxs = idxs[torch.randperm(len(idxs), generator=rng)]
        n = len(idxs)
        train = int(0.8 * n)
        valid = int(0.9 * n)
        return {"train": idxs[:train], "valid": idxs[train:valid], "test": idxs[valid:]}

    part_y = split_indices(young_idx)[split]
    part_o = split_indices(old_idx)[split]

    # Limit per domain
    if limit is not None:
        part_y = part_y[:limit]
        part_o = part_o[:limit]

    # Wrap subsets in unpaird Dataset
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

    young_ds = Subset(full_ds, part_y)
    old_ds = Subset(full_ds, part_o)
    paired = Unpaired(young_ds, old_ds)

    logger.info(
        f"- UTK {split}: young={len(young_ds)}  old={len(old_ds)}" f"(limit={limit})"
    )
    return DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


def prepare_dataset(
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    num_workers: int = 2,
    center_crop_size: int = 178,
    resize_size: int = 256,
    train_size: int | None = None,  # None = use all
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int = 42,
):
    data_dir = Path(__file__).resolve().parents[2] / "data"
    os.makedirs(data_dir, exist_ok=True)

    transform = T.Compose(
        [
            T.CenterCrop(center_crop_size),
            T.Resize(resize_size),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # loaders
    logger.info("Initializing dataset...")
    train_loader = make_unpaired_loader(
        str(data_dir),
        "train",
        transform,
        train_batch_size,
        num_workers,
        train_size,
        seed,
    )
    val_loader = make_unpaired_loader(
        str(data_dir), "valid", transform, eval_batch_size, num_workers, val_size, seed
    )
    test_loader = make_unpaired_loader(
        str(data_dir), "test", transform, eval_batch_size, num_workers, test_size, seed
    )
    logger.info("Done.")
    return train_loader, val_loader, test_loader
