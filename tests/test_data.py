from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from aging_gan import data


def create_utk_dataset(tmp_path, num_per_split=6):
    root = Path(tmp_path)
    ds_root = root / "utkface_aligned_cropped" / "UTKFace"
    ds_root.mkdir(parents=True)
    # create young images ages 18..(18+num_per_split-1)
    for i in range(num_per_split):
        age = 18 + i
        img = Image.new("RGB", (32, 32), color=(i, i, i))
        img.save(ds_root / f"{age}_0_0_202001010000{i}.jpg")
    for i in range(num_per_split):
        age = 40 + i
        img = Image.new("RGB", (32, 32), color=(i, i, i))
        img.save(ds_root / f"{age}_0_0_202001010100{i}.jpg")
    return root


def test_utkface_len_and_getitem(tmp_path):
    root = create_utk_dataset(tmp_path)
    ds = data.UTKFace(str(root))
    assert len(ds) == 12
    img, age = ds[0]
    assert isinstance(age, int)
    assert isinstance(img, Image.Image)


def test_make_unpaired_loader(tmp_path):
    root = create_utk_dataset(tmp_path)
    loader = data.make_unpaired_loader(
        str(root),
        "train",
        T.Compose([T.ToTensor()]),
        batch_size=2,
        num_workers=1,
        seed=0,
        young_max=23,
        old_min=40,
    )
    x, y = next(iter(loader))
    assert x.shape == y.shape
    assert x.shape[0] == 2
