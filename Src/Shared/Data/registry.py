"""Dataset construction driven by a model bundle."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from Src.Shared.Config.model_config import ModelBundleSpec
from Src.Shared.Config.paths import bundle_paths

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_transform(bundle: ModelBundleSpec, *, train: bool = False):
    from torchvision import transforms

    size = bundle.input_shape[1]
    operations = []
    if train:
        operations.extend((transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip()))
    else:
        operations.extend((transforms.Resize(size), transforms.CenterCrop(size)))
    operations.extend((transforms.ToTensor(), transforms.Normalize(bundle.mean, bundle.std)))
    return transforms.Compose(operations)


class FilenameClassImageDataset(Dataset):
    """Flat image folder whose class is encoded as a filename prefix."""

    def __init__(self, image_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.transform = transform
        self.samples = []
        class_names = []
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            class_name = self._class_from_name(path.stem)
            self.samples.append((path, class_name))
            if class_name not in class_names:
                class_names.append(class_name)
        if not self.samples:
            raise ValueError(f"No images found under {self.image_dir}")
        self.classes = sorted(class_names)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[class_name]) for path, class_name in self.samples
        ]
        self.targets = [target for _, target in self.samples]

    @staticmethod
    def _class_from_name(stem: str) -> str:
        match = re.match(r"(.+)_\d+$", stem)
        if match:
            return match.group(1)
        return stem.rsplit("_", 1)[0] if "_" in stem else stem

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, target


def _build_neucls64_dataset(root: Path, *, train: bool, transform, num_classes: int):
    from torchvision import datasets

    split_candidates = ("train",) if train else ("val", "valid", "test")
    split_dir = next((root / name for name in split_candidates if (root / name).is_dir()), None)
    if split_dir is None:
        raise FileNotFoundError(
            f"NEU-CLS-64 split not found under {root}; expected one of {split_candidates}"
        )

    image_dir = split_dir / "images"
    if image_dir.is_dir():
        dataset = FilenameClassImageDataset(image_dir, transform=transform)
    else:
        dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    if len(dataset.classes) != num_classes:
        raise ValueError(
            f"neucls64 requires exactly {num_classes} classes, "
            f"found {len(dataset.classes)} under {split_dir}: {dataset.classes}"
        )
    return dataset


def build_dataset(
    bundle: ModelBundleSpec,
    split: str,
    *,
    data_root: str | Path | None = None,
    download: bool = False,
):
    from torchvision import datasets

    root = Path(data_root or bundle_paths(bundle.bundle_id).dataset_root)
    train = split == "train"
    transform = build_transform(bundle, train=train)
    if bundle.dataset_id == "cifar10":
        return datasets.CIFAR10(root=str(root), train=train, transform=transform, download=download)
    if bundle.dataset_id == "imagenet100":
        if download:
            raise ValueError(f"{bundle.dataset_id} download is not supported")
        directory = root / ("train" if train else "val")
        dataset = datasets.ImageFolder(str(directory), transform=transform)
        if len(dataset.classes) != bundle.num_classes:
            raise ValueError(
                f"{bundle.dataset_id} requires exactly {bundle.num_classes} classes, "
                f"found {len(dataset.classes)} under {directory}"
            )
        return dataset
    if bundle.dataset_id == "neucls64":
        if download:
            raise ValueError("neucls64 download is not supported")
        return _build_neucls64_dataset(
            root,
            train=train,
            transform=transform,
            num_classes=bundle.num_classes,
        )
    raise ValueError(f"Unsupported dataset_id: {bundle.dataset_id}")


def build_loader(
    bundle: ModelBundleSpec,
    split: str,
    *,
    batch_size=64,
    num_workers=0,
    data_root=None,
    download=False,
):
    dataset = build_dataset(bundle, split, data_root=data_root, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers)


class TestPackageDataset(Dataset):
    """Manifest-backed image dataset exported for device-side testing."""

    def __init__(self, bundle: ModelBundleSpec, package_root: str | Path):
        self.bundle = bundle
        self.package_root = Path(package_root)
        self.manifest_path = self.package_root / "manifest.csv"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Test package manifest not found: {self.manifest_path}")
        self.transform = build_transform(bundle, train=False)
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Test package manifest is empty: {self.manifest_path}")
        required = {"sample_id", "label", "relative_path"}
        missing = required.difference(self.rows[0])
        if missing:
            raise ValueError(
                f"Test package manifest missing columns {sorted(missing)}: {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        image_path = self.package_root / row["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Test package image not found: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row["label"])
        metadata = {
            "sample_id": row.get("sample_id", ""),
            "source_index": row.get("source_index", ""),
            "difficulty": row.get("difficulty", ""),
        }
        return tensor, label, metadata


def build_test_package_dataset(
    bundle: ModelBundleSpec,
    package_root: str | Path,
) -> TestPackageDataset:
    return TestPackageDataset(bundle, package_root)


def build_test_package_loader(
    bundle: ModelBundleSpec,
    package_root: str | Path,
    *,
    batch_size=1,
    num_workers=0,
):
    dataset = build_test_package_dataset(bundle, package_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
