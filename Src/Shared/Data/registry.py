"""Dataset construction driven by a model bundle."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from Src.Shared.Config.model_config import ModelBundleSpec
from Src.Shared.Config.paths import bundle_paths


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
            raise ValueError("ImageNet100 download is not supported")
        directory = root / ("train" if train else "val")
        dataset = datasets.ImageFolder(str(directory), transform=transform)
        if len(dataset.classes) != bundle.num_classes:
            raise ValueError(
                f"ImageNet100 requires exactly {bundle.num_classes} classes, "
                f"found {len(dataset.classes)} under {directory}"
            )
        return dataset
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
