from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from Src.Models.model_config import RESNET50 as MODEL_CFG


VALID_DIFFICULTIES = {"easy", "medium", "hard"}
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_IMAGE_SIZE = 227


@dataclass(frozen=True)
class DifficultyRecord:
    image_id: int
    true_label: int
    difficulty: str
    confidence: float


def default_difficulty_table_path() -> Path:
    return (
        MODEL_CFG.profile_dir
        / f"{MODEL_CFG.artifact_prefix}_difficulty_labeled.csv"
    )


def build_cifar10_test_transform():
    """Return the canonical CIFAR-10 test transform used by sim and deploy."""

    from PIL import Image

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)

    def transform(image):
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"Expected a 3-D image array, got shape {array.shape}")
            if array.shape[0] == 3:
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                array = np.clip(array, 0.0, 1.0) * 255.0
                array = array.astype(np.uint8)
            pil_image = Image.fromarray(array, mode="RGB")

        resized = pil_image.resize((CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE), resample)
        data = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1)
        return (tensor - mean) / std

    return transform


def _normalise_difficulty_filter(
    difficulty: str | Iterable[str] | None,
) -> set[str] | None:
    if difficulty is None:
        return None
    if isinstance(difficulty, str):
        values = {difficulty}
    else:
        values = set(difficulty)
    invalid = values - VALID_DIFFICULTIES
    if invalid:
        raise ValueError(
            f"Unknown difficulty labels: {sorted(invalid)}. "
            f"Expected labels: {sorted(VALID_DIFFICULTIES)}"
        )
    return values


def _resolve_cifar10_batch_dir(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = [root, root / "cifar-10-batches-py"]
    for candidate in candidates:
        if (candidate / "test_batch").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find CIFAR-10 test_batch. Expected either "
        f"{root / 'test_batch'} or {root / 'cifar-10-batches-py' / 'test_batch'}."
    )


def _download_cifar10_if_needed(data_root: str | Path) -> None:
    root = Path(data_root)
    download_root = root.parent if root.name == "cifar-10-batches-py" else root
    import torchvision

    torchvision.datasets.CIFAR10(
        root=str(download_root),
        train=False,
        download=True,
    )


def _load_cifar10_test_batch(data_root: str | Path, download: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        batch_dir = _resolve_cifar10_batch_dir(data_root)
    except FileNotFoundError:
        if not download:
            raise
        _download_cifar10_if_needed(data_root)
        batch_dir = _resolve_cifar10_batch_dir(data_root)

    with (batch_dir / "test_batch").open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def _load_difficulty_records(
    difficulty_table_path: str | Path,
    difficulty: str | Iterable[str] | None,
) -> list[DifficultyRecord]:
    table_path = Path(difficulty_table_path)
    labels = _normalise_difficulty_filter(difficulty)
    required = {"image_id", "true_label", "confidence", "difficulty"}

    with table_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{table_path} is missing required columns: {sorted(missing)}"
            )

        records = []
        seen: set[int] = set()
        for row in reader:
            image_id = int(row["image_id"])
            if image_id in seen:
                raise ValueError(f"Duplicate image_id value in {table_path}: {image_id}")
            seen.add(image_id)

            label = str(row["difficulty"])
            if label not in VALID_DIFFICULTIES:
                raise ValueError(
                    f"Unknown difficulty label {label!r} in {table_path}. "
                    f"Expected labels: {sorted(VALID_DIFFICULTIES)}"
                )
            if labels is not None and label not in labels:
                continue

            records.append(
                DifficultyRecord(
                    image_id=image_id,
                    true_label=int(row["true_label"]),
                    difficulty=label,
                    confidence=float(row["confidence"]),
                )
            )

    return sorted(records, key=lambda record: record.image_id)


class CIFAR10TestDataset(Dataset):
    """CIFAR-10 test dataset with optional offline difficulty filtering."""

    def __init__(
        self,
        data_root: str | Path,
        difficulty_table_path: str | Path | None = None,
        difficulty: str | Iterable[str] | None = None,
        train: bool = False,
        transform=None,
        download: bool = False,
        include_difficulty_metadata: bool = False,
        include_image_id: bool = False,
    ):
        if train:
            raise ValueError("CIFAR10TestDataset only supports the CIFAR-10 test split.")
        if difficulty is not None and difficulty_table_path is None:
            raise ValueError("difficulty_table_path is required when difficulty is set.")

        self.images, self.labels = _load_cifar10_test_batch(data_root, download)
        self.transform = transform or build_cifar10_test_transform()
        self.include_difficulty_metadata = include_difficulty_metadata
        self.include_image_id = include_image_id

        if difficulty_table_path is None:
            self.records = [
                DifficultyRecord(
                    image_id=image_id,
                    true_label=int(label),
                    difficulty="",
                    confidence=float("nan"),
                )
                for image_id, label in enumerate(self.labels)
            ]
        else:
            self.records = _load_difficulty_records(difficulty_table_path, difficulty)
            for record in self.records:
                if record.image_id < 0 or record.image_id >= len(self.labels):
                    raise ValueError(
                        f"image_id={record.image_id} in {difficulty_table_path} "
                        f"is outside CIFAR-10 test range [0, {len(self.labels) - 1}]"
                    )
                label = int(self.labels[record.image_id])
                if label != record.true_label:
                    raise ValueError(
                        f"CIFAR-10 label mismatch for image_id={record.image_id}: "
                        f"dataset={label}, table={record.true_label}"
                    )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = self.transform(self.images[record.image_id])
        item = [image, int(self.labels[record.image_id])]

        if self.include_difficulty_metadata:
            item.extend([record.difficulty, float(record.confidence)])
        if self.include_image_id:
            item.append(record.image_id)
        return tuple(item)


class DifficultyAwareDataset(CIFAR10TestDataset):
    """Backward-compatible dataset that returns difficulty metadata by default."""

    def __init__(
        self,
        data_root: str | Path,
        difficulty_table_path: str | Path,
        difficulty: str | Iterable[str] | None,
        train: bool = False,
        transform=None,
        download: bool = False,
        include_image_id: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            difficulty_table_path=difficulty_table_path,
            difficulty=difficulty,
            train=train,
            transform=transform,
            download=download,
            include_difficulty_metadata=True,
            include_image_id=include_image_id,
        )


def iter_cifar10_test_samples(
    data_root: str | Path = MODEL_CFG.data_dir / MODEL_CFG.dataset_name / "cifar-10-batches-py",
    difficulty_table_path: str | Path | None = None,
    difficulty: str | Iterable[str] | None = None,
    include_difficulty_metadata: bool = False,
    include_image_id: bool = False,
    download: bool = False,
) -> Iterator[tuple]:
    dataset = CIFAR10TestDataset(
        data_root=data_root,
        difficulty_table_path=difficulty_table_path,
        difficulty=difficulty,
        download=download,
        include_difficulty_metadata=include_difficulty_metadata,
        include_image_id=include_image_id,
    )
    for sample in dataset:
        image = sample[0].unsqueeze(0)
        yield (image, *sample[1:])
