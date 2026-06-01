from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def build_cifar10_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )


def _normalise_difficulty_filter(difficulty: str | Iterable[str] | None) -> set[str] | None:
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


class DifficultyAwareDataset(Dataset):
    """CIFAR-10 test dataset filtered by offline difficulty labels.

    Example:
        easy_set = DifficultyAwareDataset(
            data_root="Data/CIFAR10",
            difficulty_table_path="Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv",
            difficulty="easy",
        )
        loader = DataLoader(easy_set, batch_size=64, shuffle=False)
        # each batch: images, labels, difficulties, confidences
    """

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
        table_path = Path(difficulty_table_path)
        df = pd.read_csv(table_path)
        required = {"image_id", "true_label", "confidence", "difficulty"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{table_path} is missing required columns: {sorted(missing)}"
            )

        labels = _normalise_difficulty_filter(difficulty)
        if labels is not None:
            df = df[df["difficulty"].isin(labels)]

        if df["image_id"].duplicated().any():
            duplicated = df[df["image_id"].duplicated()]["image_id"].head().tolist()
            raise ValueError(f"Duplicate image_id values in {table_path}: {duplicated}")

        self.table = df.sort_values("image_id").reset_index(drop=True)
        self.include_image_id = include_image_id
        self.base_dataset = torchvision.datasets.CIFAR10(
            root=str(data_root),
            train=train,
            download=download,
            transform=transform,
        )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        image_id = int(row["image_id"])
        image, label = self.base_dataset[image_id]
        true_label = int(row["true_label"])
        if int(label) != true_label:
            raise ValueError(
                f"CIFAR-10 label mismatch for image_id={image_id}: "
                f"dataset={label}, table={true_label}"
            )

        item = (
            image,
            int(label),
            str(row["difficulty"]),
            float(row["confidence"]),
        )
        if self.include_image_id:
            return (*item, image_id)
        return item
