"""Compatibility exports for difficulty-aware CIFAR-10 loading."""

from Src.Deploy.Shared.dataloader import (
    CIFAR10TestDataset,
    DifficultyAwareDataset,
    VALID_DIFFICULTIES,
    build_cifar10_test_transform,
)

__all__ = [
    "CIFAR10TestDataset",
    "DifficultyAwareDataset",
    "VALID_DIFFICULTIES",
    "build_cifar10_test_transform",
]
