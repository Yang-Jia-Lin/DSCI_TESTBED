"""Validated model-bundle specifications shared by every phase."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExitSpec:
    exit_id: str
    attach_point: str


@dataclass(frozen=True)
class ModelBundleSpec:
    bundle_id: str
    architecture: str
    dataset_id: str
    num_classes: int
    input_shape: tuple[int, int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    exits: tuple[ExitSpec, ...]
    version: int = 1

    @property
    def artifact_prefix(self) -> str:
        return self.bundle_id

    @property
    def manifest_id(self) -> str:
        return f"{self.bundle_id}-partition-v{self.version}"


_CIFAR10 = {
    "dataset_id": "cifar10",
    "num_classes": 10,
    "input_shape": (3, 227, 227),
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}
_IMAGENET100 = {
    "dataset_id": "imagenet100",
    "num_classes": 100,
    "input_shape": (3, 224, 224),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
_EXITS = (
    ExitSpec("after_layer2", "layer2"),
    ExitSpec("after_layer3", "layer3"),
)


def _bundle(architecture: str, dataset: dict) -> ModelBundleSpec:
    return ModelBundleSpec(
        bundle_id=f"{architecture.lower()}-{dataset['dataset_id']}-ee-v1",
        architecture=architecture.lower(),
        exits=_EXITS,
        **dataset,
    )


BUNDLE_REGISTRY: dict[str, ModelBundleSpec] = {
    spec.bundle_id: spec
    for spec in (
        _bundle("resnet18", _CIFAR10),
        _bundle("resnet50", _CIFAR10),
        _bundle("resnet18", _IMAGENET100),
        _bundle("resnet50", _IMAGENET100),
    )
}

DEFAULT_BUNDLE_ID = "resnet50-cifar10-ee-v1"


def get_bundle(bundle_id: str | None = None) -> ModelBundleSpec:
    selected = bundle_id or os.environ.get("DSCI_BUNDLE_ID") or DEFAULT_BUNDLE_ID
    try:
        return BUNDLE_REGISTRY[selected]
    except KeyError as exc:
        raise KeyError(
            f"Unknown bundle_id {selected!r}; known bundles: "
            f"{', '.join(sorted(BUNDLE_REGISTRY))}"
        ) from exc


def require_bundle_id(payload: dict) -> ModelBundleSpec:
    if "bundle_id" not in payload:
        raise KeyError("bundle_id is required; legacy model_name is not supported")
    if "model_name" in payload:
        raise ValueError("legacy model_name is not supported; use bundle_id")
    return get_bundle(str(payload["bundle_id"]))
