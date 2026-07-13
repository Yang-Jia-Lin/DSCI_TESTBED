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
    pretrained_source: str | None = None
    interpolation: str = "bilinear"
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
_NEUCLS64_RESNET = {
    "dataset_id": "neucls64",
    "num_classes": 6,
    "input_shape": (3, 227, 227),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
_NEUCLS64_DEIT = {
    "dataset_id": "neucls64",
    "num_classes": 6,
    "input_shape": (3, 224, 224),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
_CIFAR10_DEIT = {
    **_CIFAR10,
    "input_shape": (3, 224, 224),
}
_EXITS = (
    ExitSpec("after_layer2", "layer2"),
    ExitSpec("after_layer3", "layer3"),
)
_RESNET101_EXITS = (
    ExitSpec("after_layer3_block5", "layer3.4"),
    ExitSpec("after_layer3_block10", "layer3.9"),
    ExitSpec("after_layer3_block15", "layer3.14"),
    ExitSpec("after_layer3_block20", "layer3.19"),
    ExitSpec("after_layer4", "layer4"),
)
_DEIT_SMALL_EXITS = (
    ExitSpec("after_block4", "blocks.3"),
    ExitSpec("after_block8", "blocks.7"),
)
_RESNET50_V2_EXITS = (
    ExitSpec("after_layer1", "layer1"),
    ExitSpec("after_layer2", "layer2"),
    ExitSpec("after_layer3", "layer3"),
)
_VIT_BASE_EXITS = (
    ExitSpec("after_block3", "blocks.2"),
    ExitSpec("after_block6", "blocks.5"),
    ExitSpec("after_block9", "blocks.8"),
)


def _bundle(
    architecture: str,
    dataset: dict,
    exits: tuple[ExitSpec, ...] = _EXITS,
) -> ModelBundleSpec:
    return ModelBundleSpec(
        bundle_id=f"{architecture.lower()}-{dataset['dataset_id']}-ee-v1",
        architecture=architecture.lower(),
        exits=exits,
        **dataset,
    )


def _experiment_bundle(architecture, dataset, exits, *, pretrained_source, interpolation):
    return ModelBundleSpec(
        bundle_id=f"{architecture}-{dataset['dataset_id']}", architecture=architecture,
        exits=exits, pretrained_source=pretrained_source, interpolation=interpolation, **dataset,
    )


BUNDLE_REGISTRY: dict[str, ModelBundleSpec] = {
    spec.bundle_id: spec
    for spec in (
        _bundle("resnet18", _CIFAR10),
        _bundle("resnet50", _CIFAR10),
        _bundle("resnet101", _CIFAR10, _RESNET101_EXITS),
        _bundle("resnet50", _NEUCLS64_RESNET),
        _bundle("resnet18", _IMAGENET100),
        _bundle("resnet50", _IMAGENET100),
        _bundle("resnet101", _IMAGENET100, _RESNET101_EXITS),
        _bundle("deit-small", _CIFAR10_DEIT, _DEIT_SMALL_EXITS),
        _bundle("deit-small", _NEUCLS64_DEIT, _DEIT_SMALL_EXITS),
        _bundle("deit-small", _IMAGENET100, _DEIT_SMALL_EXITS),
    )
}

for dataset in (_CIFAR10, _IMAGENET100, _NEUCLS64_RESNET):
    resnet_dataset = dict(dataset, input_shape=(3, 224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    spec = _experiment_bundle("resnet50", resnet_dataset, _RESNET50_V2_EXITS,
        pretrained_source="torchvision/resnet50:IMAGENET1K_V2", interpolation="bilinear")
    BUNDLE_REGISTRY[spec.bundle_id] = spec
for dataset in (_CIFAR10_DEIT, _IMAGENET100, _NEUCLS64_DEIT):
    vit_dataset = dict(dataset, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    spec = _experiment_bundle("vit-base", vit_dataset, _VIT_BASE_EXITS,
        pretrained_source="timm/vit_base_patch16_224.orig_in21k_ft_in1k", interpolation="bicubic")
    BUNDLE_REGISTRY[spec.bundle_id] = spec

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
