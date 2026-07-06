"""Model factory shared by offline preparation and runtime code."""

from __future__ import annotations

import torch.nn as nn

from Src.Shared.Config.model_config import ModelBundleSpec


def build_model(bundle: ModelBundleSpec) -> nn.Module:
    if bundle.architecture.startswith("resnet"):
        from Src.Shared.Models.ModelNet.MultiExitResNet import build_model as build_resnet

        return build_resnet(bundle)
    if bundle.architecture == "deit-small":
        from Src.Shared.Models.ModelNet.MultiExitDeiT import MultiExitDeiT

        return MultiExitDeiT(bundle)
    raise ValueError(f"Unsupported architecture: {bundle.architecture}")


def freeze_for_exit(model: nn.Module, exit_id: str | None) -> None:
    if hasattr(model, "freeze_for_exit"):
        model.freeze_for_exit(exit_id)
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    if exit_id is None:
        if hasattr(model, "final_classifier_module"):
            head = model.final_classifier_module()
        elif hasattr(model, "fc"):
            head = model.fc
        else:
            head = model.head
    else:
        head = model.exit_heads[exit_id]
    for parameter in head.parameters():
        parameter.requires_grad = True

