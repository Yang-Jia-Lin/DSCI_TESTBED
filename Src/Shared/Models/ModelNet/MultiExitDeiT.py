"""Multi-exit DeiT-Small wrapper.

The implementation uses timm when available.  The dependency is optional so
repositories without timm can still import readiness/reporting code; attempting
to instantiate a DeiT bundle explains the missing dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from Src.Shared.Config.model_config import ModelBundleSpec


class MissingTimmError(ImportError):
    pass


class MultiExitDeiT(nn.Module):
    def __init__(self, bundle: ModelBundleSpec):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise MissingTimmError(
                "DeiT-Small bundles require the optional 'timm' package. "
                "Install timm in the conda DSCI environment before training or profiling DeiT."
            ) from exc
        self.bundle = bundle
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=False,
            num_classes=bundle.num_classes,
        )
        embed_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.exit_heads = nn.ModuleDict(
            {item.exit_id: nn.Linear(embed_dim, bundle.num_classes) for item in bundle.exits}
        )
        self.exit_attach_points = {
            item.exit_id: item.attach_point for item in bundle.exits
        }

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        else:
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.backbone.pos_embed
            x = self.backbone.pos_drop(x)
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        return x

    def partition_segment_names(self) -> list[str]:
        return (
            ["patch_embed"]
            + [f"blocks.{index}" for index in range(len(self.backbone.blocks))]
            + ["norm", "final_classifier"]
        )

    def execute_partition_segment(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "patch_embed":
            return self._embed(x)
        if name.startswith("blocks."):
            index = int(name.split(".")[1])
            return self.backbone.blocks[index](x)
        if name == "norm":
            return self.backbone.norm(x)[:, 0]
        if name == "final_classifier":
            return self.backbone.head(x)
        raise ValueError(f"Unknown DeiT segment: {name}")

    def resolve_partition_segment(self, name: str):
        return lambda x: self.execute_partition_segment(name, x)

    def classify_exit(self, exit_id: str, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = self.backbone.norm(features)[:, 0]
        return self.exit_heads[exit_id](features)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        x = self._embed(x)
        for index, block in enumerate(self.backbone.blocks):
            x = block(x)
            features[f"blocks.{index}"] = x
        features["norm"] = self.backbone.norm(x)[:, 0]
        return features

    def forward(self, x: torch.Tensor, exit_id: str | None = None) -> torch.Tensor:
        features = self.forward_features(x)
        if exit_id is not None:
            return self.classify_exit(exit_id, features[self.exit_attach_points[exit_id]])
        return self.backbone.head(features["norm"])

    def final_classifier_module(self) -> nn.Module:
        return self.backbone.head

    def freeze_for_exit(self, exit_id: str | None) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        head = self.final_classifier_module() if exit_id is None else self.exit_heads[exit_id]
        for parameter in head.parameters():
            parameter.requires_grad = True

