"""Bundle-aware model loader used by PyTorch runtime workers."""

from __future__ import annotations

import os

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Partitioning.manifest import PartitionManifest, validate_model_file

_MODELS = {}


def resolve_torch_device(device_name: str | None = None):
    import torch

    requested = device_name or os.environ.get("DSCI_PYTORCH_DEVICE", "auto")
    requested = str(requested).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"Requested DSCI_PYTORCH_DEVICE={requested!r}, "
            "but CUDA is unavailable; falling back to CPU"
        )
        return torch.device("cpu")
    return torch.device(requested)


def load_full_model(manifest: PartitionManifest, device_name: str | None = None):
    import torch

    device = resolve_torch_device(device_name)
    cache_key = (manifest.bundle_id, str(device))
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    from Src.Shared.Models.factory import build_model

    paths = bundle_paths(manifest.bundle_id)
    validate_model_file(manifest, paths.weight_path)
    model = build_model(get_bundle(manifest.bundle_id)).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {manifest.bundle_id} PyTorch model on {device}")
    _MODELS[cache_key] = model
    return model
