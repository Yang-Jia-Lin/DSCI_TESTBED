"""Bundle-aware model loader used by PyTorch runtime workers."""

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Partitioning.manifest import PartitionManifest, validate_model_file

_MODELS = {}


def load_full_model(manifest: PartitionManifest):
    if manifest.bundle_id in _MODELS:
        return _MODELS[manifest.bundle_id]
    import torch
    from Src.Shared.Models.ModelNet.MultiExitResNet import build_model

    paths = bundle_paths(manifest.bundle_id)
    validate_model_file(manifest, paths.weight_path)
    model = build_model(get_bundle(manifest.bundle_id))
    model.load_state_dict(torch.load(paths.weight_path, map_location="cpu", weights_only=True))
    model.eval()
    _MODELS[manifest.bundle_id] = model
    return model
