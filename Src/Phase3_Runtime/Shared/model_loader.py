from Src.Shared.Config.paths import RESNET50_PATHS as MODEL_PATHS
from Src.Shared.Partitioning.manifest import PartitionManifest, validate_model_file

FULL_MODEL_PATH = MODEL_PATHS.resolve_weight_path()

EXIT_LAYER_BY_STAGE = {
    2: 57,
    3: 103,
    4: 128,
}

_MODEL = None


def load_full_model(manifest: PartitionManifest | None = None):
    global _MODEL
    if manifest is not None:
        if manifest.model_name != MODEL_PATHS.model.name:
            raise ValueError(
                f"Manifest model {manifest.model_name!r} != runtime model "
                f"{MODEL_PATHS.model.name!r}"
            )
        validate_model_file(manifest, MODEL_PATHS.resolve_weight_path())
    if _MODEL is None:
        import torch

        from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50

        model = MultiEEResNet50(
            Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
        )
        state_dict = torch.load(
            MODEL_PATHS.resolve_weight_path(), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        _MODEL = model
    return _MODEL


def stage_end_from_partition_boundary(boundary, default):
    if boundary is None:
        return int(default)
    boundary = int(boundary)
    if boundary <= 4:
        return 0
    if boundary <= 27:
        return 1
    if boundary <= 57:
        return 2
    if boundary <= 103:
        return 3
    return 4


def threshold_for_stage(exit_thresholds, stage):
    exit_layer = EXIT_LAYER_BY_STAGE.get(stage)
    if exit_layer is None:
        return None, None
    if stage == 4:
        return exit_layer, None
    return exit_layer, float(exit_thresholds[str(exit_layer)])


# Export target: ONNX/MNN. Keep forward_partial traceable —
# avoid dynamic Python control flow inside torch.no_grad blocks.
