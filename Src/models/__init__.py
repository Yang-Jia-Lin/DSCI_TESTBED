"""Model package exports.

Keep package import lightweight so configuration modules can be imported without
requiring torch to be installed. Model classes are loaded lazily on demand.
"""

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    "MultiEEResNet50",
    "freeze_layers",
]


def __getattr__(name):
    if name in __all__:
        from Src.Models.ModelNet.Resnet50 import (
            BasicBlock,
            Bottleneck,
            MultiEEResNet50,
            ResNet,
            freeze_layers,
        )

        exports = {
            "BasicBlock": BasicBlock,
            "Bottleneck": Bottleneck,
            "ResNet": ResNet,
            "MultiEEResNet50": MultiEEResNet50,
            "freeze_layers": freeze_layers,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
