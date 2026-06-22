"""Compatibility imports for backend-specific segment worker entrypoints."""

from __future__ import annotations

from Src.Phase3_Runtime.Shared.mnn_segment_worker import (
    execute_mnn_range,
    init_mnn_worker,
)
from Src.Phase3_Runtime.Shared.pytorch_segment_worker import (
    execute_pytorch_range,
    init_pytorch_worker,
)

__all__ = [
    "execute_mnn_range",
    "execute_pytorch_range",
    "init_mnn_worker",
    "init_pytorch_worker",
]
