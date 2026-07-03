"""Helpers for moving tensors across process and socket boundaries."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

_MARKER = "__dsci_tensor_transport__"
_TENSOR_KIND = "tensor"
_NDARRAY_KIND = "ndarray"


def _transport_dtype() -> str:
    value = os.environ.get("DSCI_TENSOR_TRANSPORT_DTYPE", "float32")
    value = str(value).strip().lower()
    if value in {"", "none", "fp32", "float32"}:
        return "float32"
    if value in {"fp16", "float16"}:
        return "float16"
    raise ValueError(
        "DSCI_TENSOR_TRANSPORT_DTYPE must be one of: float32, fp32, none, "
        "float16, fp16"
    )


def tensors_to_cpu(value):
    """Return a copy of nested payload data with torch tensors detached on CPU."""
    return prepare_for_transport(value)


def prepare_for_transport(value):
    """Move nested tensors to CPU and optionally compress floating tensors for transit."""
    mode = _transport_dtype()
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if mode == "float16" and tensor.is_floating_point():
            return {
                _MARKER: True,
                "kind": _TENSOR_KIND,
                "transport_dtype": "float16",
                "tensor": tensor.to(torch.float16),
            }
        return tensor
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        if mode == "float16" and np.issubdtype(value.dtype, np.floating):
            return {
                _MARKER: True,
                "kind": _NDARRAY_KIND,
                "transport_dtype": "float16",
                "array": np.asarray(value, dtype=np.float16),
            }
        return value
    if isinstance(value, Mapping):
        return {key: prepare_for_transport(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(prepare_for_transport(item) for item in value)
    if isinstance(value, list):
        return [prepare_for_transport(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(prepare_for_transport(item) for item in value)
    return value


def restore_from_transport(value):
    """Restore nested transport-compressed tensors to runtime float32 values."""
    if isinstance(value, Mapping) and value.get(_MARKER) is True:
        kind = value.get("kind")
        if kind == _TENSOR_KIND:
            tensor = value["tensor"]
            if value.get("transport_dtype") == "float16":
                return tensor.float()
            return tensor
        if kind == _NDARRAY_KIND:
            array = value["array"]
            if value.get("transport_dtype") == "float16":
                try:
                    import numpy as np
                except ImportError:
                    return array
                return np.asarray(array, dtype=np.float32)
            return array
    if isinstance(value, Mapping):
        return {key: restore_from_transport(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(restore_from_transport(item) for item in value)
    if isinstance(value, list):
        return [restore_from_transport(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(restore_from_transport(item) for item in value)
    return value
