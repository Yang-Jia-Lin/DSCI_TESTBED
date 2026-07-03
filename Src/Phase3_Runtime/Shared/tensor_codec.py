"""Helpers for moving tensors across process and socket boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def tensors_to_cpu(value):
    """Return a copy of nested payload data with torch tensors detached on CPU."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(tensors_to_cpu(item) for item in value)
    return value
