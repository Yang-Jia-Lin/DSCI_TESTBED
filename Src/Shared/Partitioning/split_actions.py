"""Deployment split-pair semantics shared by scheduler and runtimes."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def final_boundary_id(boundary_ids: Iterable[int]) -> int:
    ids = sorted({int(value) for value in boundary_ids})
    if not ids or ids[0] != 0:
        raise ValueError("partition boundaries must include boundary 0")
    return int(ids[-1])


def is_valid_deployment_pair(first: int, second: int, final_boundary: int) -> bool:
    first = int(first)
    second = int(second)
    final_boundary = int(final_boundary)
    pure_cloud = first == second == 0
    pure_device = first == second == final_boundary
    normal_pipeline = 0 <= first < second <= final_boundary
    return pure_cloud or pure_device or normal_pipeline


def deployment_pair_kind(first: int, second: int, final_boundary: int) -> str:
    if not is_valid_deployment_pair(first, second, final_boundary):
        raise ValueError(
            f"invalid deployment pair ({first}, {second}) for final {final_boundary}"
        )
    first = int(first)
    second = int(second)
    final_boundary = int(final_boundary)
    if first == second == final_boundary:
        return "pure_device"
    if first == 0 and second == final_boundary:
        return "pure_edge"
    if first == second == 0:
        return "pure_cloud"
    if second == final_boundary:
        return "device_edge"
    if first == 0:
        return "edge_cloud"
    return "device_edge_cloud"


def enumerate_deployment_pairs(boundary_ids: Iterable[int]) -> list[tuple[int, int]]:
    """Enumerate PPO/BF actions as explicit deployment modes.

    The set is:
    pure_device, pure_edge, pure_cloud, device_edge, edge_cloud, and three-part.
    """
    ids = sorted({int(value) for value in boundary_ids})
    final = final_boundary_id(ids)
    middle = [value for value in ids if 0 < value < final]

    pairs: list[tuple[int, int]] = [
        (final, final),
        (0, final),
        (0, 0),
    ]
    pairs.extend((b1, final) for b1 in middle)
    pairs.extend((0, b2) for b2 in middle)
    pairs.extend((b1, b2) for b1 in middle for b2 in middle if b1 < b2)

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    valid = set(ids)
    for first, second in pairs:
        pair = (int(first), int(second))
        if pair in seen:
            continue
        if first not in valid or second not in valid:
            continue
        if not is_valid_deployment_pair(first, second, final):
            continue
        ordered.append(pair)
        seen.add(pair)
    return ordered


def decode_split_row(x_row: np.ndarray) -> tuple[int, int]:
    """Decode one X row into ``(b1, b2)``.

    Equal-boundary deployments are stored as a single one-hot entry because a
    matrix row cannot contain two distinct markers at the same index.
    """
    row = np.asarray(x_row)
    ones = np.flatnonzero(row > 0.5)
    final = len(row) - 1
    if len(ones) == 0:
        return final, final
    if len(ones) == 1:
        boundary = int(ones[0])
        if boundary == 0:
            return 0, 0
        if boundary == final:
            return final, final
        return boundary, final
    if len(ones) != 2:
        raise ValueError(f"Partition row must contain one or two boundaries, got {len(ones)}")
    first, second = int(ones[0]), int(ones[1])
    if not is_valid_deployment_pair(first, second, final):
        raise ValueError(
            f"Invalid deployment pair ({first}, {second}) for final boundary {final}"
        )
    return first, second


def encode_split_row(first: int, second: int, width: int, *, dtype=np.float32) -> np.ndarray:
    final = int(width) - 1
    if not is_valid_deployment_pair(int(first), int(second), final):
        raise ValueError(
            f"Invalid deployment pair ({first}, {second}) for final boundary {final}"
        )
    row = np.zeros(int(width), dtype=dtype)
    row[int(first)] = 1.0
    row[int(second)] = 1.0
    return row
