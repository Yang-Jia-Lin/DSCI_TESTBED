"""Read bundle-scoped exit curves and decode partition matrices."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parsing_rate_and_acc(paras, table_path: str | Path | None = None):
    path = Path(table_path or paras.bundle_paths.offline_table_path)
    frame = pd.read_csv(path)
    required = {"threshold", "final_accuracy"}
    for exit_id in paras.exit_ids:
        required.update({f"{exit_id}_rate", f"{exit_id}_accuracy"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is not a bundle exit-curve table; missing columns: {sorted(missing)}"
        )
    rows = max(101, len(frame))
    rates = np.zeros((rows, paras.m), dtype=np.float64)
    accs = np.zeros((rows, paras.m), dtype=np.float64)
    for row_index, row in frame.iterrows():
        for exit_id, boundary_id in zip(paras.exit_ids, paras.E):
            rates[row_index, boundary_id] = float(row[f"{exit_id}_rate"])
            accs[row_index, boundary_id] = float(row[f"{exit_id}_accuracy"])
        accs[row_index, paras.m - 1] = float(row["final_accuracy"])
    if len(frame) < rows:
        rates[len(frame) :] = rates[len(frame) - 1]
        accs[len(frame) :] = accs[len(frame) - 1]
    return rates, accs


def _decode_split_points(x_row: np.ndarray) -> Tuple[int, int]:
    ones = np.flatnonzero(x_row)
    if len(ones) != 2:
        raise ValueError(f"Partition row must contain exactly two boundaries, got {len(ones)}")
    return int(ones[0]), int(ones[1])


def split_points_matrix(X: np.ndarray) -> np.ndarray:
    return np.array([_decode_split_points(row) for row in X], dtype=int)
