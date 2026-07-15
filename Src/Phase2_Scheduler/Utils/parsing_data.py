"""Read bundle-scoped exit curves and decode partition matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from Src.Shared.Partitioning.split_actions import decode_split_row


def parsing_rate_and_acc(paras, table_path: str | Path | None = None):
    path = Path(table_path or paras.bundle_paths.offline_table_path)
    frame = pd.read_csv(path)
    # Some synchronized Phase-1 CSVs were written with a space after each
    # delimiter (for example ``" final_accuracy"``).  Column names are part of
    # the bundle contract, so normalize harmless surrounding whitespace before
    # validating the three-exit schema.
    frame.columns = [str(column).strip() for column in frame.columns]
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
    return decode_split_row(x_row)


def split_points_matrix(X: np.ndarray) -> np.ndarray:
    return np.array([_decode_split_points(row) for row in X], dtype=int)
