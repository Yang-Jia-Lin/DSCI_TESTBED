"""Bundle-aware ablation matrix construction."""

from __future__ import annotations

import numpy as np

from Scripts.EvaluationCommon.solutions import no_early_exit_matrix, split_matrix
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import decode_split_row


def _middle_boundaries(paras: Paras) -> list[int]:
    final = int(paras.partition_manifest.final_boundary_id)
    return [int(value) for value in paras.partition_boundary_ids if 0 < int(value) < final]


def _representative_middle(paras: Paras, preferred: int | None = None) -> int:
    middle = _middle_boundaries(paras)
    if not middle:
        return int(paras.partition_manifest.final_boundary_id)
    if preferred in middle:
        return int(preferred)
    return int(middle[len(middle) // 2])


def _representative_three_tier(paras: Paras, X_opt: np.ndarray) -> tuple[int, int]:
    final = int(paras.partition_manifest.final_boundary_id)
    middle = _middle_boundaries(paras)
    if len(middle) < 2:
        return 0, final
    try:
        first, second = decode_split_row(np.asarray(X_opt)[0])
        if 0 < first < second < final:
            return int(first), int(second)
    except Exception:
        pass
    one_third = middle[len(middle) // 3]
    two_third = middle[(2 * len(middle)) // 3]
    if one_third >= two_third:
        one_third, two_third = middle[0], middle[-1]
    return int(one_third), int(two_third)


def build_ablation_cases(paras: Paras, X_opt: np.ndarray, Y_opt: np.ndarray) -> list[dict]:
    n = int(paras.n)
    final = int(paras.partition_manifest.final_boundary_id)
    opt_first, opt_second = _representative_three_tier(paras, X_opt)
    device_edge_first = _representative_middle(paras, opt_first)
    edge_cloud_second = _representative_middle(paras, opt_second)
    no_exit = no_early_exit_matrix(paras, n)
    with_exit = np.asarray(Y_opt, dtype=np.float32)

    placements = [
        ("single_device", "Single Device", final, final),
        ("single_edge", "Single Edge", 0, final),
        ("single_cloud", "Single Cloud", 0, 0),
        ("two_device_edge", "Two-tier Device-Edge", device_edge_first, final),
        ("two_edge_cloud", "Two-tier Edge-Cloud", 0, edge_cloud_second),
        ("three_device_edge_cloud", "Three-tier Device-Edge-Cloud", opt_first, opt_second),
    ]
    cases: list[dict] = []
    for group, label, first, second in placements:
        X = split_matrix(paras, first, second, n)
        cases.append(
            {
                "group": group,
                "name": f"{label} / no EE",
                "X": X,
                "Y": no_exit,
            }
        )
        cases.append(
            {
                "group": group,
                "name": f"{label} / with EE",
                "X": X,
                "Y": with_exit,
            }
        )
    cases.append(
        {
            "group": "ours",
            "name": "Ours / optimized split + EE",
            "X": np.asarray(X_opt, dtype=np.float32),
            "Y": with_exit,
        }
    )
    return cases

