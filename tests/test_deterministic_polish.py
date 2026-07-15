from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Src.Phase2_Scheduler.Optimizer.DSCI.agent import PPOAgent
from Src.Shared.Partitioning.split_actions import decode_split_row, encode_split_row


@pytest.mark.parametrize("exit_count", (1, 2, 3))
def test_deterministic_polish_searches_every_exit_without_regression(exit_count):
    boundaries = list(range(1, exit_count + 1))
    final_boundary = exit_count + 1
    m = final_boundary + 1
    targets = {
        boundary: 0.2 + 0.2 * index
        for index, boundary in enumerate(boundaries)
    }

    def objective(X, Y, F_e, F_c):
        del X, F_e, F_c
        return -sum(
            (float(Y[0, boundary]) - target) ** 2
            for boundary, target in targets.items()
        )

    fake_agent = SimpleNamespace(
        paras=SimpleNamespace(n=1, m=m, E=boundaries),
        policy=SimpleNamespace(
            x_pairs=torch.tensor([[0, final_boundary]], dtype=torch.long)
        ),
        _objective=objective,
    )
    X = encode_split_row(
        0, final_boundary, m, dtype=np.float32
    )[None, :]
    Y = np.ones((1, m), dtype=np.float32)
    for boundary in boundaries:
        Y[0, boundary] = 0.95
    F_e = np.zeros((1, 1), dtype=np.float32)
    F_c = np.zeros((1, 1), dtype=np.float32)
    baseline = objective(X, Y, F_e, F_c)

    polished_value, polished = PPOAgent._deterministic_polish(
        fake_agent, (X, Y, F_e, F_c)
    )
    X_out, Y_out, _, _ = polished

    assert polished_value >= baseline
    assert decode_split_row(X_out[0]) == (0, final_boundary)
    for boundary, target in targets.items():
        assert 0.0 <= float(Y_out[0, boundary]) <= 1.0
        assert float(Y_out[0, boundary]) == pytest.approx(target, abs=0.011)
    if exit_count == 3:
        assert float(Y_out[0, boundaries[2]]) != pytest.approx(0.95)

    non_exit_boundaries = set(range(m)) - set(boundaries)
    assert all(float(Y_out[0, boundary]) == 1.0 for boundary in non_exit_boundaries)
