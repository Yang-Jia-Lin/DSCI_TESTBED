"""Phase 2 objective weights and optimizer hyperparameter defaults."""

from dataclasses import dataclass
import os


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


@dataclass
class AlgoConfig:
    alpha: float = _env_float("DSCI_OBJECTIVE_ALPHA", 1.0)
    beta: float = _env_float("DSCI_OBJECTIVE_BETA", 5.0)

    # Simulation defaults in FLOP/s. Measured testbed runs load calibrated profiles.
    edge_max_freq: float = 20e9
    cloud_max_freq: float = 50e9

    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    buffer_size: int = 64


DEFAULT = AlgoConfig()
