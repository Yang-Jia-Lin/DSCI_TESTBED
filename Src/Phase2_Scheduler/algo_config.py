"""Phase 2 objective weights and optimizer hyperparameter defaults."""

from dataclasses import dataclass


@dataclass
class AlgoConfig:
    alpha: float = 1.0
    beta: float = 5.0

    # Simulation defaults in FLOP/s. Measured testbed runs load calibrated profiles.
    edge_max_freq: float = 20e9
    cloud_max_freq: float = 50e9

    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    buffer_size: int = 64


DEFAULT = AlgoConfig()
