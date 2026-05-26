"""Algorithm defaults for DSCI."""

from dataclasses import dataclass


@dataclass
class AlgoConfig:
    alpha: float = 1.0
    beta: float = 5.0

    edge_max_freq: float = 20.0
    cloud_max_freq: float = 50.0

    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    buffer_size: int = 64


DEFAULT = AlgoConfig()
