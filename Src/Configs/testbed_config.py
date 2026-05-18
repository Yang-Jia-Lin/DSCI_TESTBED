"""Src/Configs/testbed_config.py"""

from dataclasses import dataclass


@dataclass
class NodeConfig:
    ip: str
    port: int


@dataclass
class TestbedConfig:
    edge: NodeConfig | None = None
    cloud: NodeConfig | None = None
    algo_server_port: int = 8080

    default_bw_d2e: float = 10.0
    default_bw_e2c: float = 50.0
    num_users: int = 10

    def __post_init__(self):
        if self.edge is None:
            self.edge = NodeConfig(ip="192.168.1.10", port=9001)
        if self.cloud is None:
            self.cloud = NodeConfig(ip="192.168.1.20", port=9002)


DEFAULT = TestbedConfig()
