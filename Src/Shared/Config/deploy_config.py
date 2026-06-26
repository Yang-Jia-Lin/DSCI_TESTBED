"""Static runtime topology: node addresses, ports, and network fallbacks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestbedConfig:
    """Addresses, ports, and fallback measurements used by deploy entrypoints."""

    listen_host: str = "0.0.0.0"

    # device_host: str = "100.106.152.89"
    # edge_host: str = "100.72.193.11"
    # cloud_host: str = "172.16.6.101"

    # algo_host: str = "100.72.193.11"

    # 本地测试
    device_host: str = "100.72.118.57"
    edge_host: str = "127.0.0.1"
    cloud_host: str = "172.16.6.101"
    algo_host: str = "127.0.0.1"

    edge_feature_port: int = 9001
    edge_status_port: int = 9002
    edge_iperf_port: int = 5001

    cloud_feature_port: int = 32266
    cloud_status_port: int = 32265
    cloud_iperf_port: int = 32264

    algo_server_port: int = 8000

    default_bw_d2e: float = 10.0
    default_bw_e2c: float = 50.0
    num_users: int = 10

    @property
    def algo_decision_url(self) -> str:
        return f"http://{self.algo_host}:{self.algo_server_port}/api/v1/decision"

    @property
    def algo_base_url(self) -> str:
        return f"http://{self.algo_host}:{self.algo_server_port}"


DEFAULT = TestbedConfig()

# Backward-compatible module constants for older scripts/imports.
DEVICE_HOST = DEFAULT.device_host
EDGE_HOST = DEFAULT.edge_host
CLOUD_HOST = DEFAULT.cloud_host
ALGO_HOST = DEFAULT.algo_host
LISTEN_HOST = DEFAULT.listen_host

EDGE_PORT = DEFAULT.edge_feature_port
CLOUD_PORT = DEFAULT.cloud_feature_port
ALGO_PORT = DEFAULT.algo_server_port
IPERF_PORT = DEFAULT.edge_iperf_port

EDGE_FEATURE_PORT = DEFAULT.edge_feature_port
EDGE_STATUS_PORT = DEFAULT.edge_status_port
EDGE_IPERF_PORT = DEFAULT.edge_iperf_port
CLOUD_FEATURE_PORT = DEFAULT.cloud_feature_port
CLOUD_STATUS_PORT = DEFAULT.cloud_status_port
CLOUD_IPERF_PORT = DEFAULT.cloud_iperf_port
