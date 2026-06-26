"""iperf3 bandwidth measurement helpers."""

from __future__ import annotations

import json
import os
import subprocess

IPERF_EXE = os.environ.get("DSCI_IPERF_EXE", "iperf3")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


DEFAULT_IPERF_DURATION_S = _env_float("DSCI_IPERF_DURATION_S", 8.0)
DEFAULT_IPERF_TIMEOUT_MARGIN_S = _env_float("DSCI_IPERF_TIMEOUT_MARGIN_S", 15.0)


def measure_bandwidth_iperf(
    server_ip: str,
    port: int = 5001,
    duration: float | None = None,
    timeout_s: float | None = None,
) -> float | None:
    """Measure upload bandwidth to an iperf3 server and return Mbps.

    Slow overlay networks such as Tailscale can spend the first couple of seconds
    ramping up, so the default duration is intentionally longer than a LAN probe.
    ``None`` is returned on failure so callers can apply their own fallback.
    """
    duration = float(DEFAULT_IPERF_DURATION_S if duration is None else duration)
    timeout_s = float(
        duration + DEFAULT_IPERF_TIMEOUT_MARGIN_S
        if timeout_s is None
        else timeout_s
    )
    cmd = [
        IPERF_EXE,
        "-c",
        str(server_ip),
        "-p",
        str(int(port)),
        "-t",
        str(duration),
        "-J",
    ]
    try:
        print(
            f"iperf3 measuring {server_ip}:{int(port)}, "
            f"duration={duration:.1f}s, timeout={timeout_s:.1f}s"
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0 or not result.stdout:
            print("iperf3 measurement failed, return code:", result.returncode)
            if result.stderr:
                print("iperf3 stderr:", result.stderr)
            return None

        data = json.loads(result.stdout)
        bw_bps = float(data["end"]["sum_sent"]["bits_per_second"])
        bw_mbps = bw_bps / 1e6
        if bw_mbps <= 0:
            print("iperf3 measured non-positive bandwidth")
            return None
        return bw_mbps
    except subprocess.TimeoutExpired:
        print(
            f"iperf3 measurement timed out: target={server_ip}:{int(port)}, "
            f"duration={duration:.1f}s, timeout={timeout_s:.1f}s"
        )
        return None
    except FileNotFoundError:
        print(f"iperf3 executable not found: {IPERF_EXE}")
        return None
    except Exception as exc:
        print(f"iperf3 measurement error: {exc}")
        return None
