"""Prepare Figure 3 data for online decision granularity overhead."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.Exp0_Motivation.config import (  # noqa: E402
    DEFAULT_CONFIG,
    data_dir,
    prepare_run_dir,
    save_config,
    update_paper_numbers,
)


def _bytes_per_ms(bytes_count: float, bandwidth_mbps: float) -> float:
    bytes_per_second = bandwidth_mbps * 1e6 / 8.0
    return bytes_count / bytes_per_second * 1000.0


def _per_request_observed_overhead_ms(n_users: int) -> float:
    cfg = DEFAULT_CONFIG
    state_bytes = cfg.exp3_state_vector_dim * 4
    state_collection = _bytes_per_ms(state_bytes, cfg.exp3_control_bandwidth_mbps) + cfg.exp3_rtt_ms / 2.0
    config_dispatch = _bytes_per_ms(cfg.exp3_config_bytes, cfg.exp3_control_bandwidth_mbps) + cfg.exp3_rtt_ms / 2.0
    scheduler_queue = n_users * cfg.exp3_decision_latency_ms
    return state_collection + scheduler_queue + config_dispatch


def _slow_timeslot_overhead_per_request_ms(n_users: int) -> float:
    cfg = DEFAULT_CONFIG
    state_bytes = cfg.exp3_state_vector_dim * 4
    state_refresh = (
        _bytes_per_ms(n_users * state_bytes, cfg.exp3_control_bandwidth_mbps)
        + cfg.exp3_rtt_ms / 2.0
    )
    config_broadcast = (
        _bytes_per_ms(n_users * cfg.exp3_config_bytes, cfg.exp3_control_bandwidth_mbps)
        + cfg.exp3_rtt_ms / 2.0
    )
    period_overhead = (
        cfg.exp3_slow_optimization_latency_ms
        + state_refresh
        + config_broadcast
    )
    requests_per_period = n_users * (
        cfg.exp3_schedule_period_s * 1000.0 / cfg.exp3_inference_latency_ms
    )
    return cfg.exp3_slow_fastpath_latency_ms + period_overhead / max(
        requests_per_period, 1.0
    )


def _throughput_rps(n_users: int, overhead_ms: float) -> float:
    cfg = DEFAULT_CONFIG
    return n_users * 1000.0 / (cfg.exp3_inference_latency_ms + overhead_ms)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    run_dir = prepare_run_dir(args.run_id, prefer_latest=True)
    save_config(run_dir, cfg)

    rows = []
    for n_users in cfg.exp3_users:
        per_request_overhead = _per_request_observed_overhead_ms(n_users)
        slow_overhead = _slow_timeslot_overhead_per_request_ms(n_users)
        rows.extend(
            [
                {
                    "n_users": int(n_users),
                    "scheduler": "Per-request joint decision",
                    "scheduling_overhead_ms_per_request": per_request_overhead,
                    "effective_throughput_rps": _throughput_rps(n_users, per_request_overhead),
                },
                {
                    "n_users": int(n_users),
                    "scheduler": "Slow-timeslot joint decision",
                    "scheduling_overhead_ms_per_request": slow_overhead,
                    "effective_throughput_rps": _throughput_rps(n_users, slow_overhead),
                },
            ]
        )

    results = pd.DataFrame(rows)
    per_request = results[results["scheduler"] == "Per-request joint decision"][
        "scheduling_overhead_ms_per_request"
    ].tolist()
    slow = results[results["scheduler"] == "Slow-timeslot joint decision"][
        "scheduling_overhead_ms_per_request"
    ].tolist()
    if any(curr < prev - 1e-9 for prev, curr in zip(per_request, per_request[1:])):
        raise AssertionError("Per-request overhead must be non-decreasing with users")
    if any(curr > prev + 1e-9 for prev, curr in zip(slow, slow[1:])):
        raise AssertionError("Slow-timeslot amortized overhead should not increase with users")

    out_path = data_dir(run_dir) / "exp3_decision_overhead.csv"
    results.to_csv(out_path, index=False)
    update_paper_numbers(
        run_dir,
        "figure3",
        {
            "output_csv": str(out_path),
            "max_per_request_overhead_ms": float(max(per_request)),
            "min_slow_timeslot_overhead_ms": float(min(slow)),
            "max_users": int(max(cfg.exp3_users)),
            "per_request_overhead_growth_x": float(max(per_request) / min(per_request)),
        },
    )
    print(f"Figure 3 data: {out_path}")


if __name__ == "__main__":
    main()
