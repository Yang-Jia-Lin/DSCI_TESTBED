"""
Scripts/Exp0_Motivation/exp2_scalability/run_exp2.py
实验2：控制-推理耦合的可扩展性瓶颈。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Scripts.Exp0_Motivation.exp2_scalability.scheduler import (
    PerRequestScheduler,
    QuasiStaticScheduler,
)
from Scripts.Exp0_Motivation.utils.config import (
    DECISION_LATENCY_MS,
    OPTIMIZATION_LATENCY_MS,
    RTT_MS,
    SCHEDULE_PERIOD_S,
)
from Scripts.Exp0_Motivation.utils.output_paths import create_run_output_dirs

CONCURRENT_USERS = [1, 2, 4, 8, 16, 32]
INFERENCE_LATENCY_MS = 50.0
BANDWIDTH_MBPS = 4.0
RTT_MS_VAL = RTT_MS
STATE_VECTOR_DIM = 20
CONFIG_BYTES = 16
SIMULATION_DURATION_S = 60.0
SCHEDULE_PERIOD_S_VAL = SCHEDULE_PERIOD_S


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _, output_dir = create_run_output_dirs(Path(PROJECT_ROOT))
    log = logging.getLogger(__name__)
    _setup_logging(output_dir / "Logs" / "exp2.log")

    per_req = PerRequestScheduler(
        state_vector_dim=STATE_VECTOR_DIM,
        config_bytes=CONFIG_BYTES,
        decision_latency_ms=DECISION_LATENCY_MS,
        bandwidth_mbps=BANDWIDTH_MBPS,
        rtt_ms=RTT_MS_VAL,
    )
    quasi = QuasiStaticScheduler(
        optimization_latency_ms=OPTIMIZATION_LATENCY_MS,
        config_bytes=CONFIG_BYTES,
        schedule_period_s=SCHEDULE_PERIOD_S_VAL,
        bandwidth_mbps=BANDWIDTH_MBPS,
        rtt_ms=RTT_MS_VAL,
    )

    results: list[dict] = []

    for n in CONCURRENT_USERS:
        log.info("[Exp2] N=%d concurrent users", n)

        for label, sched, is_quasi in (
            ("Per-Request", per_req, False),
            ("DSCI", quasi, True),
        ):
            if is_quasi:
                overhead_per_req = sched.overhead_per_request_ms(
                    n, INFERENCE_LATENCY_MS
                )
                overhead_round = sched.compute_scheduling_overhead_ms(n)
            else:
                overhead_round = sched.compute_scheduling_overhead_ms(n)
                overhead_per_req = overhead_round / n

            throughput = sched.compute_effective_throughput(
                n, INFERENCE_LATENCY_MS, SIMULATION_DURATION_S
            )
            drift = sched.compute_config_drift(n)
            denom = INFERENCE_LATENCY_MS + overhead_per_req
            overhead_ratio = (
                100.0 * overhead_per_req / denom if denom > 0 else 0.0
            )

            results.append(
                {
                    "n_users": n,
                    "scheduler": label,
                    "scheduling_overhead_ms_round": round(overhead_round, 3),
                    "scheduling_overhead_per_request_ms": round(
                        overhead_per_req, 4
                    ),
                    "scheduling_overhead_ratio_pct": round(overhead_ratio, 2),
                    "effective_throughput_rps": round(throughput, 2),
                    "config_drift": round(drift, 4),
                }
            )

            log.info(
                "  %-12s: overhead=%5.1f%% | throughput=%5.1f rps | drift=%.3f",
                label,
                overhead_ratio,
                throughput,
                drift,
            )

        pr = next(r for r in results if r["n_users"] == n and r["scheduler"] == "Per-Request")
        ds = next(r for r in results if r["n_users"] == n and r["scheduler"] == "DSCI")
        if pr["effective_throughput_rps"] > 0:
            speedup = ds["effective_throughput_rps"] / pr["effective_throughput_rps"]
            log.info(
                "  >>> DSCI throughput advantage: %.2fx at N=%d",
                speedup,
                n,
            )

    out_path = output_dir / "Data" / "exp2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "concurrent_users": CONCURRENT_USERS,
                    "inference_latency_ms": INFERENCE_LATENCY_MS,
                    "bandwidth_mbps": BANDWIDTH_MBPS,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    log.info("[Exp2] 结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
