"""
Scripts/Exp0_Motivation/exp1_decoupling_failure/run_exp1.py
实验1：解耦失效定理 — 主运行脚本。
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

from Scripts.Exp0_Motivation.exp1_decoupling_failure.model_wrapper import (
    ResNet50EEWrapper,
)
from Scripts.Exp0_Motivation.exp1_decoupling_failure.simulator import simulate_latency
from Scripts.Exp0_Motivation.exp1_decoupling_failure.strategies import (
    DSCIJointStrategy,
    DecoupledStrategy,
    EEOnlyStrategy,
    LocalStrategy,
    SCOnlyStrategy,
)
from Scripts.Exp0_Motivation.utils.output_paths import create_run_output_dirs

# ---- 实验超参 ----
BANDWIDTHS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # Mbps，含极窄带以触发性能倒置
THRESHOLD_GROUPS = {
    "easy": [0.85, 0.88, 0.90, 0.92],
    "hard": [0.55, 0.60, 0.65, 0.70],
}


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


def _format_split(split: int, final_layer: int) -> str:
    if split <= 0:
        return "-"
    if split >= final_layer:
        return "末层"
    return f"L{split}"


def _primary_tau(thresholds: list[float]) -> float:
    return float(thresholds[0]) if thresholds else 1.0


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _, output_dir = create_run_output_dirs(Path(PROJECT_ROOT))
    log_path = output_dir / "Logs" / "exp1.log"
    _setup_logging(log_path)
    log = logging.getLogger(__name__)

    wrapper = ResNet50EEWrapper(project_root=Path(PROJECT_ROOT))
    model_info = wrapper.get_model_info()
    final_layer = model_info["final_layer"]

    if model_info["rates_from_csv"]:
        log.info("[Exp1] 早退率来自 Data/OfflineTables/Resnet50_rates.csv（阈值线性插值）")
    else:
        log.info("[Exp1] 早退率由 Beta 分布合成（CSV 不可用）")

    strategies = {
        "Local": LocalStrategy(),
        "EE-Only": EEOnlyStrategy(),
        "SC-Only": SCOnlyStrategy(),
        "Decoupled": DecoupledStrategy(),
        "DSCI": DSCIJointStrategy(),
    }

    results: list[dict] = []
    inversions: list[str] = []

    for group, tau_grid in THRESHOLD_GROUPS.items():
        for bw in BANDWIDTHS:
            log.info("[Exp1] Group=%s | BW=%.1fMbps", group, bw)
            group_latencies: dict[str, float] = {}

            for name, strategy in strategies.items():
                split, thresholds, lat, tx = strategy.optimize(
                    bw, wrapper, model_info, tau_grid=tau_grid
                )
                tau = _primary_tau(thresholds)
                rates = wrapper.get_exit_rates_from_csv(tau)
                lat_check, tx_check = simulate_latency(
                    {"split_layer": split, "bandwidth_mbps": bw},
                    rates,
                    model_info,
                )

                record = {
                    "group": group,
                    "bandwidth_mbps": bw,
                    "strategy": name,
                    "avg_latency_ms": round(lat_check, 2),
                    "split_layer_chosen": split,
                    "exit_threshold_used": round(tau, 4),
                    "actual_transmission_ratio": round(tx_check, 4),
                    "tau_grid": tau_grid,
                }
                results.append(record)
                group_latencies[name] = lat_check

                extra = ""
                if name in ("EE-Only", "Decoupled", "DSCI"):
                    extra = f" | τ={tau:.2f}"
                log.info(
                    "  %-12s: latency=%6.1fms | split=%s | tx_ratio=%.2f%s",
                    name,
                    lat_check,
                    _format_split(split, final_layer),
                    tx_check,
                    extra,
                )

            if (
                group_latencies.get("Decoupled", 0)
                > group_latencies.get("Local", float("inf"))
            ):
                msg = (
                    f">>> [!] Decoupled > Local at BW={bw}Mbps "
                    f"(Performance Inversion confirmed)"
                )
                log.info(msg)
                inversions.append(f"{group}@{bw}Mbps")

    out_json = output_dir / "Data" / "exp1_results.json"
    payload = {
        "model_info": {
            "exit_layer_indices": model_info["exit_layer_indices"],
            "final_layer": model_info["final_layer"],
            "total_flops": model_info["total_flops"],
            "rates_from_csv": model_info["rates_from_csv"],
        },
        "threshold_groups": THRESHOLD_GROUPS,
        "bandwidths": BANDWIDTHS,
        "inversions": inversions,
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("[Exp1] 结果已保存: %s", out_json)


if __name__ == "__main__":
    main()
