"""
Scripts/Exp0_Motivation/exp2_scalability/plot_exp2.py
实验2：配置漂移 + 有效吞吐量/调度开销占比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style  # noqa: E402

from Scripts.Exp0_Motivation.utils.output_paths import resolve_output_dir


def _load(output_dir: Path) -> dict:
    with open(output_dir / "data" / "exp2_results.json", encoding="utf-8") as f:
        return json.load(f)


def _get_series(results: list[dict], scheduler: str, field: str):
    rows = sorted(
        [r for r in results if r["scheduler"] == scheduler],
        key=lambda x: x["n_users"],
    )
    n = [r["n_users"] for r in rows]
    y = [r[field] for r in rows]
    return np.array(n), np.array(y)


def plot_drift(payload: dict, output_dir: Path) -> None:
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    results = payload["results"]
    n_pr, drift_pr = _get_series(results, "Per-Request", "config_drift")
    n_ds, drift_ds = _get_series(results, "DSCI", "config_drift")

    ax.plot(n_pr, drift_pr, color="#F44336", marker="o", label="Per-request")
    ax.plot(n_ds, drift_ds, color="#4CAF50", marker="s", label="DSCI")
    ax.fill_between(n_pr, drift_pr, drift_ds, color="#F44336", alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_pr)
    ax.set_xticklabels([str(int(x)) for x in n_pr])
    ax.set_xlabel("Concurrent Users (N)")
    ax.set_ylabel("Normalized Config Drift")
    ymax = max(float(drift_pr.max()) * 1.15, 0.05)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.text(
        0.97,
        0.05,
        "O(N) drift",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#F44336",
        fontsize=9,
    )
    ax.text(
        0.97,
        0.18,
        "O(1) stable",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#4CAF50",
        fontsize=9,
    )
    ax.set_title("Configuration Drift vs. Concurrency")
    plt.tight_layout(pad=0.15)
    save_fig_for_ieee(output_dir / "figures" / "exp2_drift", fig=fig)
    plt.close(fig)


def plot_throughput_overhead(payload: dict, output_dir: Path) -> None:
    """
    左图：扣除调度开销后的有效吞吐 (req/s)。
    右图：调度耗时占「推理+调度」的比例 (%)，体现 Per-request 随 N 恶化。
    """
    set_ieee_style(mode="double")
    fig, axes = plt.subplots(1, 2)
    results = payload["results"]

    n_pr, t_pr = _get_series(results, "Per-Request", "effective_throughput_rps")
    n_ds, t_ds = _get_series(results, "DSCI", "effective_throughput_rps")

    axes[0].plot(n_pr, t_pr, color="#F44336", marker="o", label="Per-request")
    axes[0].plot(n_ds, t_ds, color="#4CAF50", marker="s", label="DSCI")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(n_pr)
    axes[0].set_xticklabels([str(int(x)) for x in n_pr])
    axes[0].set_xlabel("Concurrent Users (N)")
    axes[0].set_ylabel("Effective Throughput (req/s)")
    axes[0].set_title("(a) Throughput After Scheduling Cost")
    axes[0].legend(loc="upper left", fontsize=9)
    if t_pr[-1] > 0:
        speedup = t_ds[-1] / t_pr[-1]
        axes[0].annotate(
            f"{speedup:.2f}x @ N={int(n_ds[-1])}",
            xy=(n_ds[-1], t_ds[-1]),
            xytext=(n_ds[-3], t_pr[-1] * 1.02),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
            fontsize=9,
        )

    n_pr, o_pr = _get_series(
        results, "Per-Request", "scheduling_overhead_ratio_pct"
    )
    n_ds, o_ds = _get_series(results, "DSCI", "scheduling_overhead_ratio_pct")
    x = np.arange(len(n_pr))
    w = 0.35
    axes[1].bar(x - w / 2, o_pr, width=w, color="#F44336", label="Per-request")
    axes[1].bar(x + w / 2, o_ds, width=w, color="#4CAF50", label="DSCI")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(v)) for v in n_pr])
    axes[1].set_xlabel("Concurrent Users (N)")
    axes[1].set_ylabel("Scheduling Overhead (%)")
    axes[1].set_title("(b) Scheduling Cost Share per Request")
    ymax = max(float(o_pr.max()), float(o_ds.max()), 5.0)
    axes[1].set_ylim(0, min(35.0, ymax * 1.12 + 2))
    axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Scalability: Per-Request Control vs. DSCI Quasi-Static Broadcast",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(pad=0.15)
    save_fig_for_ieee(output_dir / "figures" / "exp2_throughput_overhead", fig=fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = resolve_output_dir(Path(PROJECT_ROOT), args.timestamp)
    payload = _load(output_dir)
    plot_drift(payload, output_dir)
    plot_throughput_overhead(payload, output_dir)
    print(f"[plot_exp2] 图表已写入 {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
