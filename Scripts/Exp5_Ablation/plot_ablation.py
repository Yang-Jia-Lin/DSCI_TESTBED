"""
Src/Experiments/Exp4_Ablation/plot_ablation.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Src.Algorithm.Utils.plot_utils import save_fig_for_ieee, set_ieee_style
from Src.paras import COLORS, RESULT_ABLATION_PATH


def _to_float(x):
    """Safely convert pandas/numpy scalars (including complex) to float."""
    try:
        return float(x)
    except Exception:
        try:
            # handle numpy scalars
            return float(np.asarray(x).item())
        except Exception:
            # fallback: real part
            return float(np.real(x))


def plot_bubble_chart(data: pd.DataFrame, save_dir=Path(RESULT_ABLATION_PATH)):
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    SEMANTIC_COLORS = {
        "Device": COLORS["grey"],
        "Edge": COLORS["brown"],
        "Cloud": COLORS["green"],
        "Co-only": COLORS["purple"],
        "Edge+EE": COLORS["blue"],  # 保持你代码中的 bule 拼写
        "Ours": COLORS["red"],
    }
    name_map = {
        "仅在终端": "Device",
        "仅在边端": "Edge",
        "仅在云端": "Cloud",
        "仅协同": "Co-only",
        "边端 + 早退": "Edge+EE",
        "协同 + 早退": "Ours",
    }
    data["display_name"] = data["name"].map(
        lambda x: name_map.get(str(x), str(x)) if pd.notna(x) else x
    )
    # 计算气泡大小
    obj_min, obj_max = data["objective"].min(), data["objective"].max()
    if obj_max != obj_min:
        # 将基础大小从 150 提至 200，映射极差提至 1500，气泡会更饱满
        data["bubble_size"] = (
            200 + (data["objective"] - obj_min) / (obj_max - obj_min + 1e-6) * 1500
        )
    else:
        data["bubble_size"] = 1000
    df_sorted = data.sort_values("latency_ms").reset_index(drop=True)
    for i, row in enumerate(df_sorted.itertuples(index=False)):
        label = str(row.display_name).replace("+", "\n+")
        ax.scatter(
            _to_float(row.latency_ms),
            row.accuracy,
            s=row.bubble_size,
            color=SEMANTIC_COLORS[row.display_name],
            alpha=0.9,
            edgecolors="w",
            zorder=10 + i,
        )
        # 标签偏移逻辑
        # y_offset = 14 if i % 2 == 0 else -14
        # v_align = 'bottom' if i % 2 == 0 else 'top'
        txt = ax.annotate(
            label,
            xy=(row.latency_ms, row.accuracy),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            zorder=20 + i,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="w")])
    ax.set_xlabel("Total Inference Latency (ms)")
    ax.set_ylabel("Total Accuracy")
    ax.margins(x=0.1, y=0.25)
    # ax.grid()
    plt.tight_layout(pad=0.2)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir / f"baseline_bubble_chart_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


def plot_utility_bar(data: pd.DataFrame, save_dir=Path(RESULT_ABLATION_PATH)):
    set_ieee_style(mode="single")
    fig, ax2 = plt.subplots()
    SEMANTIC_COLORS = {
        "Device": COLORS["grey"],
        "Edge": COLORS["brown"],
        "Cloud": COLORS["green"],
        "Co-only": COLORS["purple"],
        "Edge+EE": COLORS["blue"],  # 保持你代码中的 bule 拼写
        "Ours": COLORS["red"],
    }
    name_map = {
        "仅在终端": "Device",
        "仅在边端": "Edge",
        "仅在云端": "Cloud",
        "仅协同": "Co-only",
        "边端 + 早退": "Edge+EE",
        "协同 + 早退": "Ours",
    }
    data["display_name"] = data["name"].map(
        lambda x: name_map.get(str(x), str(x)) if pd.notna(x) else x
    )
    colors = [SEMANTIC_COLORS[name] for name in data["display_name"]]
    bars = ax2.bar(
        data["display_name"], data["objective"], color=colors[: len(data)], width=0.7
    )
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h > 0 else "top"
        offset = (
            0.02 * data["objective"].max()
            if h > 0
            else -0.05 * abs(data["objective"].min())
        )
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            f"{h:.2f}",
            ha="center",
            va=va,
        )
    ax2.set_ylabel("Total Utility")
    y_min = min(0, data["objective"].min() * 1.2)
    y_max = max(0, data["objective"].max() * 1.2)
    ax2.set_ylim(y_min, y_max)
    plt.xticks(rotation=15)
    plt.tight_layout(pad=0.2)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir / f"baseline_bar_chart_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


if __name__ == "__main__":
    mock_data = pd.DataFrame(
        {
            "name": [
                "仅在终端",
                "仅在边端",
                "仅在云端",
                "仅协同",
                "边端 + 早退",
                "协同 + 早退",
            ],
            "latency_ms": [30, 80, 250, 150, 60, 110],
            "accuracy": [75.5, 88.2, 98.5, 92.0, 85.0, 96.5],
            "objective": [0.45, 0.62, 0.35, 0.78, 0.72, 0.95],
        }
    )

    from Src.paras import RESULT_TEST_PATH

    plot_bubble_chart(mock_data, save_dir=Path(RESULT_TEST_PATH))
    plot_utility_bar(mock_data, save_dir=Path(RESULT_TEST_PATH))
