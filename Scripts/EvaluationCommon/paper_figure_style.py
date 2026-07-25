"""Shared style and paths for the four compact Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "Scripts" / "Results"

SEAM_COLOR = "#0072B2"
ISPLITEE_COLOR = "#D55E00"
RANDOM_COLOR = "#009E73"
GA_COLOR = "#CC79A7"
NEUTRAL_COLOR = "#6B6B6B"

SEAM_MARKER = "o"
ISPLITEE_MARKER = "s"

COMPARISON_FIGSIZE = (3.55, 1.95)
COMPARISON_SUBPLOTS = {
    "left": 0.115,
    "right": 0.99,
    "top": 0.82,
    "bottom": 0.25,
}


def apply_compact_ieee_style() -> None:
    """Configure Matplotlib for a final-width 0.24-textwidth vector panel."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 7.0,
            "font.weight": "bold",
            "axes.labelsize": 7.6,
            "axes.labelweight": "bold",
            "axes.titlesize": 7.6,
            "axes.titleweight": "bold",
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.85,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "lines.linewidth": 1.65,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.9,
            "hatch.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def apply_comparison_figure_style() -> None:
    """Apply the larger, bold typography shared by the comparison figures."""
    apply_compact_ieee_style()
    mpl.rcParams.update(
        {
            "font.size": 8.0,
            "font.weight": "bold",
            "axes.labelsize": 8.6,
            "axes.labelweight": "bold",
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def add_comparison_legend(
    fig: plt.Figure,
    handles: list,
    labels: list[str],
    *,
    ncol: int | None = None,
) -> plt.Legend:
    """Place a compact, consistently styled legend above the plot area."""
    return fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.83),
        ncol=ncol or len(labels),
        borderaxespad=0.0,
        borderpad=0.12,
        labelspacing=0.15,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.65,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        prop={"size": 6.8, "weight": "bold"},
    )


def bold_tick_labels(*axes: plt.Axes) -> None:
    """Keep tick-label weight consistent with the axes labels."""
    for ax in axes:
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")


def save_pdf(
    fig: plt.Figure,
    output_path: Path,
    *,
    fixed_canvas: bool = False,
) -> Path:
    """Save a vector PDF and close its Matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = (
        {"bbox_inches": fig.bbox_inches, "pad_inches": 0.0} if fixed_canvas else {}
    )
    fig.savefig(output_path, format="pdf", **save_kwargs)
    plt.close(fig)
    return output_path


def pending_panel(
    ax: plt.Axes, title: str | None = None, detail: str = "Experiment not run"
) -> None:
    """Render an explicit incomplete panel without inventing numeric values."""
    if title:
        ax.set_title(title, pad=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#A0A0A0")
        spine.set_linestyle("--")
        spine.set_linewidth(0.65)
    ax.text(
        0.5,
        0.56,
        "Data pending",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=NEUTRAL_COLOR,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.39,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=NEUTRAL_COLOR,
        wrap=True,
    )
