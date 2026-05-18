"""
Src/Utils/plot_utils.py
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_ieee_style(mode="single"):
    """
    配置 IEEE 论文风格的绘图参数。
    :param mode: 'single' 代表单栏图片（3.5英寸宽），'double' 代表跨栏大图（7.0英寸宽）
    """
    # 1. 选择基础风格
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Liberation Serif",
        "DejaVu Serif",
        "serif",
    ]

    # 2. 计算画布尺寸 (单位为英寸)
    # IEEE 单栏建议宽度为 3.5 英寸，高度可根据内容调整（推荐 2.8 - 3.2）
    if mode == "single":
        fig_width = 4.0
        fig_height = 2.8
    else:
        fig_width = 7.0
        fig_height = 4.5

    # 3. 核心参数字典
    ieee_params = {
        # 字体设置
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": False,  # 如果系统没装 LaTeX 环境，设为 False
        # 字体大小
        "axes.titlesize": 14,  # 标题大小
        "axes.labelsize": 12,  # 坐标轴标签大小
        "font.size": 12,  # 全局字体
        "legend.fontsize": 11,  # 图例大小
        "xtick.labelsize": 11,  # 刻度大小
        "ytick.labelsize": 11,
        # 线条与点
        "lines.linewidth": 2,  # 稍微加粗线条
        "lines.markersize": 4,  # 标记点大小
        # 嵌入字体
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # 网格与布局
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (fig_width, fig_height),
        "figure.dpi": 300,  # 高分辨率
        "figure.autolayout": True,
    }

    # 4. 更新全局配置
    mpl.rcParams.update(ieee_params)
    print(f"IEEE Style ({mode}) initialized.")


def save_fig_for_ieee(save_path, fig=None):
    """
    如果传了 fig，就存 fig；没传，就存当前活跃的图 (plt)
    """
    target = fig if fig is not None else plt
    target.savefig(
        save_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0
    )
    target.savefig(
        save_path.with_suffix(".png"),
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    print(f"图已保存至: {save_path}")
