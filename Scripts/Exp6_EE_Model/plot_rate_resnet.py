"""
Src/Experiments/Exp5_EE_Model/plot_rate_resnet.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import RESULT_EE_MODEL_PATH, COLORS


def plot_early_exit_probability(data_dir: Path, save_dir: Path = Path(RESULT_EE_MODEL_PATH)):
    """
    绘制 Early Exit 概率随阈值变化的曲线图。
    """
    # 数据
    df = pd.read_csv(data_dir)
    if not data_dir.exists():
        print(f"错误: 找不到文件 {data_dir}")
        return

    # 绘图
    set_ieee_style(mode='single')
    plt.figure()
    plt.plot(df['threshold'], df['exit1_rate'],
             color=COLORS["blue"], marker='o', markevery=3,
             label='Early Exit 1')

    plt.plot(df['threshold'], df['exit2_rate'],
             color=COLORS["green"], marker='^', markevery=3,
             label='Early Exit 2')
    plt.xlabel('Threshold')
    plt.ylabel('Early Exit Probability (%)')
    plt.xlim(0, 1)
    plt.ylim(0, 105)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"{data_dir.stem}_exit_probability")
    plt.show()


if __name__ == "__main__":
    csv_file = Path(r"D:\Coding\Python\DSCI\Data\Resnet50_rates.csv")
    save_dir = Path(RESULT_EE_MODEL_PATH)
    plot_early_exit_probability(csv_file, save_dir)