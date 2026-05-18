"""
Src/Experiments/Exp5_EE_Model/plot_train_convergence_resnet.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import RESULT_EE_MODEL_PATH, COLORS


def plot_training_convergence(data_dir: Path, save_dir: Path = Path(RESULT_EE_MODEL_PATH)):
    """
    读取训练日志并生成收敛曲线图。
    """
    # 数据
    df = pd.read_csv(data_dir)
    val_acc1 = df['val_acc'][:50].reset_index(drop=True)
    val_acc2 = df['val_acc'][50:100].reset_index(drop=True)
    val_acc3 = df['val_acc'][100:150].reset_index(drop=True)
    epochs = list(range(1, 51))

    # 绘图
    set_ieee_style(mode='single')
    plt.figure()
    plt.plot(epochs, val_acc1, color=COLORS["red"], marker='o', markevery=3, label='Main Exit')
    plt.plot(epochs, val_acc2, color=COLORS["blue"], marker='s', markevery=3, label='Early Exit 1')
    plt.plot(epochs, val_acc3, color=COLORS["green"], marker='^', markevery=3, label='Early Exit 2')
    plt.xlabel('Training Epochs')
    plt.ylabel('Classification Accuracy (%)')
    plt.xlim(0, 50)
    plt.ylim(45, 90)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"{data_dir.stem}_convergence")
    plt.show()


if __name__ == '__main__':
    csv_file = Path(r"D:\Coding\Python\DSCI\Data\ResNet50_trainlog_0508_0137.csv")
    save_dir = Path(RESULT_EE_MODEL_PATH)
    plot_training_convergence(csv_file, save_dir)