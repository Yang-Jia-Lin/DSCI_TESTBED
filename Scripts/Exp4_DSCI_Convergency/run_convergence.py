"""
Src/Experiments/Exp3_DSCI_Convergency/run_convergence.py
"""
import pandas as pd
import json
from pathlib import Path

from Scripts.Exp3_DSCI_Convergency.plot_convergency import plot_convergence, plot_entropy, plot_lan_and_acc
from Src.paras import RESULT_CONVERGENCE_PATH


def run_convergence_analysis(data_dir: Path, output_dir: Path = Path(RESULT_CONVERGENCE_PATH)):
    # 数据
    metrics_path = data_dir / "metrics.jsonl"
    data = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    utility = df['outer_obj']
    entropy_X = df['entropy_X']
    entropy_Y = df['entropy_Y']
    latency = df['latency']
    acc = df['acc']

    # 路径
    fig_save_dir = Path(output_dir) / data_dir.name

    # 1. 总效用收敛曲线 (outer_obj)
    plot_convergence(utility, fig_save_dir)
    # 2. 熵收敛曲线 (Entropy X & Y)
    plot_entropy(entropy_X, entropy_Y, fig_save_dir)
    # 3. 性能指标变化 (Latency & Accuracy)
    plot_lan_and_acc(latency, acc, fig_save_dir)


if __name__ == "__main__":
    data_path = Path(r"D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260128_232731")
    result_path = Path(RESULT_CONVERGENCE_PATH)
    run_convergence_analysis(data_path, result_path)