"""
Src/Utils/log_function.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Src.Phase2_Scheduler.paras import Paras
from Src.Phase2_Scheduler.Reporting.plot_convergence import plot_convergence
from Src.Phase2_Scheduler.Reporting.plot_decision import plot_X, plot_Y
from Src.Shared.Utils.utils_function import NumpyEncoder, open_file


def save_experiment_results(
    save_dir: Path,
    algo_name: str,
    paras,
    best_val: float,
    best_sol: tuple,
    history: list,
    hyper_params: dict | None = None,
    extra_logs: list | None = None,
):
    # ======== 1) 创建文件夹 ========
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = save_dir / f"{algo_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    X_opt, Y_opt, F_e_opt, F_c_opt = best_sol

    # ======== 2) 保存参数到 config.json 和 metrics.jsonl ========
    paras_dict = {k: v for k, v in vars(paras).items() if not k.startswith("__")}
    config_data = {
        "Algorithm": algo_name,
        "Time": timestamp,
        "Best_Objective_Value": best_val,
        "Hyper_Parameters": hyper_params,
        "System_Parameters": paras_dict,
    }
    if extra_logs is not None:
        config_data["Extra_Logs_Keys"] = (
            sorted(list(extra_logs[0].keys())) if len(extra_logs) > 0 else []
        )
    json_path = exp_dir / "config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, cls=NumpyEncoder)
    if extra_logs is not None:
        metrics_path = exp_dir / "metrics.jsonl"
        with open(metrics_path, "w", encoding="utf-8") as f:
            for row in extra_logs:
                f.write(json.dumps(row, cls=NumpyEncoder) + "\n")

    # ======== 3) 保存结果数据到 solution.npz
    npz_path = exp_dir / "solution.npz"
    extra_npz = {}
    if extra_logs is not None and len(extra_logs) > 0:
        keys = extra_logs[0].keys()
        for k in keys:
            extra_npz[f"metrics_{k}"] = np.array(
                [row.get(k) for row in extra_logs], dtype=object
            )
    np.savez(
        npz_path,
        # 核心解
        X=X_opt,
        Y=Y_opt,
        F_e=F_e_opt,
        F_c=F_c_opt,
        # 标量和历史
        best_val=best_val,
        history=np.array(history),
        **extra_npz,
    )

    # ======== 4) 绘制并保存曲线 ========
    plot_convergence(history, alg_name=f"{algo_name}_Convergence", save_dir=exp_dir)
    plot_X(X_opt, paras.E, save_dir=exp_dir)
    plot_Y(Y_opt, paras.E, save_dir=exp_dir)


def load_and_analyze_results(exp_dir: Path, analysis=True):
    """
    加载并复现实验结果
    """
    exp_dir = Path(exp_dir)
    json_path = exp_dir / "config.json"
    npz_path = exp_dir / "solution.npz"

    if not json_path.exists() or not npz_path.exists():
        print(f"Error: 路径 {exp_dir} 下缺少 config.json 或 solution.npz")

    # ======== 1) 加载数据 ========
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    data = np.load(npz_path, allow_pickle=True)

    # ======== 2) 提取信息 ========
    algo_name = config.get("Algorithm", "Unknown")
    timestamp = config.get("Time", "Unknown")
    hyper_params = config.get("Hyper_Parameters", {})
    sys_params = config.get("System_Parameters", {})
    raise ValueError(
        "Loading legacy experiment configs is not supported; rerun the experiment with bundle_id"
    )
    best_val = config.get("Best_Objective_Value", "N/A")
    X_opt = data["X"]
    Y_opt = data["Y"]
    F_e, F_c = data["F_e"], data["F_c"]
    history = data["history"]

    if analysis:
        # ======== 3) 打印结果 ========
        print("=" * 50)
        print(f"实验结果: {algo_name} | {timestamp}")
        print("=" * 50)

        # 3.1 超参数
        print("\n核心超参数 (Hyper-Parameters):")
        for k, v in hyper_params.items():
            print(f"  - {k}: {v}")

        # 3.2 系统参数
        print("\n关键系统参数 (System Parameters):")
        target_sys_keys = ["F_u", "f_e_max", "f_c_max", "alpha", "beta"]
        for key in target_sys_keys:
            val = sys_params.get(key, "Not Found")
            # 如果是数组，打印其形状或均值，避免刷屏
            if isinstance(val, list) and len(val) > 5:
                print(f"  - {key}: List of length {len(val)} ({val[:3]}...)")
            else:
                print(f"  - {key}: {val}")

        # 3.3 结果
        print("\n训练结果 (Results):")
        print(f"  - Best Objective Value: {best_val}")
        print(f"  - Solution Matrices in NPZ: {list(data.keys())}")

        # ======== 4) 重新绘图 ========
        print("\n生成图表...")
        conv_svgs = list(exp_dir.glob("*Convergence*"))
        decs_svgs = list(exp_dir.glob("*Decisions*"))
        if conv_svgs:
            print(f"发现收敛图，正在打开: {conv_svgs[0].name}")
            open_file(conv_svgs[0])
        else:
            print("未发现收敛图，正在重新绘制...")
            plot_convergence(
                history, alg_name=f"{algo_name}_Convergence", save_dir=exp_dir
            )
        if decs_svgs:
            print(f"发现决策图，正在打开: {decs_svgs[0].name}")
            open_file(decs_svgs[0])
        else:
            print("未发现决策图，正在重新绘制...")
            plot_X(X_opt, paras.E, save_dir=exp_dir)
            plot_Y(Y_opt, paras.E, save_dir=exp_dir)

    # ======== 5) 返回结果
    return X_opt, Y_opt, F_e, F_c, history, paras


def save_thr_data(thr_data: pd.DataFrame, data_name: str, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}.csv"
    csv_path = save_dir / filename
    thr_data.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    from Src.Shared.Config.paths import RESULT_DIR

    target_path = RESULT_DIR / "Optimize" / "PPO" / "PPO_20260127_115650"
    load_and_analyze_results(target_path)
