"""
Src/Experiments/Exp2_Dynamic/run_resource_dynamic.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Scripts.Exp3_Dynamic.plot_resource_trend import plot_resource_trend
from Src.Algorithm.Optimizer.DSCI.run_DSCI import run_dsci_experiment
from Src.Algorithm.Utils.parsing_data import split_points_matrix
from Src.Algorithm.Utils.utils_function import NumpyEncoder
from Src.paras import RESULT_DYNAMIC_PATH


def start_resource_experiment(sweep_var: str, sweep_range: list, static_params):
    """
    函数一：初始化并运行实验。
    :param sweep_var: 自变量名称 (例如 'H_u', 'F_u', 'b_e', 'b_c')
    :param sweep_range: 自变量的变化范围
    :param static_params: Paras 实例或包含参数的字典
    """
    # 1. 创建目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_dir = Path(RESULT_DYNAMIC_PATH) / f"ResourceHetero_{sweep_var}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 2. 参数字典化
    if hasattr(static_params, "__dataclass_fields__") or not isinstance(
        static_params, dict
    ):
        # 处理 Paras 对象
        paras_dict = {
            k: v for k, v in vars(static_params).items() if not k.startswith("__")
        }
    else:
        paras_dict = static_params

    # 3. 保存实验配置到 config.json
    config = {
        "sweep_var": sweep_var,
        "sweep_range": list(sweep_range),
        "static_params": paras_dict,
        "created_at": timestamp,
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    print(f"New experiment started: {exp_dir}")
    print(f"Variables: Sweeping {sweep_var} over {sweep_range}")

    _run_experiment_loop(exp_dir, config)


def resume_or_analyze_experiment(exp_dir: Path, only_plot: bool = False):
    """
    函数二：根据中间文件继续运行或直接展示结果
    """
    exp_dir = Path(exp_dir)
    if only_plot:
        csv_path = exp_dir / "res_dynamic_data.csv"
        plot_resource_trend(csv_path=csv_path, save_dir=exp_dir)
        return

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in {exp_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Resuming experiment from: {exp_dir}")
    _run_experiment_loop(exp_dir, config)


def _run_experiment_loop(exp_dir, config):
    """
    内部核心循环，负责断点检测和调用 DSCI
    """
    csv_path = exp_dir / "res_dynamic_data.csv"
    sweep_var = config["sweep_var"]
    sweep_range = config["sweep_range"]
    static_params = config["static_params"]
    num_users = static_params.get("n", 20)

    # 检查进度
    completed_vals = []
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            if sweep_var in df_existing.columns:
                completed_vals = df_existing[sweep_var].tolist()
                print(f"Found {len(completed_vals)} completed steps.")
        except Exception as e:
            print(f"Warning: Could not read CSV: {e}")

    # 开始遍历
    for i, val in enumerate(sweep_range):
        # 断点检测 (考虑浮点数精度)
        if any(np.isclose(val, c_val) for c_val in completed_vals):
            continue

        print(f"[{i + 1}/{len(sweep_range)}] Running for {sweep_var} = {val}...")

        # 构造当前步的 custom_paras (浅拷贝字典)
        current_paras_dict = static_params.copy()

        # 处理不同维度的自变量
        if sweep_var in ["H_u", "F_u"]:
            current_paras_dict[sweep_var] = [val] * num_users
        else:
            current_paras_dict[sweep_var] = val

        try:
            best_val, best_sol, history, paras = run_dsci_experiment(
                custom_paras_dict=current_paras_dict, save_log=True
            )
            # 防护：best_sol 可能为 None 或非可迭代对象，避免 "None不可迭代" 错误
            if best_sol is None:
                print(
                    f"   Warning: best_sol is None for {sweep_var}={val}. Skipping..."
                )
                continue
            if not hasattr(best_sol, "__iter__"):
                print(
                    f"   Warning: best_sol is not iterable for {sweep_var}={val}. Skipping..."
                )
                continue
            try:
                X, Y, F_e, F_c = best_sol
            except Exception as e:
                print(
                    f"   Warning: unexpected best_sol structure at {sweep_var}={val}: {e}. Skipping..."
                )
                continue
            cut_points = split_points_matrix(np.array(X))
            clean_cuts = cut_points.astype(float)
            clean_cuts[clean_cuts[:, 0] == -1, 0] = paras.m
            new_row = {
                sweep_var: val,
                "avg_end_edge": np.mean(clean_cuts[:, 0]),
                "avg_edge_cloud": np.mean(clean_cuts[:, 1]),
                "total_utility": best_val,
            }
            # 增量写入 CSV
            df_new_row = pd.DataFrame([new_row])
            df_new_row.to_csv(
                csv_path, mode="a", index=False, header=not csv_path.exists()
            )
            print(f"   Success! Utility: {best_val:.4f}")

        except KeyboardInterrupt:
            print("\nExperiment paused by user. Progress saved.")
            break
        except Exception as e:
            print(f"   Error at {sweep_var}={val}: {e}. Skipping...")
            continue

    # 绘图
    if csv_path.exists():
        plot_resource_trend(csv_path=csv_path, save_dir=exp_dir)


if __name__ == "__main__":
    # 场景 1: 从头开始运行
    # from Src.paras import Paras
    # paras = Paras()
    # start_resource_experiment(sweep_var='F_u', sweep_range=np.arange(0.4e9, 3e9, 0.2e9), static_params=paras)
    #
    # # 场景 2: 继续运行未完成的实验
    # # resume_or_analyze_experiment(exp_dir=RESULT_DYNAMIC_PATH / "ResourceHetero_H_u_0128_1954")

    # 场景 3: 仅绘图
    resume_or_analyze_experiment(
        exp_dir=RESULT_DYNAMIC_PATH / "ResourceHetero_F_u_0201_1416", only_plot=True
    )
