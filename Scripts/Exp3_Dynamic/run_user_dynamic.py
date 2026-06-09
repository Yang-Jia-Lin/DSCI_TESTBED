"""
Src/Exp2_Dynamic/run_user_dynamic.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Scripts.Exp3_Dynamic.plot_decision import plot_X, plot_Y
from Scripts.Exp3_Dynamic.plot_latency_stacked import plot_latency_stacked
from Src.Algorithm.Objective.compute_latency import compute_5_latency
from Src.Algorithm.Objective.compute_P import compute_layer_exit_probs
from Src.Algorithm.Optimizer.DSCI.run_DSCI import run_dsci_experiment
from Src.Algorithm.Utils.utils_function import NumpyEncoder
from Src.paras import RESULT_DYNAMIC_PATH, Paras


def plot_user_dynamic(
    X, Y, F_e, F_c, paras, custom_name=None, sweep_var=None, sweep_range=None
):
    """
    绘制并保存实验数据、图形以及实验元配置
    """
    # ========= 1) 路径处理 ========
    prefix = f"UserHetero_{sweep_var}" if sweep_var else "UserHetero"
    folder_name = (
        f"{prefix}_{custom_name}"
        if custom_name
        else f"{prefix}_{datetime.now().strftime('%m%d_%H%M')}"
    )
    save_dir = Path(RESULT_DYNAMIC_PATH) / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= 2) 构造并保存配置参数 ========
    if hasattr(paras, "to_dict"):
        paras_dict = paras.to_dict()
    else:
        paras_dict = {k: v for k, v in vars(paras).items() if not k.startswith("__")}

    config_data = {
        "sweep_var": sweep_var,
        "sweep_range": sweep_range,
        "System_Parameters": paras_dict,
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, cls=NumpyEncoder)

    # ========= 3) 数据计算与绘图 ========
    P = compute_layer_exit_probs(Y, paras)
    T1, T2, T3, T4, T5 = compute_5_latency(X, P, F_e, F_c, paras)
    user_labels = [str(i + 1) for i in range(paras.n)]

    plot_latency_stacked(user_labels, (T1, T2, T3, T4, T5), save_dir=save_dir)
    plot_X(X, paras.E, save_dir=save_dir)
    plot_Y(Y, paras.E, save_dir=save_dir)

    # 保存计算结果
    if not (save_dir / "solution.npz").exists():
        np.savez(save_dir / "solution.npz", X=X, Y=Y, F_e=F_e, F_c=F_c)

    print(f"Results and config saved to: {save_dir}")


def dynamic_with_data(solution_dir: Path):
    solution_dir = Path(solution_dir)
    # ========= 1) 找数据 ========
    npz_path = solution_dir / "solution.npz"
    json_path = solution_dir / "config.json"
    if not json_path.exists() or not npz_path.exists():
        print(f"Error: 路径 {solution_dir} 下缺少必要文件")
        return

    # ========= 2) 加载数据 ========
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    paras = Paras.from_dict(config.get("System_Parameters", {}))
    data = np.load(npz_path, allow_pickle=True)
    X, Y, F_e, F_c = data["X"], data["Y"], data["F_e"], data["F_c"]

    # ========= 3) 绘图 (透传原有的 sweep 信息) ========
    plot_user_dynamic(
        X,
        Y,
        F_e,
        F_c,
        paras,
        custom_name=solution_dir.name,
        sweep_var=config.get("sweep_var"),
        sweep_range=config.get("sweep_range"),
    )


def dynamic_without_data(n, F_u, H_u, sweep_var=None, sweep_range=None):
    # ========= 1) 运行实验 ========
    custom_paras_dict = {"n": n, "F_u": F_u, "H_u": H_u}
    best_val, best_sol, history, paras = run_dsci_experiment(
        custom_paras_dict=custom_paras_dict, save_log=True
    )
    if best_sol is None:
        print("[Error] Optimization failed to find a solution.")
        return
    X, Y, F_e, F_c = best_sol

    # ========= 2) 画图并保存配置 ========
    plot_user_dynamic(
        X, Y, F_e, F_c, paras, sweep_var=sweep_var, sweep_range=sweep_range
    )


if __name__ == "__main__":
    # 重新运行
    # n = 18
    # F_u_list = [0.1e9] * 6 + [1.0e9] * 6 + [8.0e9] * 6
    # H_u_list = [2.0] * 18
    #
    # dynamic_without_data(
    #     n = 18,
    #     F_u=np.array(F_u_list, dtype=np.float32),
    #     H_u=np.array(H_u_list, dtype=np.float32),
    #     sweep_var="F_u_Hetero",
    #     sweep_range=F_u_list
    # )

    # 直接运行PPO的
    from Src.paras import RESULT_DIR

    dynamic_with_data(RESULT_DIR / "Optimize" / "PPO" / "PPO_20260129_224216")
