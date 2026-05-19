"""
Src/Experiments/Exp1_SOTA/run_SOTA_baseline.py
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Src.Configs.paras import RESULT_ABLATION_PATH, RESULT_DIR
from Src.Objective.compute_accuracy import compute_expected_accuracy
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import objective
from Src.Utils.log_function import load_and_analyze_results


def evaluate_strategy(X, Y, F_e, F_c, paras, name: str = "Strategy") -> Dict:
    P = compute_layer_exit_probs(Y, paras)
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    acc_vec = compute_expected_accuracy(Y, P, paras)
    obj_val = objective(X, Y, F_e, F_c, paras)
    weighted_latency = paras.beta * np.sum(latency_vec)
    weighted_acc = paras.alpha * np.sum(acc_vec)
    return {
        "name": name,
        "latency": weighted_latency,
        "accuracy": weighted_acc,
        "objective": obj_val,
    }


def get_baseline_decisions(mode: str, paras, X_opt, Y_opt, F_e_opt, F_c_opt) -> Tuple:
    n, m = paras.n, paras.m
    X = np.zeros((n, m))
    Y = np.ones((n, m))
    Y[:, m - 1] = 0

    if mode == "DADS":
        # 固定点边云协同，无早退
        X[:, 50] = 1
        return X, Y, F_e_opt, F_c_opt

    elif mode == "CEED":
        # 早退，无协同（边缘计算）
        X[:, 0] = 1
        return X, Y_opt, F_e_opt, F_c_opt

    elif mode == "CutEdge":
        # 动态协同，无早退
        return X_opt, Y, F_e_opt, F_c_opt

    elif mode == "DCSI":
        # 端边云协同+早退（DSCI）
        return X_opt, Y_opt, F_e_opt, F_c_opt

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_baseline(PPO_path: Path, save_dir: Path):
    X_opt, Y_opt, F_e_opt, F_c_opt, _, paras = load_and_analyze_results(
        exp_dir=PPO_path, analysis=False
    )
    baseline_modes = [
        ("DADS", "DADS"),
        ("CEED", "CEED"),
        ("CutEdge", "CutEdge"),
        ("Ours", "DCSI"),
    ]
    results = []
    print(f"{'策略名称':<15} | {'Latency':<10} | {'Accuracy':<10} | {'Objective':<10}")
    print("-" * 60)
    for display_name, mode in baseline_modes:
        X, Y, Fe, Fc = get_baseline_decisions(
            mode, paras, X_opt, Y_opt, F_e_opt, F_c_opt
        )
        res = evaluate_strategy(X, Y, Fe, Fc, paras, name=display_name)
        results.append(res)
        print(
            f"{res['name']:<15} | {res['latency']:<10.4f} | {res['accuracy']:<10.4f} | {res['objective']:<10.4f}"
        )


if __name__ == "__main__":
    data_dir = RESULT_DIR / "Optimize" / "DSCI" / "DSCI_20260202_040737"
    save_dir = Path(RESULT_ABLATION_PATH)
    run_baseline(data_dir, save_dir)
