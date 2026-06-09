"""
Src/Optimizer/DSCI/run_DSCI.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Src.Algorithm.Optimizer.DSCI.agent import PPOAgent
from Src.Algorithm.Utils.log_function import save_experiment_results
from Src.paras import RESULT_PPO_PATH, Paras

_DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "gamma": 0.95,
    "lam": 0.95,
    "lr": 1e-4,
    "eps_clip": 0.15,
    "max_epochs": 200,
    "target_steps": 1500,
    "k_epochs": 10,
    "entropy_coef": 0.01,
    "entropy_decay": 0.995,
    "grad_clip": 0.5,
    "obj_scale": 1000.0,
    "outer_ema": 0.02,
}


def _build_ppo_params(custom_ppo_hyperparams: dict | None) -> dict:
    params = dict(_DEFAULT_PPO_PARAMS)
    if custom_ppo_hyperparams:
        params.update(custom_ppo_hyperparams)
    return params


def infer_one_round(
    paras: Paras,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_state_dict: dict | None = None,
    agent: PPOAgent | None = None,
    custom_ppo_hyperparams: dict | None = None,
    F_e: np.ndarray | None = None,
    F_c: np.ndarray | None = None,
    deterministic: bool = True,
    outer_ema: float = 1.0,
    record_transitions: bool = False,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Paras]:
    """
    Single testbed decision round: one PPO episode + closed-form resource allocation.

    Does not run PPO updates or the multi-epoch ``train()`` loop. Load a trained
    policy via ``checkpoint_path`` / ``checkpoint_state_dict``, or pass a
    pre-built ``agent`` (e.g. with ``agent.load_checkpoint`` already called).

    Returns:
        objective value, (X, Y, F_e, F_c), paras
    """
    if agent is None:
        agent = PPOAgent(paras, _build_ppo_params(custom_ppo_hyperparams))
    else:
        agent.paras = paras

    if checkpoint_state_dict is not None:
        agent.load_policy_state_dict(checkpoint_state_dict)
    elif checkpoint_path is not None:
        agent.load_checkpoint(checkpoint_path)

    X, Y, F_e_out, F_c_out, obj = agent.act_one_episode(
        F_e=F_e,
        F_c=F_c,
        deterministic=deterministic,
        outer_ema=outer_ema,
        record_transitions=record_transitions,
    )
    return float(obj), (X, Y, F_e_out, F_c_out), paras


def run_dsci_experiment(
    custom_paras_dict=None, custom_ppo_hyperparams=None, save_log=True
):
    """
    封装 DSCI 运行逻辑，支持动态参数注入
    """
    # 1. 初始化参数
    paras = Paras.from_dict(custom_paras_dict or {})

    ppo_params = _build_ppo_params(custom_ppo_hyperparams)

    # 3. 算法优化
    agent = PPOAgent(paras, ppo_params)
    best_val, best_sol, history = agent.train()

    # 4. 日志保存
    if save_log and best_sol is not None:
        save_experiment_results(
            save_dir=Path(RESULT_PPO_PATH),
            algo_name="DSCI",
            paras=paras,
            best_val=best_val,
            best_sol=best_sol,
            history=history,
            hyper_params=ppo_params,
            extra_logs=agent.logs,
        )
    elif best_sol is None:
        print("[Warning] DSCI optimization returned no solution.")

    return best_val, best_sol, history, paras


if __name__ == "__main__":
    run_dsci_experiment(save_log=True)
