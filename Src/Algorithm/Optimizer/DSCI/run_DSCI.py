"""
Src/Optimizer/DSCI/run_DSCI.py
"""
from pathlib import Path
from Src.paras import Paras, RESULT_PPO_PATH
from Src.Optimizer.DSCI.agent import PPOAgent
from Src.Utils.log_function import save_experiment_results


def run_dsci_experiment(custom_paras_dict=None, custom_ppo_hyperparams=None, save_log=True):
    """
    封装 DSCI 运行逻辑，支持动态参数注入
    """
    # 1. 初始化参数
    paras = Paras.from_dict(custom_paras_dict or {})

    # 2. DSCI 超参数
    ppo_params = {
        'gamma': 0.95,
        'lam': 0.95,
        'lr': 1e-4,
        'eps_clip': 0.15,
        'max_epochs': 200,
        'target_steps': 1500,
        'k_epochs': 10,
        'entropy_coef': 0.01,
        'entropy_decay': 0.995,
        'grad_clip': 0.5,
        'obj_scale': 1000.0
    }
    if custom_ppo_hyperparams:
        ppo_params.update(custom_ppo_hyperparams)

    # 3. 算法优化
    agent = PPOAgent(paras, ppo_params)
    best_val, best_sol, history = agent.train()

    # 4. 日志保存
    if save_log:
        save_experiment_results(
            save_dir=Path(RESULT_PPO_PATH),
            algo_name="DSCI",
            paras=paras,
            best_val=best_val,
            best_sol=best_sol,
            history=history,
            hyper_params=ppo_params,
            extra_logs=agent.logs
        )

    return best_val, best_sol, history, paras


if __name__ == '__main__':
    run_dsci_experiment(save_log=True)
