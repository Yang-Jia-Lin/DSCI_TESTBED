"""
Src/Optimizer/BF/run_BF.py
"""
from pathlib import Path
from Src.paras import Paras, RESULT_BF_PATH
from Src.Optimizer.BF.alg_BF import optimize_BF
from Src.Utils.log_function import save_experiment_results


paras = Paras()
BF_hyperparams = {
    'max_iter': 5,
    'restarts': 2,
    'threshold_step': 0.05
}
BF_best_val, BF_best_sol, BF_history = optimize_BF(
    paras,
    max_iter=BF_hyperparams['max_iter'],
    restarts=BF_hyperparams['restarts'],
    threshold_step=BF_hyperparams['threshold_step']
)
save_experiment_results(
    save_dir=Path(RESULT_BF_PATH),
    algo_name="BF",
    paras=paras,
    best_val=BF_best_val,
    best_sol=BF_best_sol,
    history=BF_history,
    hyper_params=BF_hyperparams
)
