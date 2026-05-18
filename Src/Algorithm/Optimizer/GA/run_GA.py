"""
Src/Optimizer/GA/run_GA.py
"""

from pathlib import Path

from Src.Algo.Optimizer.GA.alg_GA import optimize_GA
from Src.Algo.Utils.log_function import save_experiment_results
from Src.paras import RESULT_GA_PATH, Paras

paras = Paras()
GA_hyperparams = {"population_size": 50, "generations": 150, "mutation_rate": 0.1}
GA_best_val, GA_best_sol, GA_history = optimize_GA(
    paras,
    population_size=GA_hyperparams["population_size"],
    generations=GA_hyperparams["generations"],
    mutation_rate=GA_hyperparams["mutation_rate"],
)
save_experiment_results(
    save_dir=Path(RESULT_GA_PATH),
    algo_name="GA",
    paras=paras,
    best_val=GA_best_val,
    best_sol=GA_best_sol,
    history=GA_history,
    hyper_params=GA_hyperparams,
)
