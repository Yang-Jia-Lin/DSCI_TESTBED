"""
计算全部任务的准确率
Src/Objective/compute_accuracy.py
"""

import numpy as np

from Src.paras import Paras


def _get_acc(Y_ij, j, exit_rates):
    closest_idx = round(Y_ij * 100)

    # 确保整数在 0 到 100 之间
    if closest_idx < 0:
        closest_idx = 0
    elif closest_idx > 100:
        closest_idx = 100

    # print(f"my test: exit_rates is {exit_rates[0, 127]:.6f}, shape is {exit_rates.shape}")
    return exit_rates[closest_idx, j] / 100


def compute_expected_accuracy(Y, P, paras):
    n, m = Y.shape
    acc = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if j in paras.E:
                acc[i, j] = _get_acc(Y[i, j], j, paras.accs)
                # print(f"第{i}行，第{j}列，阈值为{Y[i,j]}，精度为{acc[i, j]}，退出概率为{P[i,j]}")
        acc[i, m - 1] = paras.accs[100, m - 1] / 100.0
        # acc[i, m - 1] = 0.8651
    accuracy = acc * P
    return np.sum(accuracy, axis=1)


if __name__ == "__main__":
    from Src.Algorithm.Objective.compute_P import compute_layer_exit_probs
    from Src.Algorithm.Utils.parsing_data import parsing_rate_and_acc

    paras = Paras()
    paras.rates, paras.accs = parsing_rate_and_acc(paras)
    n = paras.n
    m = paras.m

    # Variable
    X = np.zeros([n, m])
    Y = np.ones([n, m])
    F_e = np.ones((n, 1)) * paras.f_e_max / n
    F_c = np.ones((n, 1)) * paras.f_c_max / n

    P = compute_layer_exit_probs(Y, paras)
    acc_vec = compute_expected_accuracy(Y, P, paras)
