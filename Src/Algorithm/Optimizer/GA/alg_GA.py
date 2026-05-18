"""
Src/Optimizer/GA/alg_GA.py
"""

import random
import numpy as np
from Src.Objective.objective import objective


def optimize_GA(
    paras, population_size: int = 50, generations: int = 150, mutation_rate: float = 0.1
):
    """
    使用遗传算法优化混合连续/离散优化问题.

    参数
    ----------
    - paras : Paras 包含所有参数的对象（如用户数、层数、资源上限等）
    - population_size : int 种群大小
    - generations : int 迭代次数
    - mutation_rate : float 变异率

    返回
    -------
    - best_value:    最优目标函数值（最大值）。
    - best_solution: 包含 (X, Y, F_e, F_c) 的最优变量解。
    - history:       列表，记录每一代的最佳目标值（用于分析收敛过程）。
    """

    # ---------- 0. 基本尺寸 ----------
    n_rows = paras.n  # 用户数
    n_cols = paras.m  # 模型层数
    f_e_max = paras.f_e_max
    f_c_max = paras.f_c_max
    len_fe = len_fc = n_rows

    # 早退层布尔表
    early_exit_flags = [j in paras.E for j in range(n_cols)]

    # ---------- 2. 约束修复工具 ----------
    def repair_X(mat):
        """修复X使每行最多只有2个1"""
        for r in mat:
            while sum(r) > 2:
                idx = random.choice([j for j, v in enumerate(r) if v == 1])
                r[idx] = 0
        return mat

    def repair_Y(y):
        """修复Y的取值约束，非早退层强制为1，早退层限制在[0, 1)"""
        for j in range(n_cols):
            if early_exit_flags[j]:
                y[j] = max(0.0, min(1.0 - 1e-6, y[j]))
            else:
                y[j] = 1.0
        return y

    def repair_F(vec, f_max):
        """修复F使其元素非负且总和不超过f_max"""
        vec = [max(1e-6, v) for v in vec]  # 确保不为0
        tot = sum(vec)
        if tot > f_max and tot > 0:
            scale = f_max / tot
            vec = [v * scale for v in vec]
        return vec

    # ---------- 3. 初始化种群 ----------
    population = []
    for _ in range(population_size):
        # X
        X = [[0] * n_cols for _ in range(n_rows)]
        for i in range(n_rows):
            for j in random.sample(range(n_cols), k=random.choice([0, 1, 2])):
                X[i][j] = 1
        X = repair_X(X)

        # Y（只存一行）
        Y = [random.random() if early_exit_flags[j] else 1.0 for j in range(n_cols)]
        Y = repair_Y(Y)

        # F_e, F_c 随机分配
        F_e = repair_F([random.random() for _ in range(len_fe)], f_e_max)
        F_c = repair_F([random.random() for _ in range(len_fc)], f_c_max)
        population.append((X, Y, F_e, F_c))

    # ---------- 4. 评估 ----------
    def evaluate(ind):
        X, Y, F_e, F_c = ind
        X_mat = np.array(X, dtype=int)
        Y_mat = np.tile(np.array(Y, dtype=float), (n_rows, 1))
        F_e_arr = np.array(F_e, dtype=float).reshape(n_rows, 1)
        F_c_arr = np.array(F_c, dtype=float).reshape(n_rows, 1)
        return objective(X_mat, Y_mat, F_e_arr, F_c_arr, paras)

    def best_of(pop):
        best = max(pop, key=evaluate)
        return best, evaluate(best)

    best_ind, best_val = best_of(population)
    history = [best_val]
    print(f"{best_val}")

    # ---------- 5. 进化 ----------
    for generation in range(generations):
        new_pop = [best_ind]  # 精英保留
        while len(new_pop) < population_size:
            # ---- 选择（2路锦标赛）
            p1 = max(random.sample(population, 2), key=evaluate)
            p2 = max(random.sample(population, 2), key=evaluate)

            # ---- 交叉
            X1, Y1, Fe1, Fc1 = [], [], [], []
            X2, Y2, Fe2, Fc2 = [], [], [], []
            # X: 均匀交叉
            flat1 = [b for row in p1[0] for b in row]
            flat2 = [b for row in p2[0] for b in row]
            child_flat1, child_flat2 = [], []
            for a, b in zip(flat1, flat2):
                if random.random() < 0.5:
                    child_flat1.append(a)
                    child_flat2.append(b)
                else:
                    child_flat1.append(b)
                    child_flat2.append(a)
            for i in range(n_rows):
                X1.append(child_flat1[i * n_cols : (i + 1) * n_cols])
                X2.append(child_flat2[i * n_cols : (i + 1) * n_cols])
            X1, X2 = repair_X(X1), repair_X(X2)

            # Y: 均匀交叉（仅早退层可变）
            Y1 = p1[1][:]
            Y2 = p2[1][:]
            for j in range(n_cols):
                if early_exit_flags[j] and random.random() < 0.5:
                    Y1[j], Y2[j] = Y2[j], Y1[j]
            Y1, Y2 = repair_Y(Y1), repair_Y(Y2)

            # F_e/F_c: 均匀交叉
            Fe1, Fe2, Fc1, Fc2 = [], [], [], []
            for a, b in zip(p1[2], p2[2]):
                Fe1.append(a if random.random() < 0.5 else b)
                Fe2.append(b if random.random() < 0.5 else a)
            for a, b in zip(p1[3], p2[3]):
                Fc1.append(a if random.random() < 0.5 else b)
                Fc2.append(b if random.random() < 0.5 else a)
            Fe1, Fe2 = repair_F(Fe1, f_e_max), repair_F(Fe2, f_e_max)
            Fc1, Fc2 = repair_F(Fc1, f_c_max), repair_F(Fc2, f_c_max)

            # 变异时修复 Y 和 F
            def mutate(ind):
                X, Y, Fe, Fc = ind
                # 变异 X
                for i in range(n_rows):
                    ones = sum(X[i])
                    for j in range(n_cols):
                        if random.random() < mutation_rate:
                            if X[i][j] == 1:
                                X[i][j] = 0
                                ones -= 1
                            elif ones < 2:
                                X[i][j] = 1
                                ones += 1
                X = repair_X(X)

                # Y 变异
                for j in range(n_cols):
                    if early_exit_flags[j] and random.random() < mutation_rate:
                        Y[j] += random.gauss(0, 0.1)
                        Y[j] = max(0.0, min(1.0 - 1e-6, Y[j]))
                Y = repair_Y(Y)

                # F 变异
                for k in range(len_fe):
                    if random.random() < mutation_rate:
                        Fe[k] += random.gauss(0, 0.1 * f_e_max)
                        Fe[k] = max(0, min(Fe[k], f_e_max))  # 保证不超范围
                for k in range(len_fc):
                    if random.random() < mutation_rate:
                        Fc[k] += random.gauss(0, 0.1 * f_c_max)
                        Fc[k] = max(0, min(Fc[k], f_c_max))  # 保证不超范围
                Fe = repair_F(Fe, f_e_max)
                Fc = repair_F(Fc, f_c_max)

                return X, Y, Fe, Fc

            new_pop.append(mutate((X1, Y1, Fe1, Fc1)))
            if len(new_pop) < population_size:
                new_pop.append(mutate((X2, Y2, Fe2, Fc2)))

        population = new_pop
        best_ind, best_val = best_of(population)
        history.append(best_val)
        print(f"Generation {generation}: {best_val}")

    # --------- 6. 返回 ---------
    X_mat = np.array(best_ind[0], dtype=int)
    Y_mat = np.tile(np.array(best_ind[1]), (n_rows, 1))
    Fe_arr = np.array(best_ind[2]).reshape(n_rows, 1)
    Fc_arr = np.array(best_ind[3]).reshape(n_rows, 1)

    return best_val, (X_mat, Y_mat, Fe_arr, Fc_arr), history
