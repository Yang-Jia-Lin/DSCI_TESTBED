"""
计算退出概率
Src/Objective/compute_P.py
"""

import numpy as np


# one user in one layer (p_ij)
def _get_independent_prob(Y_ij, j, exit_rates):
    closest_idx = round(Y_ij * 100)
    # 确保整数在 0 到 100 之间
    if closest_idx < 0:
        closest_idx = 0
    elif closest_idx > 100:
        closest_idx = 100
    return exit_rates[closest_idx, j] / 100


def compute_layer_exit_probs(Y, paras):
    n, m = Y.shape
    p = np.zeros((n, m))
    P = np.zeros((n, m))

    for i in range(n):
        # 1) 计算独立退出概率
        for j in range(m):
            if j in paras.E:
                p[i, j] = _get_independent_prob(Y[i, j], j, paras.rates)  # 查表

        # 2) 计算组合退出概率
        for j in range(m):
            if j == 0:
                P[i, j] = p[i, j]
            else:
                # noinspection PyTypeChecker
                P[i, j] = p[i, j] * np.prod(1 - p[i, :j])
        # 3) 整体归一化
        total = P[i].sum()
        if total > 0:
            P[i] = P[i] / total
        else:
            # 全零时，按业务给个默认分配，比如只退出最后一层
            P[i, -1] = 1
    return P


# ==========================================
# Test Block
# ==========================================
if __name__ == "__main__":
    from Src.Phase2_Scheduler.paras import Paras

    print(">>> 正在初始化参数并读取真实数据...")

    # 初始化 Paras 对象
    paras = Paras()
    assert paras.rates is not None, "Rates data not loaded"
    print(f">>> 数据加载完成. Rates shape: {paras.rates.shape}")

    n = paras.n
    m = paras.m

    # 2. 构造测试变量 Y
    # 默认全为 1 (根据定义，全为1意味着全不退出，概率全为0)
    Y = np.ones([n, m])

    # 设置几个典型值：
    # Layer 0: 0.1 (低阈值，容易退出)
    # Layer 1: 0.5 (中等)
    # Layer 2: 0.9 (高阈值，难退出)
    if m > 3:
        Y[0, 57] = 1
        Y[0, 103] = 1
        # 如果还有更多层，保持为 1.0

    print(f"\n>>> 测试用户 [0,57] 的阈值 Y 设置: {Y[0, 50:60]} ...")

    # 3. 运行你的函数计算 P (累积概率)
    P_matrix = compute_layer_exit_probs(Y, paras)
    P_user = P_matrix[0]

    # 4. 手动回算小写 p (独立概率) 用于展示对比
    # 注意：这里复用了函数内部的逻辑来展示中间变量
    p_independent = np.zeros(m)
    for j in range(m):
        if j in paras.E:
            val_y = Y[0, j]
            # 这里的逻辑和你函数里保持一致
            idx = round(val_y * 100)
            if idx < 0:
                idx = 0
            if idx > 100:
                idx = 100
            assert paras.rates is not None, "Rates data not loaded"
            p_independent[j] = paras.rates[idx, j] / 100.0
        else:
            p_independent[j] = 0.0

    # 5. 打印
    print("\n" + "=" * 70)
    print(
        f"{'Layer':<8} | {'Y_ij (阈值)':<12} | {'p_ij (独立概率)':<15} | {'P_ij (最终概率)':<15}"
    )
    print("-" * 70)
    for j in range(m):
        is_exit = " [Exit]" if j in paras.E else ""
        print(
            f"{str(j) + is_exit:<8} | {Y[0, j]:<12.2f} | {p_independent[j]:<15.4f} | {P_user[j]:<15.4f}"
        )
    print("-" * 70)
    print(f"Sum(P_ij) = {P_user.sum():.6f}")

    # 简单的逻辑检查
    if P_user.sum() > 1.0001:
        print("警告: 概率和大于1，请检查归一化逻辑。")
    elif P_user.sum() < 0.99:
        print("提示: 概率和小于1，意味着有概率流失到最后没有退出 (符合 y=1 设定)。")
    else:
        print("状态: 概率和正常 (approx 1)。")
