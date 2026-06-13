"""
目标函数
Src/Objective/Objective.py
"""

from Src.Phase2_Scheduler.Objective.compute_accuracy import compute_expected_accuracy
from Src.Phase2_Scheduler.Objective.compute_latency import compute_total_latency
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs


def objective(X, Y, F_e, F_c, paras):

    # 1.Exit probabilities
    P = compute_layer_exit_probs(Y, paras)

    # 2.Delay
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    latency = sum(latency_vec)

    # 3.Accuracy
    acc_vec = compute_expected_accuracy(Y, P, paras)
    acc = sum(acc_vec)

    return paras.alpha * acc - paras.beta * latency


def get_lat_and_acc(X, Y, F_e, F_c, paras):
    P = compute_layer_exit_probs(Y, paras)
    latency = sum(compute_total_latency(X, P, F_e, F_c, paras))
    acc = sum(compute_expected_accuracy(Y, P, paras))
    return latency, acc


# ==========================================
# Test Block for Objective
# ==========================================
if __name__ == "__main__":
    import numpy as np

    from Src.Phase2_Scheduler.paras import Paras

    print("\n" + "=" * 20 + " Objective Test " + "=" * 20)

    # -------------------------------------------------
    # 1. 初始化参数
    # -------------------------------------------------
    paras = Paras()
    n, m = paras.n, paras.m

    # -------------------------------------------------
    # 2. 构造变量
    # -------------------------------------------------
    X = np.zeros((n, m))
    # 给 user 0 一个典型切点，避免全零
    if m > 2:
        X[:, 0] = 1
        X[:, 1] = 1

    Y = np.ones((n, m))
    # 给 user 0 两个非 1 阈值，制造早退差异
    if m > 100:
        for boundary in paras.E:
            Y[:, boundary] = 0.9

    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)

    # -------------------------------------------------
    # 3. Exit Probabilities
    # -------------------------------------------------
    print("\n[1] Exit Probabilities")
    P = compute_layer_exit_probs(Y, paras)
    print("P shape:", P.shape)

    # 打印 user 0 的主要概率质量
    P0 = P[0]
    topk = np.argsort(-P0)[:10]
    print("User 0 top-10 exit probabilities:")
    for j in topk:
        # 修改为 .8f，直接打印小数
        print(f"  layer {j:4d}: P = {P0[j]:.8f}")
    print(f"sum(P[0]) = {np.sum(P0):.8f}")

    # -------------------------------------------------
    # 4. Latency
    # -------------------------------------------------
    print("\n[2] Latency")
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    latency = float(np.sum(latency_vec))

    for i in range(min(3, n)):
        # 修改为 .8f，直接打印小数
        print(f"User {i} latency = {latency_vec[i]:.8f} s")

    # 修改为 .8f，直接打印小数
    print(f"Total latency (sum over users) = {latency:.8f} s")

    # -------------------------------------------------
    # 5. Accuracy
    # -------------------------------------------------
    print("\n[3] Accuracy")
    acc_vec = compute_expected_accuracy(Y, P, paras)
    acc = float(np.sum(acc_vec))

    for i in range(min(3, n)):
        print(f"User {i} expected accuracy = {acc_vec[i]:.6f}")

    print(f"Total expected accuracy (sum over users) = {acc:.6f}")

    # -------------------------------------------------
    # 6. Objective Breakdown
    # -------------------------------------------------
    print("\n[4] Objective Breakdown")
    weighted_acc = paras.alpha * acc
    weighted_latency = paras.beta * latency
    obj = weighted_acc - weighted_latency

    print(f"alpha * acc   = {paras.alpha} * {acc:.6f} = {weighted_acc:.6f}")
    # 修改为 .8f，直接打印小数
    print(f"beta  * delay = {paras.beta} * {latency:.8f} = {weighted_latency:.8f}")
    print("-" * 50)
    # 修改为 .8f，直接打印小数
    print(f"Objective value = {obj:.8f}")

    # -------------------------------------------------
    # 7. 对照 objective() 函数
    # -------------------------------------------------
    print("\n[5] Sanity Check with objective()")
    obj_func = objective(X, Y, F_e, F_c, paras)
    # 修改为 .8f，直接打印小数
    print(f"objective() returns = {obj_func:.8f}")

    diff = abs(obj - obj_func)
    if diff < 1e-9:
        print("✅ Test passed: manual breakdown matches objective().")
    else:
        # 修改为 .10f，确保能看清微小的误差
        print(f"❌ Test failed: diff = {diff:.10f}")

    print("\n" + "=" * 60)
