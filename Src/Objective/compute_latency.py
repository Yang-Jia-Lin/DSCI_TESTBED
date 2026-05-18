"""
计算所有任务总时延
Src/Objective/compute_latency.py
"""
import numpy as np
from Src.paras import Paras
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_exit_points import compute_exit_points


def _compute_end_to_edge_delay(d_i, h_i, B_e, G, delta):
    R_i = (B_e * 1e6) * np.log2(1 + (h_i * G) / delta)
    return d_i / R_i


def _compute_edge_to_cloud_delay(d_e2c, B_c):
    return d_e2c / (B_c * 1e6)


def _compute_local_computation_delay(cut_points_i, P_i, C_i, f_u):
    """
    Local Range: [0, cut0)
    例如 cut0=0，则 range(0,0) 为空，时延为0。
    """
    cut0 = int(cut_points_i[0])
    m = len(C_i)

    # 边界规整
    if cut0 <= 0:
        return 0.0
    if cut0 > m:
        cut0 = m

    f_u = float(f_u) * 1e9
    if f_u <= 0:
        return float("inf")

    # 计算范围：[0, cut0)
    T_expected = 0.0

    # (1) 在本地段内退出：j in [0, cut0)
    for j in range(0, cut0):
        T_expected += P_i[j] * (sum(C_i[:j + 1]) / f_u)

    # (2) 退出在更后段：j in [cut0, m) => 本地必须算满到 cut0-1
    prob_reach_edge = float(sum(P_i[cut0:])) if cut0 < m else 0.0
    if prob_reach_edge > 0:
        T_expected += prob_reach_edge * (sum(C_i[:cut0]) / f_u)

    return float(T_expected)


def _compute_edge_computation_delay(cut_points_i, P_i, C, f_e):
    """
    Edge Range: [cut0, cut1)
    注意：需处理 cut1=-1 的情况，代表 Edge 负责到底
    """
    cut0 = int(cut_points_i[0])
    cut1 = int(cut_points_i[1])
    m = len(C)

    # 边界规整
    if cut0 < 0:
        cut0 = 0
    if cut0 > m:
        cut0 = m

    # 预处理 cut1：如果是 -1，表示没有切给云，Edge 负责到最后 (m)
    effective_cut1 = m if cut1 == -1 else cut1
    if effective_cut1 < 0:
        effective_cut1 = 0
    if effective_cut1 > m:
        effective_cut1 = m

    # 如果 range 为空，则 Edge 不计算
    if effective_cut1 <= cut0:
        return 0.0

    f_e = float(f_e) * 1e9
    if f_e <= 0:
        return float("inf")

    T_expected = 0.0

    # (1) 在 edge 段内退出：j in [cut0, effective_cut1)
    for j in range(cut0, effective_cut1):
        T_expected += P_i[j] * (sum(C[cut0: j + 1]) / f_e)

    # (2) 退出在更后段（进入 cloud）：j in [effective_cut1, m)
    # 只有当 cut1 != -1（确实存在 cloud 段）时，才需要这项固定前缀
    if effective_cut1 < m:
        prob_reach_cloud = float(sum(P_i[effective_cut1:]))
        if prob_reach_cloud > 0:
            T_expected += prob_reach_cloud * (sum(C[cut0: effective_cut1]) / f_e)

    return float(T_expected)



def _compute_cloud_computation_delay(cut_points_i, P_i, C, f_c):
    """
    Cloud Range: [cut1, m)
    """
    cut1 = int(cut_points_i[1])
    m = len(C)

    if cut1 == -1 or cut1 >= m:
        return 0.0
    if cut1 < 0:
        cut1 = 0

    f_c = float(f_c) * 1e9
    if f_c <= 0: return float("inf")

    T_expected = 0.0
    for j in range(cut1, m):
        T_expected += P_i[j] * (sum(C[cut1: j + 1]) / f_c)
    return float(T_expected)


def compute_total_latency(X, P, F_e, F_c, paras):
    n = X.shape[0]
    m = X.shape[1]
    C = paras.C
    D = paras.D
    H = paras.H_u
    F_u = paras.F_u
    b_e = paras.b_e
    b_c = paras.b_c
    G = paras.G
    delta = paras.delta

    T = np.zeros(n, dtype=np.float64)
    cut_points = compute_exit_points(X, paras)

    for i in range(n):
        T[i] = 0
        cut0 = int(cut_points[i][0])
        cut1 = int(cut_points[i][1])

        P_i = P[i]
        f_e = float(np.asarray(F_e).reshape(-1)[i])
        f_c = float(np.asarray(F_c).reshape(-1)[i])
        f_u = float(np.asarray(F_u).reshape(-1)[i])
        h_i = float(np.asarray(H).reshape(-1)[i])

        # ---- Local computation ----
        if cut0 > 0:
            T[i] += _compute_local_computation_delay((cut0, cut1), P_i, C, f_u)

        # 进入 edge 的概率（退出层 >= cut0）
        prob_reach_edge = float(sum(P_i[cut0:])) if 0 <= cut0 < m else 0.0

        # ---- U->E transmission & Edge computation ----
        if 0 <= cut0 < m and prob_reach_edge > 0:
            T[i] += prob_reach_edge * _compute_end_to_edge_delay(float(D[cut0]), h_i, b_e, G, delta)
            T[i] += _compute_edge_computation_delay((cut0, cut1), P_i, C, f_e)

        # 进入 cloud 的概率（退出层 >= cut1），仅当 cut1 有效且存在 cloud 段
        prob_reach_cloud = float(sum(P_i[cut1:])) if 0 <= cut1 < m else 0.0

        # ---- E->C transmission & Cloud computation ----
        if 0 <= cut1 < m and cut1 != -1 and prob_reach_cloud > 0:
            d_i_2 = float(D[cut1])
            T[i] += prob_reach_cloud * _compute_edge_to_cloud_delay(d_i_2, b_c)
            T[i] += _compute_cloud_computation_delay((cut0, cut1), P_i, C, f_c)

    return T


def compute_5_latency(X, P, F_e, F_c, paras):
    n = X.shape[0]
    m = X.shape[1]
    C = paras.C
    D = paras.D
    H = paras.H_u
    F_u = paras.F_u
    b_e = paras.b_e
    b_c = paras.b_c
    G = paras.G
    delta = paras.delta

    T1 = np.zeros(n, dtype=np.float64)
    T2 = np.zeros(n, dtype=np.float64)
    T3 = np.zeros(n, dtype=np.float64)
    T4 = np.zeros(n, dtype=np.float64)
    T5 = np.zeros(n, dtype=np.float64)
    cut_points = compute_exit_points(X, paras)

    for i in range(n):
        cut0 = int(cut_points[i][0])
        cut1 = int(cut_points[i][1])

        P_i = P[i]
        f_e = float(np.asarray(F_e).reshape(-1)[i])
        f_c = float(np.asarray(F_c).reshape(-1)[i])
        f_u = float(np.asarray(F_u).reshape(-1)[i])
        h_i = float(np.asarray(H).reshape(-1)[i])

        # ---- Local computation ----
        if cut0 > 0:
            T1[i] = _compute_local_computation_delay((cut0, cut1), P_i, C, f_u)

        # 进入 edge 的概率（退出层 >= cut0）
        prob_reach_edge = float(sum(P_i[cut0:])) if 0 <= cut0 < m else 0.0

        # ---- U->E transmission & Edge computation ----
        if 0 <= cut0 < m and prob_reach_edge > 0:
            T2[i] = prob_reach_edge * _compute_end_to_edge_delay(float(D[cut0]), h_i, b_e, G, delta)
            T3[i] = _compute_edge_computation_delay((cut0, cut1), P_i, C, f_e)

        # 进入 cloud 的概率（退出层 >= cut1），仅当 cut1 有效且存在 cloud 段
        prob_reach_cloud = float(sum(P_i[cut1:])) if 0 <= cut1 < m else 0.0

        # ---- E->C transmission & Cloud computation ----
        if 0 <= cut1 < m and cut1 != -1 and prob_reach_cloud > 0:
            d_i_2 = float(D[cut1])
            T4[i] =  prob_reach_cloud * _compute_edge_to_cloud_delay(d_i_2, b_c)
            T5[i] =  _compute_cloud_computation_delay((cut0, cut1), P_i, C, f_c)
    return T1, T2, T3, T4, T5


def compute_user_latency(
    u: int,
    cut0: int,
    cut1: int,
    P_row: np.ndarray,
    F_e_u: float,
    F_c_u: float,
    paras,
) -> float:
    """
    严格等价于 compute_total_latency() 里对单个用户 i 的那一段计算。
    只计算用户 u 的期望总时延 T[u]。

    参数:
      - cut0, cut1: 由 compute_exit_points 得到的切分点（cut1 可能为 -1）
      - P_row: P[u]，长度 m
      - F_e_u, F_c_u: 分配给该用户的 edge/cloud 频率 (GHz，与你的 compute_total_latency 一致)
    """
    C = np.asarray(paras.C, dtype=np.float64)
    D = np.asarray(paras.D, dtype=np.float64)
    H = np.asarray(paras.H_u, dtype=np.float64).reshape(-1)
    F_u = np.asarray(paras.F_u, dtype=np.float64).reshape(-1)

    b_e = float(paras.b_e)
    b_c = float(paras.b_c)
    G = float(paras.G)
    delta = float(paras.delta)

    m = len(C)

    cut0 = int(cut0)
    cut1 = int(cut1)

    P_i = np.asarray(P_row, dtype=np.float64).reshape(-1)

    f_e = float(F_e_u)
    f_c = float(F_c_u)
    f_u = float(F_u[u])
    h_i = float(H[u])

    T = 0.0

    # ---- Local computation ----
    if cut0 > 0:
        T += _compute_local_computation_delay((cut0, cut1), P_i, C, f_u)

    # 进入 edge 的概率（退出层 >= cut0）
    prob_reach_edge = float(np.sum(P_i[cut0:])) if 0 <= cut0 < m else 0.0

    # ---- U->E transmission & Edge computation ----
    if 0 <= cut0 < m and prob_reach_edge > 0:
        T += prob_reach_edge * _compute_end_to_edge_delay(float(D[cut0]), h_i, b_e, G, delta)
        T += _compute_edge_computation_delay((cut0, cut1), P_i, C, f_e)

    # 进入 cloud 的概率（退出层 >= cut1），仅当 cut1 有效且存在 cloud 段
    prob_reach_cloud = float(np.sum(P_i[cut1:])) if 0 <= cut1 < m else 0.0

    # ---- E->C transmission & Cloud computation ----
    if 0 <= cut1 < m and cut1 != -1 and prob_reach_cloud > 0:
        T += prob_reach_cloud * _compute_edge_to_cloud_delay(float(D[cut1]), b_c)
        T += _compute_cloud_computation_delay((cut0, cut1), P_i, C, f_c)

    return float(T)


# ==========================================
# Test Block for Latency
# ==========================================
if __name__ == "__main__":
    print(">>> 初始化参数...")
    paras = Paras()
    n, m = paras.n, paras.m

    # 2. 准备输入数据
    X = np.zeros((n, m))
    X[0][50] = 1
    X[0][100] = 1

    Y = np.ones((n, m))
    # Y[0, 57] = 0.9
    # Y[0, 103] = 0.8

    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)

    P = compute_layer_exit_probs(Y, paras)
    cut_points = compute_exit_points(X, paras)
    c0 = int(cut_points[0][0])
    c1 = int(cut_points[0][1])


    # 3. 手动拆解时延计算 (使用刚算出来的 c0, c1)
    print(f"\n{'=' * 20} User 0 Latency Breakdown {'=' * 20}")
    print(f"cut point is ({c0}, {c1})")
    P_i = P[0]
    f_e = F_e[0].item()
    f_c = F_c[0].item()
    f_u = float(paras.F_u[0])
    h_i = float(paras.H_u[0])
    cut_tuple = (c0, c1)

    # A. Local
    t_local = _compute_local_computation_delay(cut_tuple, P_i, paras.C, f_u)
    if c0 > 0:
        print(f"1. Local Comp Layers [0, {c0}): \t{t_local:.12f} s")
    else:
        print(f"1. Local Comp (None): \t0 s")

    # B. Trans U->E
    if c0 < m:
        data_u2e = float(paras.D[c0])
        t_trans_1 = _compute_end_to_edge_delay(data_u2e, h_i, paras.b_e, paras.G, paras.delta)
        print(f"2. Trans U->E (Data D[{c0}]):   \t{t_trans_1:.12f} s")
    else:
        t_trans_1 = 0.0
        print(f"2. Trans U->E (None):   \t0 s")

    # C. Edge Comp
    if c0 < m:
        t_edge = _compute_edge_computation_delay(cut_tuple, P_i, paras.C, f_e)
        print(f"3. Edge Comp Layers [{c0}, {c1}): \t{t_edge:.12f} s")
    else:
        t_edge = 0.0
        print(f"3. Edge Comp Layers (None): \t0 s")

    # D. Trans E->C
    if 0 < c1 < m:
        data_e2c = float(paras.D[c1])
        t_trans_2 = _compute_edge_to_cloud_delay(data_e2c, paras.b_c)
        print(f"4. Trans E->C (Data D[{c1}]):   \t{t_trans_2:.12f} s")
    else:
        t_trans_2 = 0.0
        print(f"4. Trans E->C (None):   \t0 s")

    # E. Cloud Comp
    if 0 < c1 < m:
        t_cloud = _compute_cloud_computation_delay(cut_tuple, P_i, paras.C, f_c)
        print(f"5. Cloud Comp Layers [{c1}, m): \t{t_cloud:.12f} s")
    else:
        t_cloud = 0.0
        print(f"5. Cloud Comp Layers (None): \t0 s")

    # F. Sum
    manual_total = t_local + t_trans_1 + t_edge + t_trans_2 + t_cloud
    print("-" * 60)
    print(f"Manual Total Sum: \t\t\t{manual_total:.6f} s")

    print(f"\n{'=' * 20} User 0 Latency Details {'=' * 20}")
    # ---- 基础打印：算力资源 ----
    print("\n[Resources]")
    print(f"  f_u = {f_u:.6f} GHz  ({f_u * 1e9:.3e} Hz)")
    print(f"  f_e = {f_e:.6f} GHz  ({f_e * 1e9:.3e} Hz)")
    print(f"  f_c = {f_c:.6f} GHz  ({f_c * 1e9:.3e} Hz)")

    # ---- 概率质量：到达切点概率 ----
    prob_reach_edge = float(np.sum(P_i[c0:])) if 0 <= c0 < m else 0.0
    prob_reach_cloud = float(np.sum(P_i[c1:])) if 0 <= c1 < m else 0.0

    print("\n[Exit Probabilities]")
    print(f"  sum(P_i) = {float(np.sum(P_i)):.6f}")
    if 0 <= c0 < m:
        print(f"  Pr(reach edge)  = sum(P_i[{c0}:]) = {prob_reach_edge:.6f}")
    else:
        print(f"  Pr(reach edge)  = 0.000000 (invalid c0)")
    if 0 <= c1 < m:
        print(f"  Pr(reach cloud) = sum(P_i[{c1}:]) = {prob_reach_cloud:.6f}")
    else:
        print(f"  Pr(reach cloud) = 0.000000 (invalid c1)")

    # 可选：打印 P_i 的 Top-10，帮助解释 edge 为什么小
    topk = np.argsort(-P_i)[:10]
    print("\n[P_i Top-10]")
    for j in topk:
        print(f"  j={int(j):4d}  P_i[j]={float(P_i[j]):.6e}")

    # A. Local computation
    t_local = _compute_local_computation_delay(cut_tuple, P_i, paras.C, f_u)
    if c0 > 0:
        print(f"\n1. Local Comp Layers [0, {c0}):")
        print(f"   - used f_u = {f_u:.6f} GHz")
        print(f"   - result   = {t_local:.12f} s")
    else:
        print(f"\n1. Local Comp (None):\n   - result   = 0 s")

    # B. Trans U->E (按到达 edge 概率加权，与 compute_total_latency 对齐)
    if 0 <= c0 < m and prob_reach_edge > 0:
        data_u2e = float(paras.D[c0])
        t_trans_1_raw = _compute_end_to_edge_delay(data_u2e, h_i, paras.b_e, paras.G, paras.delta)
        t_trans_1 = prob_reach_edge * t_trans_1_raw
        print(f"\n2. Trans U->E (Data D[{c0}]):")
        print(f"   - D[{c0}]      = {data_u2e} (raw unit)")
        print(f"   - bandwidth b_e= {paras.b_e} (-> {paras.b_e * 1e6:.3e})")
        print(f"   - reach prob   = {prob_reach_edge:.6f}")
        print(f"   - raw delay    = {t_trans_1_raw:.12f} s")
        print(f"   - expected     = {t_trans_1:.12f} s")
    else:
        t_trans_1 = 0.0
        print(f"\n2. Trans U->E (None):\n   - expected = 0 s")

    # C. Edge computation（只有可能到达 edge 才有意义；函数内部已按 P_i 做期望）
    if 0 <= c0 < m and prob_reach_edge > 0:
        t_edge = _compute_edge_computation_delay(cut_tuple, P_i, paras.C, f_e)
        print(f"\n3. Edge Comp Layers [{c0}, {c1}):")
        print(f"   - used f_e = {f_e:.6f} GHz")
        print(f"   - result   = {t_edge:.12f} s")
    else:
        t_edge = 0.0
        print(f"\n3. Edge Comp (None):\n   - result   = 0 s")

    # D. Trans E->C (按到达 cloud 概率加权，与 compute_total_latency 对齐)
    # 注意：cut1 == -1 表示 edge 负责到底，无 E->C
    if 0 <= c1 < m and c1 != -1 and prob_reach_cloud > 0:
        data_e2c = float(paras.D[c1])
        t_trans_2_raw = _compute_edge_to_cloud_delay(data_e2c, paras.b_c)
        t_trans_2 = prob_reach_cloud * t_trans_2_raw
        print(f"\n4. Trans E->C (Data D[{c1}]):")
        print(f"   - D[{c1}]      = {data_e2c} (raw unit)")
        print(f"   - bandwidth b_c= {paras.b_c} (-> {paras.b_c * 1e6:.3e})")
        print(f"   - reach prob   = {prob_reach_cloud:.6f}")
        print(f"   - raw delay    = {t_trans_2_raw:.12f} s")
        print(f"   - expected     = {t_trans_2:.12f} s")
    else:
        t_trans_2 = 0.0
        print(f"\n4. Trans E->C (None):\n   - expected = 0 s")

    # E. Cloud computation（只有可能到达 cloud 才有意义；函数内部已按 P_i 做期望）
    if 0 <= c1 < m and c1 != -1 and prob_reach_cloud > 0:
        t_cloud = _compute_cloud_computation_delay(cut_tuple, P_i, paras.C, f_c)
        print(f"\n5. Cloud Comp Layers [{c1}, m):")
        print(f"   - used f_c = {f_c:.6f} GHz")
        print(f"   - result   = {t_cloud:.12f} s")
    else:
        t_cloud = 0.0
        print(f"\n5. Cloud Comp (None):\n   - result   = 0 s")

    # F. Sum（手动 sum 与 compute_total_latency 对齐）
    manual_total = t_local + t_trans_1 + t_edge + t_trans_2 + t_cloud
    print("\n" + "-" * 60)
    print(f"Manual Total Sum: \t\t\t{manual_total:.6f} s")

    # 6. 调用主函数 compute_total_latency 进行验证
    print("\n>>> 调用 compute_total_latency 函数...")
    T_vec = compute_total_latency(X, P, F_e, F_c, paras)
    func_total = float(T_vec[0])
    print(f"Function Result: \t\t\t{func_total:.6f} s")

    # 7. 最终校验
    diff = abs(manual_total - func_total)
    if diff < 1e-9:
        print("\n✅ 测试通过：手动拆解计算结果与主函数结果一致。")
    else:
        print(f"\n❌ 测试失败：误差为 {diff}。")
