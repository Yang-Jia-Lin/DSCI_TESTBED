"""
根据 X 计算早退点
Src/Objective/compute_exit_points.py
"""

import numpy as np


# one user exit points
def _compute_one_exit_points(X_i):
    ones = np.where(X_i == 1)[0]

    if len(ones) == 0:
        L = len(X_i)
        return L, L

    # 第一个分割点
    p_i1 = ones[0]
    # 第二个分割点（如果有的话）
    tail = X_i[p_i1 + 1 :]
    next_ones = np.where(tail == 1)[0]
    if len(next_ones) > 0:
        p_i2 = next_ones[0] + p_i1 + 1
    else:
        p_i2 = -1
    return p_i1, p_i2


# all users exit points
def compute_exit_points(X, paras):
    n = X.shape[0]
    cut_points = np.zeros((n, 2), dtype=int)
    for i in range(n):
        cut_points[i] = _compute_one_exit_points(X[i])
    return cut_points


# --------------------------------------------------------
# 原始来的错误函数 (为了复现问题)
# --------------------------------------------------------
def _original_compute_one_exit_points(X_i):
    ones = np.where(X_i == 1)[0]

    if len(ones) == 0:
        L = len(X_i)
        return L - 1, L - 1

    p_i1 = ones[0]
    tail = X_i[p_i1 + 1 :]
    next_ones = np.where(tail == 1)[0]

    if len(next_ones) > 0:
        # 注意：这里原代码逻辑是相对索引找回绝对索引
        p_i2 = next_ones[0] + p_i1 + 1
    else:
        p_i2 = -1  # <--- 问题就在这里

    return p_i1, p_i2


# --------------------------------------------------------
# 辅助打印函数：将切分点翻译为各端负责的范围
# 基于 compute_latency 中的 range 逻辑推断
# --------------------------------------------------------
def interpret_ranges(cut0, cut1, total_layers):
    # Latency 代码逻辑回顾:
    # Local: 0 ... cut0
    # Edge:  cut0 ... cut1 (python range excludes cut1, so effectively cut0...cut1-1)
    # Cloud: cut1 ... total_layers

    # 修正：Local通常是 [0, cut0] (包含cut0层)，因为切分意味着在第j层"后"传输
    # 但根据你的 latency 代码: range(cut0 + 1) -> 0..cut0. 正确。

    local_range = f"[0, {cut0}]"

    if cut1 == -1:
        # 模拟 range(cut0, -1) -> Empty
        edge_range = "EMPTY (Error!)"
        # Cloud range 逻辑不明，假设是Empty
        cloud_range = "Unknown"
    else:
        if cut0 >= cut1:
            edge_range = "EMPTY"
        else:
            edge_range = f"[{cut0 + 1}, {cut1}]"  # 实际上是 computation 层的索引

        if cut1 >= total_layers:
            cloud_range = "EMPTY"
        else:
            cloud_range = f"[{cut1 + 1}, {total_layers - 1}]"

    return local_range, edge_range, cloud_range


# --------------------------------------------------------
# 测试主逻辑
# --------------------------------------------------------
if __name__ == "__main__":
    m = 6  # 假设模型有 6 层 (0-5)

    # --------------------------------------------------------
    # 错误原因分析
    # --------------------------------------------------------
    print(
        f"{'Case Type':<15} | {'X Vector':<20} | {'Result (c0, c1)':<18} | {'Local':<10} {'Edge':<15} {'Cloud':<10}"
    )
    print("-" * 95)

    # Case 1: 0次切分 (Sum=0) -> 全在本地
    x_case0 = np.zeros(m, dtype=int)
    c0, c1 = _original_compute_one_exit_points(x_case0)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    print(
        f"{'0 Cut (Local)':<15} | {str(x_case0):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10}"
    )

    # Case 2: 2次切分 (Sum=2) -> 端-边-云 (标准情况)
    x_case2 = np.zeros(m, dtype=int)
    x_case2[1] = 1  # 切分点1
    x_case2[4] = 1  # 切分点2
    c0, c1 = _original_compute_one_exit_points(x_case2)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    print(
        f"{'2 Cuts (L-E-C)':<15} | {str(x_case2):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10}"
    )

    # Case 3: 1次切分 (Sum=1) -> 端-边 (User -> Edge -> End)
    # 【预期问题】这里应该显示 Edge 负责剩余所有层，但原来的 -1 会导致 Empty
    x_case1 = np.zeros(m, dtype=int)
    x_case1[2] = 1
    c0, c1 = _original_compute_one_exit_points(x_case1)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    status = "❌ BUG" if c1 == -1 else "✅ OK"
    print(
        f"{'1 Cut (L-E)':<15} | {str(x_case1):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10} <--- {status}"
    )

    print("-" * 95)
    print("分析 Case 3 (1 Cut):")
    if c1 == -1:
        print(
            "  当前返回 (-1) 会导致 compute_latency 中的 range(cut0, cut1) 变为空 range(2, -1)。"
        )
        print("  结果：边缘服务器明明该工作，但计算时延变成了 0。")

    # --------------------------------------------------------
    # 正确代码简单测试
    # --------------------------------------------------------
    print("\n" + "=" * 40 + "修改后" + "=" * 40)
    print(
        f"{'Case Type':<15} | {'X Vector':<20} | {'Result (c0, c1)':<18} | {'Local':<10} {'Edge':<15} {'Cloud':<10}"
    )
    print("-" * 95)

    # Case 1: 0次切分 (Sum=0) -> 全在本地
    x_case0 = np.zeros(m, dtype=int)
    c0, c1 = _compute_one_exit_points(x_case0)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    print(
        f"{'0 Cut (Local)':<15} | {str(x_case0):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10}"
    )

    # Case 2: 2次切分 (Sum=2) -> 端-边-云 (标准情况)
    x_case2 = np.zeros(m, dtype=int)
    x_case2[1] = 1  # 切分点1
    x_case2[4] = 1  # 切分点2
    c0, c1 = _compute_one_exit_points(x_case2)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    print(
        f"{'2 Cuts (L-E-C)':<15} | {str(x_case2):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10}"
    )

    # Case 3: 1次切分 (Sum=1) -> 端-边 (User -> Edge -> End)
    # 【预期问题】这里应该显示 Edge 负责剩余所有层，但原来的 -1 会导致 Empty
    x_case1 = np.zeros(m, dtype=int)
    x_case1[2] = 1
    c0, c1 = _compute_one_exit_points(x_case1)
    loc, edg, cld = interpret_ranges(c0, c1, m)
    status = "❌ BUG" if c1 == -1 else "✅ OK"
    print(
        f"{'1 Cut (L-E)':<15} | {str(x_case1):<20} | {f'({c0}, {c1})':<18} | {loc:<10} {edg:<15} {cld:<10} <--- {status}"
    )

    print("-" * 95)

    # --------------------------------------------------------
    # 正确代码全面测试
    # --------------------------------------------------------
    print("\n" + "=" * 40 + "修改后全面测试" + "=" * 40)
    test_cases = [
        # --- 基础回顾 ---
        ("0. 全本地 (无切分)", np.array([0, 0, 0, 0, 0, 0])),
        # --- 1次切分 (Sum=1) 的边界测试 ---
        (
            "1. 极早切分 (第0层后)",
            np.array([1, 0, 0, 0, 0, 0]),
        ),  # Local做完Layer0就扔给Edge
        ("2. 中间切分", np.array([0, 0, 1, 0, 0, 0])),
        ("3. 极晚切分 (倒数第2层)", np.array([0, 0, 0, 0, 1, 0])),
        (
            "4. 最后切分 (跑完全部)",
            np.array([0, 0, 0, 0, 0, 1]),
        ),  # Local做完所有，传给Edge结果?
        # --- 2次切分 (Sum=2) 的边界测试 ---
        (
            "5. 连续切分 (0, 1)",
            np.array([1, 1, 0, 0, 0, 0]),
        ),  # Local做0, Edge做1, Cloud做剩下
        ("6. 连续切分 (中间)", np.array([0, 0, 1, 1, 0, 0])),
        (
            "7. 极限首尾 (0, 5)",
            np.array([1, 0, 0, 0, 0, 1]),
        ),  # Local做0, Cloud做无? Edge包圆?
        ("8. 间隔切分", np.array([0, 1, 0, 0, 1, 0])),
    ]

    print(
        f"{'ID':<3} | {'Description':<20} | {'X Vector':<20} | {'Result (c0, c1)':<15} | {'Logic Check'}"
    )
    print("-" * 90)

    for i, (desc, x_vec) in enumerate(test_cases):
        c0, c1 = _compute_one_exit_points(x_vec)

        # 简单逻辑解读
        if c0 == c1 == m - 1:
            check = "All Local"
        elif c1 == m:
            check = f"Local[0..{c0}] -> Edge[Rest]"
        else:
            check = f"Local[0..{c0}] -> Edge -> Cloud[{c1 + 1}..]"

        # 格式化向量显示
        vec_str = str(x_vec).replace("\n", "")

        print(f"{i:<3} | {desc:<20} | {vec_str:<20} | {f'({c0}, {c1})':<15} | {check}")
