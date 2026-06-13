"""
Src/Optimizer/BF/alg_BF.py
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from Src.Phase2_Scheduler.Objective.compute_accuracy import compute_expected_accuracy
from Src.Phase2_Scheduler.Objective.compute_exit_points import compute_exit_points
from Src.Phase2_Scheduler.Objective.compute_latency import (
    compute_total_latency,
    compute_user_latency,
)
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.paras import Paras


def _generate_all_valid_X_rows(
    m: int, boundary_ids: list[int] | None = None
) -> np.ndarray:
    rows = []
    candidates = list(boundary_ids or range(m))
    for a in candidates:
        for b in candidates:
            if not 0 <= a < b < m:
                continue
            x = np.zeros(m, dtype=np.int8)
            x[a] = 1
            x[b] = 1
            rows.append(x)
    return np.stack(rows, axis=0)


def _tau_grid(step: float = 0.01) -> np.ndarray:
    k = int(round(1.0 / step))
    return np.array([round(i * step, 2) for i in range(k + 1)], dtype=np.float64)


def _build_y_row(
    m: int, exit_layers: Tuple[int, int], t1: float, t2: float
) -> np.ndarray:
    y = np.ones(m, dtype=np.float64)
    e1, e2 = exit_layers
    y[e1] = t1
    y[e2] = t2
    return y


def _precompute_cut_points_for_X_candidates(
    X_candidates: np.ndarray, paras
) -> np.ndarray:
    """
    对每个 x_row 用你原始的 compute_exit_points() 精确求 cut_points。
    返回 shape=(K,2) 的 int 数组。
    """
    K, m = X_candidates.shape
    cuts = np.zeros((K, 2), dtype=np.int64)
    for k in range(K):
        X1 = X_candidates[k].reshape(1, m).astype(float)
        cp = compute_exit_points(X1, paras)  # shape (1,2)
        cuts[k, 0] = int(cp[0][0])
        cuts[k, 1] = int(cp[0][1])
    return cuts


def _compute_P_acc_for_yrow(yrow: np.ndarray, paras) -> Tuple[np.ndarray, float]:
    """
    严格调用你现有 compute_layer_exit_probs / compute_expected_accuracy。
    为了兼容 paras.n 的实现细节，构造 YN = tile 到 (paras.n,m)，然后取第 0 行。
    """
    m = paras.m
    YN = np.tile(yrow.reshape(1, m), (paras.n, 1))
    P = compute_layer_exit_probs(YN, paras)
    acc_vec = compute_expected_accuracy(YN, P, paras)
    return np.asarray(P[0], dtype=np.float64), float(acc_vec[0])


def _precompute_threshold_cache(
    paras, step: float = 0.01
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    缓存 (i1,i2) -> {"P_row": (m,), "acc": float}
    这是“严格等价缓存”，不包含任何 latency 近似/分解。
    """
    m = paras.m
    exit_layers = tuple(paras.E)
    assert len(exit_layers) == 2, "BF assumes exactly 2 early-exit layers."

    grid = _tau_grid(step)
    cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for i1, t1 in enumerate(grid):
        for i2, t2 in enumerate(grid):
            yrow = _build_y_row(m, exit_layers, t1, t2)
            P_row, acc_u = _compute_P_acc_for_yrow(yrow, paras)
            cache[(i1, i2)] = {"P_row": P_row, "acc": acc_u}

    return cache


def _init_fe_fc(
    paras, rng: np.random.Generator, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    初始化可行 F_e, F_c：非负、sum==budget（严格），避免全 0。
    """
    n = paras.n
    if getattr(paras, "resource_mode", None) == "fixed_worker_pool":
        return np.zeros((n, 1)), np.zeros((n, 1))
    fe = rng.random(n).astype(np.float64) + eps
    fc = rng.random(n).astype(np.float64) + eps

    fe = fe / float(np.sum(fe)) * float(paras.f_e_max)
    fc = fc / float(np.sum(fc)) * float(paras.f_c_max)

    return fe.reshape(n, 1), fc.reshape(n, 1)


def _objective_from_sums(
    alpha: float, beta: float, acc_sum: float, lat_sum: float
) -> float:
    return float(alpha * acc_sum - beta * lat_sum)


def _optimize_F_by_pairwise_swaps(
    X_idx: np.ndarray,
    tau_idx: List[Tuple[int, int]],
    acc_vec: np.ndarray,
    lat_vec: np.ndarray,
    F_e: np.ndarray,
    F_c: np.ndarray,
    cuts_by_k: np.ndarray,
    cache: Dict[Tuple[int, int], Dict[str, Any]],
    paras,
    iters: int = 200,
    deltas_frac: Tuple[float, ...] = (0.05, 0.02, 0.01),
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    严格 objective 增量的资源交换优化：
    - 每次只交换两个用户 i,j 的资源（edge 或 cloud）
    - 只重算 i,j 的 latency（用 compute_user_latency 原公式）
    - acc 不受 F 影响，所以不变
    """
    if getattr(paras, "resource_mode", None) == "fixed_worker_pool":
        return F_e, F_c, lat_vec, _objective_from_sums(
            float(paras.alpha),
            float(paras.beta),
            float(np.sum(acc_vec)),
            float(np.sum(lat_vec)),
        )
    if rng is None:
        rng = np.random.default_rng()

    n = paras.n
    alpha = float(paras.alpha)
    beta = float(paras.beta)

    fe = F_e.reshape(-1).astype(np.float64).copy()
    fc = F_c.reshape(-1).astype(np.float64).copy()

    acc_sum = float(np.sum(acc_vec))
    lat_sum = float(np.sum(lat_vec))
    obj = _objective_from_sums(alpha, beta, acc_sum, lat_sum)

    def _recompute_latency_user(u: int, fe_u: float, fc_u: float) -> float:
        k = int(X_idx[u])
        cut0, cut1 = int(cuts_by_k[k, 0]), int(cuts_by_k[k, 1])

        i1, i2 = tau_idx[u]
        P_row = cache[(i1, i2)]["P_row"]

        return compute_user_latency(u, cut0, cut1, P_row, fe_u, fc_u, paras)

    # --- main loop ---
    for _ in range(iters):
        improved = False

        # (A) optimize edge resources
        for frac in deltas_frac:
            delta = frac * float(paras.f_e_max)
            for __ in range(2 * n):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue

                d = min(delta, fe[i])
                if d <= 0:
                    continue

                fe_i_new = fe[i] - d
                fe_j_new = fe[j] + d

                lat_i_new = _recompute_latency_user(i, fe_i_new, fc[i])
                lat_j_new = _recompute_latency_user(j, fe_j_new, fc[j])

                if not (np.isfinite(lat_i_new) and np.isfinite(lat_j_new)):
                    continue

                lat_sum_new = (
                    lat_sum
                    - float(lat_vec[i])
                    - float(lat_vec[j])
                    + lat_i_new
                    + lat_j_new
                )
                obj_new = _objective_from_sums(alpha, beta, acc_sum, lat_sum_new)

                if np.isfinite(obj_new) and obj_new > obj + tol:
                    # accept
                    fe[i] = fe_i_new
                    fe[j] = fe_j_new
                    lat_sum = lat_sum_new
                    obj = obj_new
                    lat_vec[i] = lat_i_new
                    lat_vec[j] = lat_j_new
                    improved = True

        # (B) optimize cloud resources
        for frac in deltas_frac:
            delta = frac * float(paras.f_c_max)
            for __ in range(2 * n):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue

                d = min(delta, fc[i])
                if d <= 0:
                    continue

                fc_i_new = fc[i] - d
                fc_j_new = fc[j] + d

                lat_i_new = _recompute_latency_user(i, fe[i], fc_i_new)
                lat_j_new = _recompute_latency_user(j, fe[j], fc_j_new)

                if not (np.isfinite(lat_i_new) and np.isfinite(lat_j_new)):
                    continue

                lat_sum_new = (
                    lat_sum
                    - float(lat_vec[i])
                    - float(lat_vec[j])
                    + lat_i_new
                    + lat_j_new
                )
                obj_new = _objective_from_sums(alpha, beta, acc_sum, lat_sum_new)

                if np.isfinite(obj_new) and obj_new > obj + tol:
                    # accept
                    fc[i] = fc_i_new
                    fc[j] = fc_j_new
                    lat_sum = lat_sum_new
                    obj = obj_new
                    lat_vec[i] = lat_i_new
                    lat_vec[j] = lat_j_new
                    improved = True

        if not improved:
            break

    F_e_new = fe.reshape(n, 1)
    F_c_new = fc.reshape(n, 1)
    return F_e_new, F_c_new, lat_vec, obj


def optimize_BF(
    paras: Paras | None = None,
    max_iter: int = 10,
    restarts: int = 3,
    threshold_step: float = 0.01,
    tol: float = 1e-6,
    verbose: bool = True,
    F_opt_iters: int = 150,
):
    """
    - 阈值部分只缓存 P_row / acc
    - latency 逐用户严格用 compute_user_latency
    - 用户更新只重算该用户 acc_u/lat_u
    - F_e/F_c 用 pairwise swap 严格增量更新（每次只重算两个用户 latency）
    """
    assert paras is not None, "paras cannot be None for optimize_BF"

    n, m = paras.n, paras.m
    exit_layers = tuple(paras.E)
    assert len(exit_layers) == 2, (
        "This BF implementation assumes exactly 2 early-exit layers."
    )
    e1, e2 = exit_layers

    alpha = float(paras.alpha)
    beta = float(paras.beta)

    rng = np.random.default_rng()

    # candidates
    X_candidates = _generate_all_valid_X_rows(m, paras.partition_boundary_ids)
    K = X_candidates.shape[0]
    cuts_by_k = _precompute_cut_points_for_X_candidates(X_candidates, paras)

    # threshold cache
    cache = _precompute_threshold_cache(paras, step=threshold_step)
    grid = _tau_grid(threshold_step)
    G = len(grid)

    best_overall_val = -float("inf")
    best_overall_sol = None
    best_overall_hist: List[float] = []

    for r in range(restarts):
        # ---- init X (by index), Y (by tau indices), F ----
        X_idx = rng.integers(low=0, high=K, size=n)  # store candidate index per user
        tau_raw = rng.integers(low=0, high=G, size=(n, 2))
        tau_idx: List[Tuple[int, int]] = [
            (int(tau_raw[i, 0]), int(tau_raw[i, 1])) for i in range(n)
        ]

        F_e, F_c = _init_fe_fc(paras, rng)

        # build Y matrix
        Y = np.ones((n, m), dtype=np.float64)
        for i in range(n):
            i1, i2 = tau_idx[i]
            Y[i, e1] = grid[i1]
            Y[i, e2] = grid[i2]

        # initialize acc_vec, lat_vec exactly
        acc_vec = np.zeros(n, dtype=np.float64)
        lat_vec = np.zeros(n, dtype=np.float64)

        for i in range(n):
            i1, i2 = tau_idx[i]
            acc_vec[i] = float(cache[(i1, i2)]["acc"])

            k = int(X_idx[i])
            cut0, cut1 = int(cuts_by_k[k, 0]), int(cuts_by_k[k, 1])
            P_row = cache[(i1, i2)]["P_row"]

            lat_vec[i] = compute_user_latency(
                i,
                cut0,
                cut1,
                P_row,
                float(F_e.reshape(-1)[i]),
                float(F_c.reshape(-1)[i]),
                paras,
            )

        acc_sum = float(np.sum(acc_vec))
        lat_sum = float(np.sum(lat_vec))
        val = _objective_from_sums(alpha, beta, acc_sum, lat_sum)
        fixed_workers = paras.resource_mode == "fixed_worker_pool"
        if fixed_workers:
            X_current = np.stack(
                [X_candidates[int(index)] for index in X_idx], axis=0
            ).astype(np.float64)
            P_current = compute_layer_exit_probs(Y, paras)
            lat_vec = compute_total_latency(X_current, P_current, F_e, F_c, paras)
            lat_sum = float(np.sum(lat_vec))
            val = float(objective(X_current, Y, F_e, F_c, paras))

        if not np.isfinite(val):
            if verbose:
                print(f"[BF-INC] restart={r + 1}/{restarts} init_obj not finite, skip.")
            continue

        hist = [val]
        if verbose:
            print(f"[BF-INC] restart={r + 1}/{restarts} init_obj={val:.6f}")

        # ---- BCD iterations ----
        for it in range(max_iter):
            improved_any = False
            order = rng.permutation(n)

            # (A) user-wise brute force update of (X[u], tau[u]) with F fixed
            for u in order:
                old_k = int(X_idx[u])
                old_tau = tau_idx[u]
                old_acc = float(acc_vec[u])
                old_lat = float(lat_vec[u])

                best_local_val = val
                best_k = old_k
                best_tau = old_tau
                best_acc = old_acc
                best_lat = old_lat

                fe_u = float(F_e.reshape(-1)[u])
                fc_u = float(F_c.reshape(-1)[u])

                # enumerate candidates
                for k in range(K):
                    cut0, cut1 = int(cuts_by_k[k, 0]), int(cuts_by_k[k, 1])

                    for i1 in range(G):
                        for i2 in range(G):
                            acc_u = float(cache[(i1, i2)]["acc"])
                            P_row = cache[(i1, i2)]["P_row"]

                            if fixed_workers:
                                X_tmp = np.stack(
                                    [X_candidates[int(index)] for index in X_idx],
                                    axis=0,
                                ).astype(np.float64)
                                X_tmp[u] = X_candidates[k]
                                Y_tmp = Y.copy()
                                Y_tmp[u, :] = 1.0
                                Y_tmp[u, e1] = grid[i1]
                                Y_tmp[u, e2] = grid[i2]
                                val_tmp = float(objective(X_tmp, Y_tmp, F_e, F_c, paras))
                                lat_u = old_lat
                            else:
                                lat_u = compute_user_latency(
                                    u, cut0, cut1, P_row, fe_u, fc_u, paras
                                )
                                if not np.isfinite(lat_u):
                                    continue
                                acc_sum_tmp = acc_sum - old_acc + acc_u
                                lat_sum_tmp = lat_sum - old_lat + lat_u
                                val_tmp = _objective_from_sums(
                                    alpha, beta, acc_sum_tmp, lat_sum_tmp
                                )

                            if np.isfinite(val_tmp) and val_tmp > best_local_val + tol:
                                best_local_val = val_tmp
                                best_k = k
                                best_tau = (i1, i2)
                                best_acc = acc_u
                                best_lat = lat_u

                # apply if improved
                if best_local_val > val + tol:
                    improved_any = True

                    X_idx[u] = best_k
                    tau_idx[u] = best_tau

                    # update Y row
                    i1, i2 = best_tau
                    Y[u, :] = 1.0
                    Y[u, e1] = grid[i1]
                    Y[u, e2] = grid[i2]

                    # update vectors and sums
                    acc_sum = acc_sum - old_acc + best_acc
                    lat_sum = lat_sum - old_lat + best_lat

                    acc_vec[u] = best_acc
                    if fixed_workers:
                        X_current = np.stack(
                            [X_candidates[int(index)] for index in X_idx], axis=0
                        ).astype(np.float64)
                        P_current = compute_layer_exit_probs(Y, paras)
                        lat_vec = compute_total_latency(
                            X_current, P_current, F_e, F_c, paras
                        )
                        lat_sum = float(np.sum(lat_vec))
                        val = float(objective(X_current, Y, F_e, F_c, paras))
                    else:
                        lat_vec[u] = best_lat
                        val = best_local_val
                else:
                    # keep old
                    pass

            # (B) optimize F by strict pairwise swaps (only recompute two users latency each move)
            F_e_new, F_c_new, lat_vec_new, val_new = _optimize_F_by_pairwise_swaps(
                X_idx=X_idx,
                tau_idx=tau_idx,
                acc_vec=acc_vec,
                lat_vec=lat_vec.copy(),
                F_e=F_e,
                F_c=F_c,
                cuts_by_k=cuts_by_k,
                cache=cache,
                paras=paras,
                iters=F_opt_iters,
                tol=tol,
                rng=rng,
            )

            if np.isfinite(val_new) and val_new > val + tol:
                improved_any = True
                F_e, F_c = F_e_new, F_c_new
                lat_vec = lat_vec_new
                lat_sum = float(np.sum(lat_vec))
                val = val_new

            hist.append(val)
            if verbose:
                print(
                    f"[BF-INC] restart={r + 1} iter={it + 1}/{max_iter} obj={val:.6f}"
                )

            if not improved_any:
                if verbose:
                    print(f"[BF-INC] restart={r + 1} converged (no improvement).")
                break

        # ---- build final X matrix for output ----
        X = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            X[i, :] = X_candidates[int(X_idx[i])].astype(np.float64)

        # final objective (incremental already exact; this is just a sanity check style)
        final_val = val
        if verbose:
            print(f"[BF-INC] restart={r + 1} final_obj={final_val:.6f}")

        if np.isfinite(final_val) and final_val > best_overall_val:
            best_overall_val = final_val
            best_overall_sol = (X.copy(), Y.copy(), F_e.copy(), F_c.copy())
            best_overall_hist = hist

    return best_overall_val, best_overall_sol, best_overall_hist
