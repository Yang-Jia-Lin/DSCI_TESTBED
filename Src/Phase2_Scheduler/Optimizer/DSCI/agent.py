"""
Src/Optimizer/DSCI/agent.py

改动：
1) Episode 包含 n 个 steps（每步一个用户）
2) X: categorical index over explicit deployment (k1,k2) pairs（来自 network.x_pairs）
3) Y: Beta 分布，仅对早退层集合 |E| 输出/采样（无需硬裁剪）
4) 数值稳定：adv norm、grad clip、严格 on-policy，移除 TopK/off-policy 等机制
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.objective import get_lat_and_acc, objective
from Src.Phase2_Scheduler.Optimizer.DSCI.buffer import RolloutBuffer
from Src.Phase2_Scheduler.Optimizer.DSCI.networks import ActorCritic
from Src.Phase2_Scheduler.Utils.parsing_data import split_points_matrix
from Src.Shared.Partitioning.split_actions import encode_split_row


# ---------- 状态构造（紧凑 Markov） ----------
def _build_state(
    i: int,
    n: int,
    prev_obj: float,
    F_e: np.ndarray,
    F_c: np.ndarray,
    f_e_max: float,
    f_c_max: float,
    paras,
    obj_scale: float = 1000.0,
) -> torch.Tensor:
    """
    state = base PPO features plus fixed-worker user/topology features when available.
    """
    i_norm = float(i) / float(max(n, 1))
    remaining_norm = float(n - i) / float(max(n, 1))
    prev_obj_squashed = float(np.tanh(prev_obj / obj_scale))

    # F_e, F_c are (n,1) in your code
    fe_i = float(F_e[i, 0]) if F_e.ndim == 2 else float(F_e[i])
    fc_i = float(F_c[i, 0]) if F_c.ndim == 2 else float(F_c[i])
    fe_i_norm = fe_i / float(max(f_e_max, 1e-12))
    fc_i_norm = fc_i / float(max(f_c_max, 1e-12))

    features = [i_norm, remaining_norm, prev_obj_squashed, fe_i_norm, fc_i_norm]

    if getattr(paras, "resource_mode", None) == "fixed_worker_pool":
        B_u = getattr(paras, "B_u", None)
        if B_u is not None:
            bw_arr = np.asarray(B_u, dtype=np.float64).reshape(-1)
            bw_max = float(max(float(bw_arr.max()), 1e-12))
            features.append(float(bw_arr[i]) / bw_max)
        else:
            features.append(0.0)

        seg_u = getattr(paras, "segment_latency_u", None)
        if seg_u is not None:
            seg_u = np.asarray(seg_u, dtype=np.float64)
            row = seg_u[i]
            all_max = float(max(float(seg_u.max()), 1e-12))
            sum_scale = float(max(float(seg_u.sum(axis=1).max()), 1e-12))
            features.append(float(row.sum()) / sum_scale)
            features.append(float(row.max()) / all_max)
        else:
            features.extend([0.0, 0.0])

        seg_e = getattr(paras, "segment_latency_e", None)
        if seg_e is not None:
            seg_e = np.asarray(seg_e, dtype=np.float64)
            scale = float(max(float(seg_e.sum()), 1e-12))
            features.append(float(seg_e.sum()) / scale)
        else:
            features.append(0.0)

        seg_c = getattr(paras, "segment_latency_c", None)
        if seg_c is not None:
            seg_c = np.asarray(seg_c, dtype=np.float64)
            scale = float(max(float(seg_c.sum()), 1e-12))
            features.append(float(seg_c.sum()) / scale)
        else:
            features.append(0.0)

        b_c = float(getattr(paras, "b_c", 0.0))
        if B_u is not None:
            bw_max = float(max(float(np.asarray(B_u).max()), b_c, 1e-12))
            features.append(b_c / bw_max)
        else:
            features.append(0.0)

        overhead_d2e = float(getattr(paras, "protocol_overhead_d2e_s", 0.0))
        overhead_e2c = float(getattr(paras, "protocol_overhead_e2c_s", 0.0))
        oh_max = float(max(overhead_d2e, overhead_e2c, 1e-12))
        features.append(overhead_d2e / oh_max if oh_max > 1e-12 else 0.0)
        features.append(overhead_e2c / oh_max if oh_max > 1e-12 else 0.0)

        ew = float(getattr(paras, "edge_worker_count", 1))
        cw = float(getattr(paras, "cloud_worker_count", 1))
        wmax = float(max(ew, cw, 1.0))
        features.append(ew / wmax)
        features.append(cw / wmax)

    s = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return s


# ---------- 初始化一个可行解（给未决策用户用作 baseline） ----------
def _init_feasible_XY(paras, decision_spec=None):
    """
    生成一个“默认可行”的 X, Y，用作 episode 初始基线和未决策用户的占位。
    - X: 每行一个合法部署 pair (k1,k2)，这里用 (m//3, 2m//3)
    - Y: 全 1，早退层也先设为 1（表示阈值高，倾向不早退）
    """
    n, m = paras.n, paras.m
    X = np.zeros((n, m), dtype=np.float32)
    final = m - 1
    if decision_spec is not None and decision_spec.split_rule == "fixed":
        fixed_pairs = decision_spec.split_pairs_for(n)
    else:
        allowed = (
            list(decision_spec.allowed_split_pairs)
            if decision_spec is not None and decision_spec.allowed_split_pairs
            else None
        )
        if allowed:
            fixed_pairs = [tuple(allowed[0])] * n
        else:
            k1 = max(0, min(final - 1, final // 3))
            k2 = max(k1 + 1, min(final, (2 * final) // 3))
            fixed_pairs = [(k1, k2)] * n

    for i in range(n):
        X[i] = encode_split_row(*fixed_pairs[i], m, dtype=np.float32)

    Y = np.ones((n, m), dtype=np.float32)
    # 早退层也先设 1（不强制），RL 会学到更优的阈值
    if decision_spec is not None and decision_spec.exit_rule == "fixed":
        Y = decision_spec.threshold_rows_for(paras)
    else:
        for ee in paras.E:
            if 0 <= ee < m:
                Y[:, ee] = 1.0
    return X, Y


def compute_iota_kappa(X, edge_compute_sizes, cloud_compute_sizes, exit_prob):
    """Compute expected edge/cloud work used by the closed-form allocation."""
    n, m = X.shape
    c_e = np.asarray(edge_compute_sizes, dtype=np.float64)
    c_c = np.asarray(cloud_compute_sizes, dtype=np.float64)
    iota = np.zeros(n)
    kappa = np.zeros(n)
    split_pts = split_points_matrix(X)
    final = m - 1
    for i in range(n):
        p1, p2 = split_pts[i]
        for segment_id in range(int(p1), int(p2)):
            reach_prob = float(exit_prob[i, segment_id + 1 :].sum())
            iota[i] += reach_prob * float(c_e[segment_id])
        for segment_id in range(int(p2), final):
            reach_prob = float(exit_prob[i, segment_id + 1 :].sum())
            kappa[i] += reach_prob * float(c_c[segment_id])
    return iota, kappa


def allocate_resources(iota, kappa, f_e_max, f_c_max):
    """计算凸优化后的资源分配"""
    sqrt_i, sqrt_k = np.sqrt(iota + 1e-12), np.sqrt(kappa + 1e-12)
    f_e = f_e_max * sqrt_i / max(sqrt_i.sum(), 1e-12)
    f_c = f_c_max * sqrt_k / max(sqrt_k.sum(), 1e-12)
    return f_e, f_c


class PPOAgent:
    def __init__(self, paras, hyperparams, *, evaluator=None, decision_spec=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paras = paras
        self.hparams = hyperparams
        self.evaluator = evaluator
        self.decision_spec = decision_spec
        self.initial_entropy_coef = hyperparams.get("entropy_coef", 0.01)  # 熵系数衰减
        self.entropy_decay = hyperparams.get("entropy_decay", 0.99)  # 熵系数衰减

        # ---------- 维度 ----------
        self.state_dim = self._compute_state_dim(paras)
        self.action_dim_Y = (
            len(self.paras.E)
            if decision_spec is None or decision_spec.exit_rule == "optimize"
            else 0
        )

        # ---------- 网络 ----------
        policy_net = ActorCritic(
            state_dim=self.state_dim,
            num_layers=self.paras.m,
            action_dim_Y=self.action_dim_Y,
            partition_boundary_ids=self.paras.partition_boundary_ids,
            allowed_split_pairs=(
                list(decision_spec.allowed_split_pairs)
                if decision_spec is not None and decision_spec.allowed_split_pairs
                else (
                    decision_spec.split_pairs_for(paras.n)
                    if decision_spec is not None and decision_spec.split_rule == "fixed"
                    else None
                )
            ),
        ).to(self.device)
        self.policy: ActorCritic = policy_net

        # ---------- 优化 ----------
        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=hyperparams["lr"]
        )

        # ---------- 记录 ----------
        self.best_policy_state_dict = (
            None  # 保存历史最优策略（用于最终 best checkpoint），不做频繁 rollback
        )
        self.logs = []  # 每个 epoch 一个 dict

    def _objective(self, X, Y, F_e, F_c):
        if self.evaluator is not None:
            return self.evaluator.evaluate(X, Y, F_e, F_c)
        return objective(X, Y, F_e, F_c, self.paras)

    @staticmethod
    def _compute_state_dim(paras) -> int:
        dim = 5
        if getattr(paras, "resource_mode", None) == "fixed_worker_pool":
            dim += 10
        return dim

    def _entropy_coef(self, epoch: int) -> float:
        """Entropy schedule; fixed-worker runs need exploration to shut down earlier."""
        coef = self.initial_entropy_coef * (self.entropy_decay**epoch)
        if getattr(self.paras, "resource_mode", None) != "fixed_worker_pool":
            return float(coef)

        fixed_decay = float(
            self.hparams.get("fixed_worker_entropy_decay", 0.94)
        )
        coef = self.initial_entropy_coef * (fixed_decay**epoch)
        stop_epoch = self.hparams.get("fixed_worker_entropy_stop_epoch", 40)
        if stop_epoch is not None and epoch >= int(stop_epoch):
            coef = 0.0
        return float(coef)

    def _y_variance_coef(self, epoch: int) -> float:
        """Late fixed-worker regularizer that makes Beta thresholds sharper."""
        if (
            getattr(self.paras, "resource_mode", None) != "fixed_worker_pool"
            or self.action_dim_Y <= 0
        ):
            return 0.0

        start_epoch = int(self.hparams.get("fixed_worker_y_variance_start_epoch", 20))
        if epoch < start_epoch:
            return 0.0

        max_coef = float(self.hparams.get("fixed_worker_y_variance_coef", 0.05))
        ramp_epochs = max(1, int(self.hparams.get("fixed_worker_y_variance_ramp", 20)))
        progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
        return float(max_coef * progress)

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor):
        """
        Args:
            state: [1, state_dim] on device
        Returns:
            x_idx: LongTensor scalar（categorical index）
            y: Tensor[|E|]（Beta sample in [0,1]）
            logprob: Tensor scalar（logp_X + logp_Y）
            value: Tensor scalar
        """
        logits_X, alpha_Y, beta_Y, value = self.policy(state)

        # X: categorical
        dist_X = torch.distributions.Categorical(logits=logits_X)  # 分类分布
        x_idx = dist_X.sample()  # 从分类分布中按照概率进行随机选择，得到一个索引
        logp_X = dist_X.log_prob(x_idx)  # 随机抽取到这个索引的对数概率
        ent_X = dist_X.entropy()  # shape [1]

        # Y: Beta（|E|=0）
        if self.action_dim_Y > 0:
            dist_Y = torch.distributions.Beta(alpha_Y, beta_Y)  # Beta分布
            y = dist_Y.sample()  # 从Beta分布中按照概率进行随机选择，得到|E|个值
            logp_Y = dist_Y.log_prob(y).sum(
                -1
            )  # 得到这些值的对数概率和（取对数前应该是乘积）
            ent_Y = dist_Y.entropy().sum(-1)  # shape [1]
        else:  # 没有早退层
            y = state.new_zeros((1, 0))
            logp_Y = state.new_zeros((1,))
            ent_Y = state.new_zeros((1,))

        logprob = (logp_X + logp_Y).detach().squeeze(0)  # 将所有对数概率汇总
        value = value.detach().view(-1)[0]  # scalar
        ent_X = ent_X.detach().squeeze(0)  # scalar
        ent_Y = ent_Y.detach().squeeze(0)  # scalar
        return x_idx.view(-1)[0], y.squeeze(0), logprob, value, ent_X, ent_Y

    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        """Greedy action for online inference (argmax X, Beta mean Y)."""
        logits_X, alpha_Y, beta_Y, value = self.policy(state)

        x_idx = logits_X.argmax(dim=-1).view(-1)[0]
        if self.action_dim_Y > 0:
            y = (alpha_Y / (alpha_Y + beta_Y)).squeeze(0)
        else:
            y = state.new_zeros((0,))
        value = value.detach().view(-1)[0]
        return x_idx, y, value

    @torch.no_grad()
    def action_logprob(
        self, state: torch.Tensor, x_idx: torch.Tensor, y_vec: torch.Tensor
    ) -> torch.Tensor:
        """Log-probability of a given (x_idx, y_vec) under the current policy."""
        logits_X, alpha_Y, beta_Y, _value = self.policy(state)
        dist_X = torch.distributions.Categorical(logits=logits_X)
        if not isinstance(x_idx, torch.Tensor):
            x_idx = torch.tensor(int(x_idx), device=state.device, dtype=torch.long)
        x_scalar = x_idx.view(-1)[0].long()
        logp_X = dist_X.log_prob(x_scalar)

        if self.action_dim_Y > 0:
            dist_Y = torch.distributions.Beta(alpha_Y, beta_Y)
            y_in = y_vec.unsqueeze(0) if y_vec.dim() == 1 else y_vec
            logp_Y = dist_Y.log_prob(y_in).sum(-1).view(-1)[0]
        else:
            logp_Y = state.new_zeros(())

        return (logp_X + logp_Y).float()

    def load_policy_state_dict(
        self, state_dict: dict, strict: bool = True
    ) -> "PPOAgent":
        self.policy.load_state_dict(state_dict, strict=strict)
        return self

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> "PPOAgent":
        ckpt = torch.load(Path(path), map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.load_policy_state_dict(ckpt, strict=strict)
        return self

    @staticmethod
    def default_resources(paras) -> tuple[np.ndarray, np.ndarray]:
        """Equal-split edge/cloud compute as in ``train()`` initialization."""
        n = paras.n
        if getattr(paras, "resource_mode", None) == "fixed_worker_pool":
            return (
                np.zeros((n, 1), dtype=np.float32),
                np.zeros((n, 1), dtype=np.float32),
            )
        F_e = np.ones((n, 1), dtype=np.float32) * (paras.f_e_max / n)
        F_c = np.ones((n, 1), dtype=np.float32) * (paras.f_c_max / n)
        return F_e, F_c

    def allocate_resources_for_XY(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        F_e: np.ndarray | None = None,
        F_c: np.ndarray | None = None,
        outer_ema: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Theorem-1 closed-form allocation for a fixed (X, Y)."""
        if getattr(self.paras, "resource_mode", None) == "fixed_worker_pool":
            return self.default_resources(self.paras)
        if F_e is None or F_c is None:
            F_e, F_c = self.default_resources(self.paras)

        exit_prob = compute_layer_exit_probs(Y, self.paras)
        iota, kappa = compute_iota_kappa(
            X, self.paras.C_e, self.paras.C_c, exit_prob
        )
        new_f_e, new_f_c = allocate_resources(
            iota, kappa, self.paras.f_e_max, self.paras.f_c_max
        )
        new_F_e = new_f_e.reshape(self.paras.n, 1).astype(np.float32)
        new_F_c = new_f_c.reshape(self.paras.n, 1).astype(np.float32)

        eta = float(np.clip(outer_ema, 0.0, 1.0))
        F_e = ((1.0 - eta) * F_e + eta * new_F_e).astype(np.float32)
        F_c = ((1.0 - eta) * F_c + eta * new_F_c).astype(np.float32)
        return cast(np.ndarray, F_e), cast(np.ndarray, F_c)

    @torch.no_grad()
    def act_one_episode(
        self,
        F_e: np.ndarray | None = None,
        F_c: np.ndarray | None = None,
        *,
        deterministic: bool = True,
        outer_ema: float = 1.0,
        record_transitions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        One Markov episode (n user steps): sample/greedy actions, then allocate resources.

        Returns:
            X, Y, F_e, F_c, objective value after outer resource update.

        If ``record_transitions`` is True, each user step is appended to ``self.buffer``
        with reward ``0`` (fill real rewards later via ``reward_adapter``).
        """
        if F_e is None or F_c is None:
            F_e, F_c = self.default_resources(self.paras)

        X, Y = _init_feasible_XY(self.paras, self.decision_spec)
        prev_obj = self._objective(X, Y, F_e, F_c)

        for i in range(self.paras.n):
            state = _build_state(
                i=i,
                n=self.paras.n,
                prev_obj=prev_obj,
                F_e=F_e,
                F_c=F_c,
                f_e_max=self.paras.f_e_max,
                f_c_max=self.paras.f_c_max,
                paras=self.paras,
                obj_scale=float(self.hparams.get("obj_scale", 1000.0)),
            ).to(self.device)

            if deterministic:
                x_idx, y_vec_t, value = self.select_action(state)
                logprob = self.action_logprob(state, x_idx, y_vec_t)
            else:
                x_idx, y_vec_t, logprob, value, _ent_X, _ent_Y = self.sample_action(
                    state
                )

            if record_transitions:
                done = float(i == self.paras.n - 1)
                self.buffer.add(
                    state.cpu(),
                    x_idx,
                    y_vec_t.detach().cpu(),
                    logprob.detach().cpu(),
                    value.detach().cpu(),
                    reward=0.0,
                    done=done,
                )

            X_new = X.copy()
            Y_new = Y.copy()
            y_vec_np = y_vec_t.detach().cpu().numpy().astype(np.float32)
            x_idx_int = (
                int(x_idx.item()) if isinstance(x_idx, torch.Tensor) else int(x_idx)
            )
            X_new, Y_new = self._apply_action_to_XY(
                X_new, Y_new, user_i=i, x_idx=x_idx_int, y_vec=y_vec_np
            )
            X, Y = X_new, Y_new
            prev_obj = self._objective(X, Y, F_e, F_c)

        F_e, F_c = self.allocate_resources_for_XY(
            X, Y, F_e=F_e, F_c=F_c, outer_ema=outer_ema
        )
        final_obj = float(self._objective(X, Y, F_e, F_c))
        return X, Y, F_e, F_c, final_obj

    def _apply_action_to_XY(
        self, X: np.ndarray, Y: np.ndarray, user_i: int, x_idx: int, y_vec: np.ndarray
    ):
        """
        将 (x_idx, y_vec) 写入第 user_i 行的 X,Y（其余用户保持原样）
        - x_idx -> (k1,k2) 通过 policy.x_pairs 映射
        - y_vec 写入早退层集合 paras.E 对应的位置
        """
        n, m = self.paras.n, self.paras.m
        assert 0 <= user_i < n

        if self.decision_spec is not None and self.decision_spec.split_rule == "fixed":
            k1, k2 = self.decision_spec.split_pairs_for(n)[user_i]
        else:
            x_pairs = cast(torch.Tensor, self.policy.x_pairs)
            pair = x_pairs[x_idx].detach().cpu().numpy()  # [k1,k2]
            k1, k2 = int(pair[0]), int(pair[1])
        X[user_i, :] = encode_split_row(k1, k2, m, dtype=np.float32)

        # ---- 写 Y：默认全 1，只写早退层阈值 ----
        Y[user_i, :] = 1.0
        if self.decision_spec is not None and self.decision_spec.exit_rule == "fixed":
            Y[user_i, :] = self.decision_spec.threshold_rows_for(self.paras)[user_i]
        elif self.decision_spec is not None and self.decision_spec.exit_rule == "disabled":
            pass
        elif len(self.paras.E) > 0:
            for j, layer_idx in enumerate(self.paras.E):
                if 0 <= layer_idx < m:
                    Y[user_i, layer_idx] = float(y_vec[j])
        return X, Y

    def update_policy(self, epoch: int):
        entropy_coef = self._entropy_coef(epoch)
        y_variance_coef = self._y_variance_coef(epoch)

        advantages, returns = self.buffer.compute_advantages(
            self.hparams["gamma"], self.hparams["lam"]
        )
        if advantages.numel() == 0:
            return

        # advantage 标准化（降方差）
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)  # 关键：避免 T=1 时 NaN
        if torch.isfinite(adv_std) and adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean

        # 在训练前检查 advantages / returns 是否有限：
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            print("[Warning] Non-finite advantages/returns, skip update.")
            return

        data = self.buffer.as_tensors(device=self.device)
        states = data["states"]  # [T, state_dim]
        actions_X = data["actions_X"]  # [T]
        actions_Y = data["actions_Y"]  # [T, |E|]
        old_logprobs = data["logprobs"].detach()  # [T]
        returns = returns.to(self.device)  # [T]
        advantages = advantages.to(self.device)  # [T]

        for _ in range(self.hparams["k_epochs"]):
            logits_X, alpha_Y, beta_Y, values_new = self.policy(
                states
            )  # logits_X [T,num_pairs]
            values_new = values_new.view(-1)  # [T]

            # X 分布
            dist_X = torch.distributions.Categorical(logits=logits_X)
            logp_X = dist_X.log_prob(actions_X)  # [T]
            ent_X = dist_X.entropy()  # [T]

            # Y 分布（Beta）
            if self.action_dim_Y > 0:
                dist_Y = torch.distributions.Beta(alpha_Y, beta_Y)
                logp_Y = dist_Y.log_prob(actions_Y).sum(-1)  # [T]
                ent_Y = dist_Y.entropy().sum(-1)  # [T]
                alpha_beta_sum = alpha_Y + beta_Y
                y_variance = (
                    alpha_Y
                    * beta_Y
                    / (alpha_beta_sum.pow(2) * (alpha_beta_sum + 1.0))
                ).sum(-1)
            else:
                logp_Y = torch.zeros_like(logp_X)
                ent_Y = torch.zeros_like(ent_X)
                y_variance = torch.zeros_like(ent_X)

            new_logprob = logp_X + logp_Y  # [T]
            if getattr(self.paras, "resource_mode", None) == "fixed_worker_pool":
                entropy_bonus = ent_X
            else:
                entropy_bonus = ent_X + ent_Y

            # DSCI ratio
            ratio = torch.exp(new_logprob - old_logprobs)  # [T]
            # 轻微 clamp 防止极端爆炸
            ratio = torch.clamp(ratio, 0.0, 10.0)

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio, 1 - self.hparams["eps_clip"], 1 + self.hparams["eps_clip"]
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new, returns)

            total_loss = (
                policy_loss
                + 0.5 * value_loss
                - entropy_coef * entropy_bonus.mean()
                + y_variance_coef * y_variance.mean()
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def _deterministic_polish(self, ppo_sol):
        """对 fixed_worker_pool 逐用户枚举最优 (split, threshold)。"""
        X_ppo, Y_ppo, F_e, F_c = ppo_sol
        n, m = self.paras.n, self.paras.m
        pairs = self.policy.x_pairs.detach().cpu().numpy()

        X_best = X_ppo.copy()
        Y_best = Y_ppo.copy()

        coarse_grid = list(range(0, 101, 5))

        for i in range(n):
            best_user_obj = -np.inf
            best_k1k2 = None
            best_thresholds = None

            for pair_idx in range(len(pairs)):
                k1, k2 = int(pairs[pair_idx, 0]), int(pairs[pair_idx, 1])
                X_try = X_best.copy()
                X_try[i] = encode_split_row(k1, k2, m, dtype=np.float32)

                if len(self.paras.E) == 0:
                    Y_try = Y_best.copy()
                    Y_try[i, :] = 1.0
                    obj = self._objective(X_try, Y_try, F_e, F_c)
                    if np.isfinite(obj) and obj > best_user_obj:
                        best_user_obj = obj
                        best_k1k2 = (k1, k2)
                        best_thresholds = []
                    continue

                if len(self.paras.E) == 1:
                    for t in coarse_grid:
                        Y_try = Y_best.copy()
                        Y_try[i, :] = 1.0
                        Y_try[i, self.paras.E[0]] = t / 100.0
                        obj = self._objective(X_try, Y_try, F_e, F_c)
                        if np.isfinite(obj) and obj > best_user_obj:
                            best_user_obj = obj
                            best_k1k2 = (k1, k2)
                            best_thresholds = [t / 100.0]
                elif len(self.paras.E) == 2:
                    for t1 in coarse_grid:
                        for t2 in coarse_grid:
                            Y_try = Y_best.copy()
                            Y_try[i, :] = 1.0
                            Y_try[i, self.paras.E[0]] = t1 / 100.0
                            Y_try[i, self.paras.E[1]] = t2 / 100.0
                            obj = self._objective(X_try, Y_try, F_e, F_c)
                            if np.isfinite(obj) and obj > best_user_obj:
                                best_user_obj = obj
                                best_k1k2 = (k1, k2)
                                best_thresholds = [t1 / 100.0, t2 / 100.0]
                else:
                    for t1 in coarse_grid:
                        for t2 in coarse_grid:
                            Y_try = Y_best.copy()
                            Y_try[i, :] = 1.0
                            Y_try[i, self.paras.E[0]] = t1 / 100.0
                            Y_try[i, self.paras.E[1]] = t2 / 100.0
                            for j in range(2, len(self.paras.E)):
                                Y_try[i, self.paras.E[j]] = Y_ppo[
                                    i, self.paras.E[j]
                                ]
                            obj = self._objective(X_try, Y_try, F_e, F_c)
                            if np.isfinite(obj) and obj > best_user_obj:
                                best_user_obj = obj
                                best_k1k2 = (k1, k2)
                                best_thresholds = [
                                    t1 / 100.0,
                                    t2 / 100.0,
                                    *[
                                        float(Y_ppo[i, self.paras.E[j]])
                                        for j in range(2, len(self.paras.E))
                                    ],
                                ]

            if (
                best_k1k2 is not None
                and best_thresholds is not None
                and len(self.paras.E) > 0
            ):
                k1, k2 = best_k1k2
                X_try = X_best.copy()
                X_try[i] = encode_split_row(k1, k2, m, dtype=np.float32)

                coarse_t = [
                    int(round(t * 100))
                    for t in best_thresholds[: min(2, len(self.paras.E))]
                ]
                fine_ranges = [
                    range(max(0, ct - 10), min(101, ct + 11)) for ct in coarse_t
                ]

                if len(self.paras.E) == 1:
                    for t in fine_ranges[0]:
                        Y_try = Y_best.copy()
                        Y_try[i, :] = 1.0
                        Y_try[i, self.paras.E[0]] = t / 100.0
                        obj = self._objective(X_try, Y_try, F_e, F_c)
                        if np.isfinite(obj) and obj > best_user_obj:
                            best_user_obj = obj
                            best_thresholds = [t / 100.0]
                else:
                    for t1 in fine_ranges[0]:
                        for t2 in fine_ranges[1]:
                            Y_try = Y_best.copy()
                            Y_try[i, :] = 1.0
                            Y_try[i, self.paras.E[0]] = t1 / 100.0
                            Y_try[i, self.paras.E[1]] = t2 / 100.0
                            for j in range(2, len(self.paras.E)):
                                Y_try[i, self.paras.E[j]] = best_thresholds[j]
                            obj = self._objective(X_try, Y_try, F_e, F_c)
                            if np.isfinite(obj) and obj > best_user_obj:
                                best_user_obj = obj
                                best_thresholds[0] = t1 / 100.0
                                best_thresholds[1] = t2 / 100.0

            if best_k1k2 is not None:
                k1, k2 = best_k1k2
                X_best[i] = encode_split_row(k1, k2, m, dtype=np.float32)
                Y_best[i, :] = 1.0
                if best_thresholds is not None:
                    for j, eidx in enumerate(self.paras.E):
                        if j < len(best_thresholds):
                            Y_best[i, eidx] = best_thresholds[j]

        final_obj = float(self._objective(X_best, Y_best, F_e, F_c))
        return final_obj, (X_best, Y_best, F_e.copy(), F_c.copy())

    def train(self, initial_solution=None):
        started_at = time.perf_counter()
        best_val = -np.inf
        best_sol = None
        history = []

        min_epochs = int(self.hparams.get("min_epochs", 100))
        patience = int(self.hparams.get("patience", 20))
        rel_tolerance = float(self.hparams.get("rel_tolerance", 1e-4))

        F_e, F_c = self.default_resources(self.paras)
        if initial_solution is not None:
            X0, Y0, F_e0, F_c0 = initial_solution
            F_e = np.asarray(F_e0, dtype=np.float32).reshape(self.paras.n, 1)
            F_c = np.asarray(F_c0, dtype=np.float32).reshape(self.paras.n, 1)
            X0 = np.asarray(X0, dtype=np.float32)
            Y0 = np.asarray(Y0, dtype=np.float32)
            initial_obj = float(self._objective(X0, Y0, F_e, F_c))
            if np.isfinite(initial_obj):
                best_val = initial_obj
                best_sol = (X0.copy(), Y0.copy(), F_e.copy(), F_c.copy())
                history.append(initial_obj)

        target_steps = int(self.hparams["target_steps"])
        outer_ema = float(self.hparams.get("outer_ema", 0.02))

        for epoch in range(self.hparams["max_epochs"]):
            self.buffer.clear()
            best_epoch_obj = -np.inf
            best_epoch_X = None
            best_epoch_Y = None
            episode_final_objs = []
            entropy_X_list = []
            entropy_Y_list = []

            steps = 0
            while steps < target_steps:
                # ---- 新 episode：以 baseline X,Y 开始 ----
                X, Y = _init_feasible_XY(self.paras, self.decision_spec)
                prev_obj = self._objective(X, Y, F_e, F_c)

                # episode 长度 = n（每步决策一个用户）
                for i in range(self.paras.n):
                    if steps >= target_steps:
                        break

                    state = _build_state(
                        i=i,
                        n=self.paras.n,
                        prev_obj=prev_obj,
                        F_e=F_e,
                        F_c=F_c,
                        f_e_max=self.paras.f_e_max,
                        f_c_max=self.paras.f_c_max,
                        paras=self.paras,
                    ).to(self.device)

                    x_idx, y_vec_t, logprob, value, ent_X, ent_Y = self.sample_action(
                        state
                    )
                    entropy_X_list.append(
                        float(ent_X.item())
                        if isinstance(ent_X, torch.Tensor)
                        else float(ent_X)
                    )
                    entropy_Y_list.append(
                        float(ent_Y.item())
                        if isinstance(ent_Y, torch.Tensor)
                        else float(ent_Y)
                    )

                    # 应用动作到第 i 个用户
                    X_new = X.copy()
                    Y_new = Y.copy()
                    y_vec_np = y_vec_t.detach().cpu().numpy().astype(np.float32)
                    X_new, Y_new = self._apply_action_to_XY(
                        X_new,
                        Y_new,
                        user_i=i,
                        x_idx=int(x_idx.item())
                        if isinstance(x_idx, torch.Tensor)
                        else int(x_idx),
                        y_vec=y_vec_np,
                    )

                    # 增量奖励：r_t = U(s_{t+1}) - U(s_t)
                    new_obj = self._objective(X_new, Y_new, F_e, F_c)
                    if not np.isfinite(new_obj) or not np.isfinite(prev_obj):
                        reward = -float(self.hparams.get("obj_scale", 1000.0))
                        new_obj = prev_obj if np.isfinite(prev_obj) else 0.0
                        X_new = X.copy()
                        Y_new = Y.copy()
                    else:
                        reward = float(new_obj - prev_obj)
                    done = 1.0 if (i == self.paras.n - 1) else 0.0

                    # 存 buffer
                    self.buffer.add(
                        state.squeeze(0).detach().cpu(),
                        int(x_idx.item())
                        if isinstance(x_idx, torch.Tensor)
                        else int(x_idx),
                        torch.tensor(y_vec_np, dtype=torch.float32),
                        logprob.detach().cpu(),
                        float(value.item())
                        if isinstance(value, torch.Tensor)
                        else float(value),
                        reward,
                        done,
                    )

                    # 状态推进
                    X, Y = X_new, Y_new
                    prev_obj = new_obj
                    steps += 1

                # episode 结束：final objective
                final_obj = prev_obj
                episode_final_objs.append(final_obj)

                if final_obj > best_epoch_obj:
                    best_epoch_obj = final_obj
                    best_epoch_X = X.copy()
                    best_epoch_Y = Y.copy()

            # 用 rollout 更新策略
            self.update_policy(epoch)

            # ===== Outer Optimization: Closed-form resource allocation (Theorem 1) =====
            if best_epoch_X is None or best_epoch_Y is None:
                print("[Warning] best_epoch_X/Y is None, skip outer update.")
            else:
                F_e, F_c = self.allocate_resources_for_XY(
                    best_epoch_X,
                    best_epoch_Y,
                    F_e=F_e,
                    F_c=F_c,
                    outer_ema=outer_ema,
                )

            # 统计 mean_obj / entropy
            mean_epoch_obj = (
                float(np.mean(episode_final_objs))
                if len(episode_final_objs) > 0
                else float("nan")
            )
            mean_entropy_X = (
                float(np.mean(entropy_X_list))
                if len(entropy_X_list) > 0
                else float("nan")
            )
            mean_entropy_Y = (
                float(np.mean(entropy_Y_list))
                if len(entropy_Y_list) > 0
                else float("nan")
            )
            entropy_coef = self._entropy_coef(epoch)
            y_variance_coef = self._y_variance_coef(epoch)

            # 统计 history 和 best checkpoint
            inner_best_obj = float(best_epoch_obj)  # 旧资源口径
            if best_epoch_X is None or best_epoch_Y is None:
                outer_obj = float("-inf")  # 外层更新后，用新资源重新评估（DSCI 口径）
                latency, acc = float("nan"), float("nan")
            else:
                outer_obj = float(
                    self._objective(best_epoch_X, best_epoch_Y, F_e, F_c)
                )
                latency, acc = get_lat_and_acc(
                    best_epoch_X, best_epoch_Y, F_e, F_c, self.paras
                )

            history.append(outer_obj)
            if (
                outer_obj > best_val
                and best_epoch_X is not None
                and best_epoch_Y is not None
            ):
                best_val = outer_obj
                best_sol = (
                    best_epoch_X.copy(),
                    best_epoch_Y.copy(),
                    F_e.copy(),
                    F_c.copy(),
                )
                self.best_policy_state_dict = {
                    k: v.clone() for k, v in self.policy.state_dict().items()
                }
            self.logs.append(
                {
                    "epoch": int(epoch),
                    "inner_best_obj": inner_best_obj,
                    "outer_obj": outer_obj,
                    "inner_mean_obj": float(
                        mean_epoch_obj
                    ),  # 这个仍然是旧资源下 episode mean
                    "latency": float(latency),
                    "acc": float(acc),
                    "entropy_X": float(mean_entropy_X),
                    "entropy_Y": float(mean_entropy_Y),
                    "entropy_coef": float(entropy_coef),
                    "y_variance_coef": float(y_variance_coef),
                    "steps_collected": int(steps),
                    "num_episodes": int(len(episode_final_objs)),
                    "elapsed_s": float(time.perf_counter() - started_at),
                }
            )
            if self.evaluator is not None:
                self.evaluator.record("ppo", epoch, outer_obj, self.evaluator.best_value)
            print(
                f"Epoch {epoch}: "
                f"inner_best_obj={inner_best_obj:.6f}, outer_obj={outer_obj:.6f}, "
                f"inner_mean_obj={mean_epoch_obj:.6f}, "
                f"latency={latency:.6f}, acc={acc:.6f}, "
                f"entropy_X={mean_entropy_X:.6f}, entropy_Y={mean_entropy_Y:.6f}, "
                f"entropy_coef={entropy_coef:.6g}, "
                f"y_variance_coef={y_variance_coef:.6g}"
            )

            # 收敛检测（窗口内波动很小就停）

            if epoch > min_epochs and len(history) >= 2 * patience:
                current_window = history[-patience:]
                previous_window = history[-2 * patience : -patience]
                curr_mean = np.mean(current_window)
                prev_mean = np.mean(previous_window)
                rel_change = abs(curr_mean - prev_mean) / (abs(prev_mean) + 1e-10)
                cv = np.std(current_window) / (abs(curr_mean) + 1e-10)
                if rel_change < rel_tolerance and cv < (rel_tolerance * 5):
                    print("[Early Stop] Converged!")
                    print(f"Epoch: {epoch}, Rel Change: {rel_change:.6f}, CV: {cv:.6f}")
                    break

        if (
            getattr(self.paras, "resource_mode", None) == "fixed_worker_pool"
            and best_sol is not None
            and (
                self.decision_spec is None
                or (
                    self.decision_spec.split_rule == "optimize"
                    and self.decision_spec.exit_rule == "optimize"
                )
            )
        ):
            polished_val, polished_sol = self._deterministic_polish(best_sol)
            if polished_val > best_val:
                print(f"[Polish] Improved {best_val:.6f} -> {polished_val:.6f}")
                best_val = polished_val
                best_sol = polished_sol
            else:
                print(f"[Polish] PPO solution already optimal ({best_val:.6f})")

        return best_val, best_sol, history
