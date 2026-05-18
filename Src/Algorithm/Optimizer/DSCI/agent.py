"""
Src/Optimizer/DSCI/agent.py

改动：
1) Episode 包含 n 个 steps（每步一个用户）
2) X: categorical index over all valid (k1,k2) pairs（来自 network.x_pairs）
3) Y: Beta 分布，仅对早退层集合 |E| 输出/采样（无需硬裁剪）
4) 数值稳定：adv norm、grad clip、严格 on-policy，移除 TopK/off-policy 等机制
"""

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from Src.Algorithm.Optimizer.DSCI.buffer import RolloutBuffer
from Src.Algorithm.Optimizer.DSCI.networks import ActorCritic
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import get_lat_and_acc, objective
from Src.Utils.parsing_data import split_points_matrix


# ---------- 状态构造（紧凑 Markov） ----------
def _build_state(
    i: int,
    n: int,
    prev_obj: float,
    F_e: np.ndarray,
    F_c: np.ndarray,
    f_e_max: float,
    f_c_max: float,
    obj_scale: float = 1000.0,
) -> torch.Tensor:
    """
    state = [i_norm, remaining_norm, tanh(prev_obj/scale), fe_i_norm, fc_i_norm]
    """
    i_norm = float(i) / float(max(n, 1))
    remaining_norm = float(n - i) / float(max(n, 1))
    prev_obj_squashed = float(np.tanh(prev_obj / obj_scale))

    # F_e, F_c are (n,1) in your code
    fe_i = float(F_e[i, 0]) if F_e.ndim == 2 else float(F_e[i])
    fc_i = float(F_c[i, 0]) if F_c.ndim == 2 else float(F_c[i])
    fe_i_norm = fe_i / float(max(f_e_max, 1e-12))
    fc_i_norm = fc_i / float(max(f_c_max, 1e-12))

    s = torch.tensor(
        [i_norm, remaining_norm, prev_obj_squashed, fe_i_norm, fc_i_norm],
        dtype=torch.float32,
    ).unsqueeze(0)
    return s


# ---------- 初始化一个可行解（给未决策用户用作 baseline） ----------
def _init_feasible_XY(paras):
    """
    生成一个“默认可行”的 X, Y，用作 episode 初始基线和未决策用户的占位。
    - X: 每行两个切分点 (k1,k2)，这里用 (m//3, 2m//3)
    - Y: 全 1，早退层也先设为 1（表示阈值高，倾向不早退）
    """
    n, m = paras.n, paras.m
    X = np.zeros((n, m), dtype=np.float32)
    k1 = max(0, m // 3)
    k2 = min(m - 1, (2 * m) // 3)
    if k1 == k2:
        k2 = min(m - 1, k1 + 1)

    for i in range(n):
        X[i, k1] = 1.0
        X[i, k2] = 1.0

    Y = np.ones((n, m), dtype=np.float32)
    # 早退层也先设 1（不强制），RL 会学到更优的阈值
    for ee in paras.E:
        if 0 <= ee < m:
            Y[:, ee] = 1.0
    return X, Y


def compute_iota_kappa(X, compute_sizes, exit_prob):
    """计算拉格朗日参数 iota 和 kappa"""
    n, m = X.shape
    c = np.asarray(compute_sizes)
    iota = np.zeros(n)
    kappa = np.zeros(n)
    split_pts = split_points_matrix(X)
    for i in range(n):
        p1, p2 = split_pts[i]
        for j in range(p1 + 1, p2 + 1):
            iota[i] += exit_prob[i, j] * c[p1 + 1 : j + 1].sum()
        for j in range(p2 + 1, m):
            kappa[i] += exit_prob[i, j] * c[p2 + 1 : j + 1].sum()
    return iota, kappa


def allocate_resources(iota, kappa, f_e_max, f_c_max):
    """计算凸优化后的资源分配"""
    sqrt_i, sqrt_k = np.sqrt(iota + 1e-12), np.sqrt(kappa + 1e-12)
    f_e = f_e_max * sqrt_i / max(sqrt_i.sum(), 1e-12)
    f_c = f_c_max * sqrt_k / max(sqrt_k.sum(), 1e-12)
    return f_e, f_c


class PPOAgent:
    def __init__(self, paras, hyperparams):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paras = paras
        self.hparams = hyperparams
        self.initial_entropy_coef = hyperparams.get("entropy_coef", 0.01)  # 熵系数衰减
        self.entropy_decay = hyperparams.get("entropy_decay", 0.99)  # 熵系数衰减

        # ---------- 维度 ----------
        self.state_dim = 5  # 状态：5 维
        self.action_dim_Y = len(self.paras.E)  # 动作 Y：早退层（|E|）

        # ---------- 网络 ----------
        policy_net = ActorCritic(
            state_dim=self.state_dim,
            num_layers=self.paras.m,
            action_dim_Y=self.action_dim_Y,
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

        # ---- 写 X：清空后置 2 个切分点 ----
        X[user_i, :] = 0.0
        x_pairs = cast(torch.Tensor, self.policy.x_pairs)
        pair = x_pairs[x_idx].detach().cpu().numpy()  # [k1,k2]
        k1, k2 = int(pair[0]), int(pair[1])
        X[user_i, k1] = 1.0
        X[user_i, k2] = 1.0

        # ---- 写 Y：默认全 1，只写早退层阈值 ----
        Y[user_i, :] = 1.0
        if len(self.paras.E) > 0:
            for j, layer_idx in enumerate(self.paras.E):
                if 0 <= layer_idx < m:
                    Y[user_i, layer_idx] = float(y_vec[j])
        return X, Y

    def update_policy(self, epoch: int):
        entropy_coef = self.initial_entropy_coef * (self.entropy_decay**epoch)

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
            else:
                logp_Y = torch.zeros_like(logp_X)
                ent_Y = torch.zeros_like(ent_X)

            new_logprob = logp_X + logp_Y  # [T]
            entropy = ent_X + ent_Y  # [T]

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

            total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def train(self):
        best_val = -np.inf
        best_sol = None
        history = []

        min_epochs = 100  # 强制最小训练轮数（观察图表，100次之前仍在快速上升）
        patience = 20  # 检测窗口大小
        rel_tolerance = 1e-4  # 相对容忍度（例如：0.01% 的改进）

        # 初始化资源
        F_e = np.ones((self.paras.n, 1), dtype=np.float32) * (
            self.paras.f_e_max / self.paras.n
        )
        F_c = np.ones((self.paras.n, 1), dtype=np.float32) * (
            self.paras.f_c_max / self.paras.n
        )

        target_steps = int(self.hparams["target_steps"])

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
                X, Y = _init_feasible_XY(self.paras)
                prev_obj = objective(X, Y, F_e, F_c, self.paras)

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
                    new_obj = objective(X_new, Y_new, F_e, F_c, self.paras)
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
                # 1. 用 best_epoch_Y 计算每个用户在每层的组合退出概率 P_ij
                exit_prob = compute_layer_exit_probs(
                    best_epoch_Y, self.paras
                )  # shape (n,m)
                assert exit_prob.shape == (self.paras.n, self.paras.m)
                # 2. 计算 iota / kappa
                compute_sizes = self.paras.C
                iota, kappa = compute_iota_kappa(best_epoch_X, compute_sizes, exit_prob)
                # 3. 闭式解分配资源（返回的是 1D (n,)）
                new_f_e, new_f_c = allocate_resources(
                    iota, kappa, self.paras.f_e_max, self.paras.f_c_max
                )
                # 4. 转成 objective 需要的形状
                new_F_e = new_f_e.reshape(self.paras.n, 1).astype(np.float32)
                new_F_c = new_f_c.reshape(self.paras.n, 1).astype(np.float32)
                # 5. EMA 平滑，避免 epoch 之间资源剧烈震荡导致训练不稳
                eta = float(self.hparams.get("outer_ema", 0.02))  # 0~1
                F_e = ((1 - eta) * F_e + eta * new_F_e).astype(np.float32)
                F_c = ((1 - eta) * F_c + eta * new_F_c).astype(np.float32)

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

            # 统计 history 和 best checkpoint
            inner_best_obj = float(best_epoch_obj)  # 旧资源口径
            if best_epoch_X is None or best_epoch_Y is None:
                outer_obj = float("-inf")  # 外层更新后，用新资源重新评估（DSCI 口径）
                latency, acc = float("nan"), float("nan")
            else:
                outer_obj = float(
                    objective(best_epoch_X, best_epoch_Y, F_e, F_c, self.paras)
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
                    "steps_collected": int(steps),
                    "num_episodes": int(len(episode_final_objs)),
                }
            )
            print(
                f"Epoch {epoch}: "
                f"inner_best_obj={inner_best_obj:.6f}, outer_obj={outer_obj:.6f}, "
                f"inner_mean_obj={mean_epoch_obj:.6f}, "
                f"latency={latency:.6f}, acc={acc:.6f}, "
                f"entropy_X={mean_entropy_X:.6f}, entropy_Y={mean_entropy_Y:.6f}"
            )

            # 收敛检测（窗口内波动很小就停）

            if epoch > min_epochs:
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

        return best_val, best_sol, history
