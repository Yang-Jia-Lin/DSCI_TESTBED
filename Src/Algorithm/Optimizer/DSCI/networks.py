"""
ActorCritic 网络
Src/Optimizer/DSCI/networks.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        action_dim_Y: int,
        hidden_dim: int = 128,
        beta_eps: float = 1e-4,
    ):
        """
        Args:
            state_dim: 状态维度
            num_layers: 模型层数 m，用于枚举所有合法 (k1,k2)
            action_dim_Y: 早退层集合 |H| 的维度（只对早退层输出阈值）
            hidden_dim: 共享层隐藏维度
            beta_eps: Beta 参数的最小值，避免 alpha/beta 过小导致数值问题
        """
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2 to form (k1,k2) pairs."
        assert action_dim_Y >= 0, "action_dim_Y should be >= 0."

        self.state_dim = state_dim
        self.num_layers = num_layers
        self.action_dim_Y = action_dim_Y
        self.beta_eps = beta_eps

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # -------- X: categorical over valid (k1,k2) pairs --------
        # 预先列出所有合法 (k1,k2) 组合，k1 < k2
        # pair_index -> (k1, k2)
        pairs = [(k1, k2) for k1 in range(num_layers) for k2 in range(k1 + 1, num_layers)]
        pair_tensor = torch.tensor(pairs, dtype=torch.long)  # [num_pairs, 2]
        self.register_buffer("x_pairs", pair_tensor)  # 不参与训练，但会随 device 一起移动
        self.num_pairs = pair_tensor.shape[0]

        # actor_X 输出 logits，维度 = num_pairs
        self.actor_X = nn.Linear(hidden_dim, self.num_pairs)

        # -------- Y: Beta distribution parameters --------
        # 输出 alpha/beta（>0），维度 = action_dim_Y
        # 用 softplus 保证正数，再加 eps 避免接近 0
        self.actor_Y_alpha = nn.Linear(hidden_dim, action_dim_Y)
        self.actor_Y_beta = nn.Linear(hidden_dim, action_dim_Y)

        # -------- Critic --------
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        # 共享层和 critic 采用默认；对 actor 输出层做一个小初始化，避免初期 logits/参数极端
        nn.init.orthogonal_(self.actor_X.weight, gain=0.01)
        nn.init.constant_(self.actor_X.bias, 0.0)

        if self.action_dim_Y > 0:
            nn.init.orthogonal_(self.actor_Y_alpha.weight, gain=0.01)
            nn.init.constant_(self.actor_Y_alpha.bias, 0.0)
            nn.init.orthogonal_(self.actor_Y_beta.weight, gain=0.01)
            nn.init.constant_(self.actor_Y_beta.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, state: torch.Tensor):
        """
        Returns:
            logits_X: [B, num_pairs]  (Categorical logits over (k1,k2) pairs)
            alpha_Y:  [B, action_dim_Y] (Beta alpha > 0)
            beta_Y:   [B, action_dim_Y] (Beta beta  > 0)
            value:    [B, 1]
        """
        # 共享层
        features = self.shared(state)

        # X的分布输出（每个索引的选择概率）
        logits_X = self.actor_X(features)

        # Y的分布输出（Beta分布的形状，由alpha和beta决定）
        if self.action_dim_Y > 0:
            alpha_raw = self.actor_Y_alpha(features)
            beta_raw = self.actor_Y_beta(features)
            alpha_Y = F.softplus(alpha_raw) + self.beta_eps
            beta_Y = F.softplus(beta_raw) + self.beta_eps
        else: # 没有早退层
            alpha_Y = state.new_zeros((state.shape[0], 0))
            beta_Y = state.new_zeros((state.shape[0], 0))

        # 价值网络输出（标量）
        value = self.critic(features)

        # 返回三组输出（X、Y、V）
        return logits_X, alpha_Y, beta_Y, value
