"""
rollout 缓冲区
Src/Optimizer/DSCI/buffer.py
"""

import torch


class RolloutBuffer:
    def __init__(self):
        self.dones = None
        self.rewards = None
        self.values = None
        self.logprobs = None
        self.actions_Y = None
        self.actions_X = None
        self.states = None
        self.clear()

    def clear(self):
        # 每个元素是单步数据
        self.states = []  # state tensor, shape [state_dim]
        self.actions_X = []  # x_idx, int/LongTensor scalar
        self.actions_Y = []  # y tensor, shape [action_dim_Y]
        self.logprobs = []  # total logprob scalar
        self.values = []  # value scalar (or shape [1])
        self.rewards = []  # reward scalar
        self.dones = []  # done scalar (0.0/1.0)

    def __len__(self):
        return len(self.rewards)  # type: ignore

    def add(self, state, action_X, action_Y, logprob, value, reward, done):
        """
        Args:
            state: Tensor[state_dim] or Tensor[1,state_dim]
            action_X: int or Tensor scalar (categorical index)
            action_Y: Tensor[action_dim_Y] (Beta sample) (can be empty dim if |H|=0)
            logprob: Tensor scalar (logp_X + logp_Y) or float
            value: Tensor scalar or float
            reward: float
            done: bool/float/int (终止标记)
        """
        # 状态：确保存成 [state_dim] 的 tensor
        if isinstance(state, torch.Tensor):
            s = state.detach()
            if s.dim() == 2 and s.shape[0] == 1:
                s = s.squeeze(0)
        else:
            s = torch.tensor(state, dtype=torch.float32)
        self.states.append(s)

        # X：categorical index，存 long scalar
        if isinstance(action_X, torch.Tensor):
            ax = action_X.detach().long().view(-1)
            ax = ax[0]  # scalar
        else:
            ax = torch.tensor(int(action_X), dtype=torch.long)
        self.actions_X.append(ax)

        # Y：Beta sample，存 float tensor [action_dim_Y]
        if isinstance(action_Y, torch.Tensor):
            ay = action_Y.detach().float()
            if ay.dim() == 2 and ay.shape[0] == 1:
                ay = ay.squeeze(0)
        else:
            ay = torch.tensor(action_Y, dtype=torch.float32)
        self.actions_Y.append(ay)

        # logprob/value/reward/done：存成标量 tensor 更统一
        lp = (
            logprob.detach()
            if isinstance(logprob, torch.Tensor)
            else torch.tensor(logprob, dtype=torch.float32)
        )
        lp = lp.view(-1)[0]
        self.logprobs.append(lp)

        v = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=torch.float32)
        )
        v = v.view(-1)[0]
        self.values.append(v)

        r = (
            reward.detach()
            if isinstance(reward, torch.Tensor)
            else torch.tensor(reward, dtype=torch.float32)
        )
        r = r.view(-1)[0]
        self.rewards.append(r)

        d = (
            done.detach()
            if isinstance(done, torch.Tensor)
            else torch.tensor(done, dtype=torch.float32)
        )
        d = d.view(-1)[0]
        # 确保是 0/1 float
        self.dones.append(torch.clamp(d, 0.0, 1.0))

    @torch.no_grad()
    def compute_advantages(self, gamma: float, lam: float):
        """
        GAE(λ) for DSCI, supports multi-step episodes via dones mask.
        Returns:
            advantages: Tensor[T]
            returns: Tensor[T]
        """
        if len(self.rewards) == 0:
            return torch.empty(0), torch.empty(0)

        rewards = torch.stack(self.rewards).float()  # [T]
        values = torch.stack(self.values).float()  # [T]
        dones = torch.stack(self.dones).float()  # [T]

        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # done=1 表示 episode 结束，不 bootstrap
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def as_tensors(self, device=None):
        """
        将 buffer 中的 list 统一 stack 成 tensor，方便训练。
        Returns dict:
            states: [T, state_dim]
            actions_X: [T] (long)
            actions_Y: [T, action_dim_Y]
            logprobs: [T]
            values: [T]
            rewards: [T]
            dones: [T]
        """
        if len(self.rewards) == 0:
            # 返回空 tensor，避免 agent 里特殊判断太多
            out = {
                "states": torch.empty(0),
                "actions_X": torch.empty(0, dtype=torch.long),
                "actions_Y": torch.empty(0),
                "logprobs": torch.empty(0),
                "values": torch.empty(0),
                "rewards": torch.empty(0),
                "dones": torch.empty(0),
            }
            return {
                k: (v.to(device) if device is not None else v) for k, v in out.items()
            }

        states = torch.stack(self.states).float()
        actions_X = torch.stack(self.actions_X).long()
        # actions_Y 可能是空维度（|H|=0），stack 也能正常工作
        actions_Y = torch.stack(self.actions_Y).float()
        logprobs = torch.stack(self.logprobs).float()
        values = torch.stack(self.values).float()
        rewards = torch.stack(self.rewards).float()
        dones = torch.stack(self.dones).float()

        if device is not None:
            states = states.to(device)
            actions_X = actions_X.to(device)
            actions_Y = actions_Y.to(device)
            logprobs = logprobs.to(device)
            values = values.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

        return {
            "states": states,
            "actions_X": actions_X,
            "actions_Y": actions_Y,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }

    def get_all_data(self):
        # 保持接口兼容：返回浅拷贝
        return {
            "states": self.states.copy(),
            "actions_X": self.actions_X.copy(),
            "actions_Y": self.actions_Y.copy(),
            "logprobs": self.logprobs.copy(),
            "values": self.values.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
        }

    def extend(self, data):
        self.states += data["states"]
        self.actions_X += data["actions_X"]
        self.actions_Y += data["actions_Y"]
        self.logprobs += data["logprobs"]
        self.values += data["values"]
        self.rewards += data["rewards"]
        self.dones += data["dones"]
