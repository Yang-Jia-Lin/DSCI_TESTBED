# Phase 2 Algorithm Service: Adaptive DSCI/PPO Logic

本文说明 `Service/algo_service.py` 中当前 DSCI 调度服务的自适应逻辑，重点是网络状态波动时如何复用历史 PPO 策略、避免每次从随机权重冷启动。

## 服务入口

`AlgoService.make_decision(state)` 是 Phase 2 在线服务入口。它接收一轮系统状态 JSON，先通过 `to_paras(state)` 构造 `Paras`，再生成 `state_signature`。签名包含：

- 模型信息：`bundle_id`、`manifest_id`、`model_hash`、层数 `m`、早退出口 `exit_ids`
- 系统结构：用户数、`resource_mode`
- 用户状态：每个用户的 `BW_d2e`、`f_u`、compute/execution profile
- 边缘/云状态：算力、`BW_e2c`、worker count、profile

服务仍保持“快速返回 + 后台训练”的模式：当前请求不会等待 PPO 收敛，后台训练完成后再更新缓存。

## 历史缓存池

服务维护多条历史缓存，默认最多保留 `max_cached_solutions=10`。每次 PPO 训练完成后会保存三类文件：

- `solution_*.npz`：最优 `X/Y/F_e/F_c/objective`
- `solution_*_meta.json`：状态签名、兼容键、状态向量、训练模式、策略路径
- `solution_*_policy.pt`：PPO policy 权重

同时维护 `latest_solution.npz`、`latest_solution_meta.json` 和 `latest_solution_policy.pt` 作为最近一次结果。

## 兼容性与相似度

历史缓存只有在 `compat_key` 完全一致时才允许复用。兼容键包括：

- `bundle_id`、`manifest_id`、`model_hash`
- `resource_mode`
- 用户数、模型层数、早退出口
- 每个用户 profile
- edge/cloud profile

兼容后，服务用归一化欧氏距离比较状态向量。状态向量包含：

- 每个用户的 `BW_d2e` 和 `f_u`
- edge `f_e_max` 和 worker count
- cloud `f_c_max`、`BW_e2c` 和 worker count

距离计算按每个字段自身量级归一化，避免 Hz、Mbps、worker count 的量纲差异直接主导结果。

## 自适应决策分档

找到最相似历史缓存后，根据距离选择行为：

| 距离 | 模式 | 行为 |
| --- | --- | --- |
| `<= 0.005` | `reuse` | 认为是小波动，直接复用历史解，不启动 PPO 训练 |
| `0.005 ~ 0.05` | `near` | 从历史 PPO 权重热启动，短训练 |
| `0.05 ~ 0.15` | `medium` | 从历史 PPO 权重热启动，中等训练 |
| `> 0.15` 或无兼容缓存 | `cold` | 使用默认 PPO 参数完整训练 |

非 exact 状态下，服务会把“默认解”和“最近历史解在当前 `Paras` 下重新计算后的 objective”做比较，先返回 objective 更高的解。后台训练若启动，则在完成后替换当前缓存。

## PPO 训练参数

离线 DSCI 默认参数保持不变，`run_DSCI.py` 仍默认 `outer_ema=0.02`。服务路径会根据 warm-start 模式调整训练强度：

| 模式 | `max_epochs` | `min_epochs` | `target_steps` | `k_epochs` | `lr` | `entropy_coef` | `outer_ema` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `near` | 30 | 8 | 400 | 5 | `5e-5` | 0.003 | 1.0 |
| `medium` | 80 | 20 | 800 | 8 | `8e-5` | 0.006 | 0.5 |
| `cold` | 200 | 100 | 1500 | 10 | `1e-4` | 0.01 | 服务默认 1.0，除非 custom hyperparams 覆盖 |

`PPOAgent.train(initial_solution=...)` 支持把历史解作为初始 best solution。这样 warm-start 训练即使没有产生更优策略，也不会把缓存退化成比历史解更差的结果。

## 10 用户网络波动示例

假设当前有 10 个用户端设备，每轮都上报不同的 `BW_d2e`：

1. 第一次没有兼容缓存，服务返回默认解，同时后台 cold/full PPO 训练。
2. 训练完成后，服务保存解和 policy 权重。
3. 后续网络轻微波动时，若距离 `<= 0.005`，直接复用历史 `X/Y/F_e/F_c`。
4. 波动稍大时，服务先返回历史解或默认解中 objective 更高者，同时后台从历史 policy 权重继续训练。
5. 波动很大或模型/profile 变化时，历史策略不再可信，服务走 cold/full 训练。

PPO 内部一个 episode 仍是 `n` 步。10 个用户时就是 10 步，每步为一个用户选择：

- `X`：两个模型切分点 `(k1, k2)`
- `Y`：早退层阈值

奖励仍是增量 objective：`objective(X_new, Y_new) - objective(X_old, Y_old)`。

## 观测字段

`AlgoService.health()` 增加了自适应缓存诊断字段：

- `cache_entries`
- `last_reuse_distance`
- `last_training_mode`
- `last_warm_start_source`
- `policy_cache_enabled`

返回的 decision 也会带 `decision_source`，例如：

- `cached_dsci:exact`
- `cached_dsci:reuse:0.001234`
- `cached_dsci:warm:0.032100`
- `default`

## 当前边界

当前版本没有启用在线 PPO 微调。`report_measurements()` 仍只计算和记录实测 reward，并返回 `policy_updated=False`。也就是说，本次改造解决的是“状态变化时 PPO 收敛加速”，不是“利用线上实测反馈持续更新 policy”。
