# Phase 2 README

Phase 2 是在线调度层。Scheduler 接收 Device/Edge/Cloud 的状态，加载 `Data/Bundles/<bundle_id>/` 与 `Data/Profiles/<profile_id>/`，为每个用户生成模型切分点和早退阈值。

## 主要入口

| 入口 | 说明 |
| --- | --- |
| `Service/api_server.py` | Flask API 服务，负责设备注册、轮次同步、状态汇总、决策下发和测量回传 |
| `Service/algo_service.py` | 在线调度服务，封装 DSCI/PPO、缓存、固定策略和 warm-start |
| `Optimizer/DSCI/run_DSCI.py` | DSCI/PPO 优化器入口 |
| `Optimizer/BF/run_BF.py` | brute-force 基线入口 |
| `Optimizer/GA/run_GA.py` | genetic algorithm 基线入口 |
| `Objective/` | 精度、时延、早退概率和总 reward 计算 |

DSCI 服务内部自适应逻辑见 [Service/README_alg_service.md](Service/README_alg_service.md)。

## 启动 Scheduler

单用户真机运行：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 1
```

多用户运行时，`--expected-users` 必须等于本轮 Device 数量：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2
```

Scheduler 默认监听 HTTP `:8000`。Device 到齐后，Scheduler 查询 Edge/Cloud 状态，构建联合优化问题，然后返回每个 `user_id` 对应的决策。

## 决策模式

默认模式使用 DSCI/PPO，并可能在后台训练或复用历史缓存。调试和消融时可用固定策略：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --fixed-split 3 10
python -m Src.Phase2_Scheduler.Service.api_server --fixed-threshold 0.7
python -m Src.Phase2_Scheduler.Service.api_server --no-auto-train
```

`--fixed-split S1 S2` 表示所有用户都使用相同的 Device/Edge/Cloud 切分边界。`--fixed-threshold VALUE` 表示所有早退出口使用同一阈值。

## 依赖的离线产物

运行前应确认 Scheduler 机器能访问：

| 产物 | 路径 |
| --- | --- |
| 模型包 manifest 和 early-exit curves | `Data/Bundles/<bundle_id>/manifest.json`、`exit_curves.csv` |
| Device profile | `Data/Profiles/<device_profile_id>/` |
| Edge profile | `Data/Profiles/<edge_profile_id>/` |
| Cloud profile | `Data/Profiles/<cloud_profile_id>/` |

新生成的 profile 必须复制到 Scheduler 机器，见 [DATA_README](../../Data/DATA_README.md) 和 [Phase1_README](../Phase1_Offline/Phase1_README.md)。

## API 流程

在线运行时主要使用 v2 轮次接口：

1. Device 使用 `round_id` 和 `user_id` 注册。
2. Device 持续 heartbeat，Scheduler 等待同一轮所有用户到齐。
3. Scheduler 查询 Edge/Cloud `/status`，合成全局 state。
4. Scheduler 调用 `AlgoService` 生成每个用户的切分点和阈值。
5. Device 通过 decisions 接口领取结果并执行推理。
6. Device 回传测量结果，Scheduler 记录 reward 和运行日志。

## 相关文档

- [Src_README](../Src_README.md)：代码总入口和快速开始。
- [Phase3_README](../Phase3_Runtime/Phase3_README.md)：Cloud/Edge/Device 启动顺序。
- [Scripts_README](../../Scripts/Scripts_README.md)：实验脚本如何调用调度产物。
## 论文实验同步决策接口

论文实验应直接调用 `Src.Phase2_Scheduler.Service.solve_decision(state, spec)`，
而不是依赖在线服务的默认/缓存决策。`DecisionSpec` 可约束切分、早退、合法
部署对、联合/独立求解、两阶段求解、优化器、随机种子和统一目标函数预算。

```python
from Src.Phase2_Scheduler.Service import DecisionSpec, solve_decision

result = solve_decision(
    state,
    DecisionSpec(
        coordination="joint",
        split_rule="optimize",
        exit_rule="disabled",       # Split only
        optimizer="ppo",
        evaluation_budget=5000,
        seed=42,
    ),
)
```

多用户同步求解会自动启用共享资源模型。共享 D2E 链路的用户应报告相同
`d2e_link_id`，并通过 `edge.d2e_capacity_mbps`（或用户级
`d2e_capacity_mbps`）报告链路总容量；E2C 可通过 `cloud.e2c_link_id` 和
`cloud.e2c_capacity_mbps` 配置。返回值同时包含决策、预测指标、统一
optimizer trace、目标函数调用次数以及 setup/solve/total 耗时。
