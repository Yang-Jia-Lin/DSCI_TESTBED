# DSCI 真机实验平台（DSCI-testbed）

DSCI（Dual-Stage Collaborative Inference）是一套面向端-边-云三层网络的协同深度学习推理框架。它在深度模型中引入早退出口（Early Exit），并结合 PPO 强化学习与资源优化，在多用户并发场景下动态决定：

- 每个用户的模型切分位置；
- 每个早退出口的置信度阈值；
- 每个任务在边缘端和云端获得的计算资源配额。

本仓库当前包含 DSCI 仿真实验代码、模型训练与评估脚本、实验绘图脚本，以及面向真机实验平台的目录骨架。真机平台目标是把原有仿真算法迁移到由真实 Device、Edge、Cloud 物理设备组成的三层系统中，在真实网络与硬件条件下验证算法决策效果。

## 项目目标

真机实验以多任务并发和批量决策为核心，按回合驱动完整闭环：

1. **离线准备**：训练模型，生成早退准确率表、退出率表、层 FLOPs 与特征尺寸表，并下发模型权重。
2. **调度决策**：部署端上报多用户网络和设备状态；算法端运行 PPO 与资源优化，生成 `(X, Y, F_e, F_c)`。
3. **协同推理**：Device、Edge、Cloud 按决策执行分段推理和动态早退，采集真实时延、准确率等测量结果并回传。

## 模块分工

| 模块 | 核心职责 |
| --- | --- |
| Algorithm | PPO 策略训练、资源分配求解、状态适配、决策编码、Reward 计算 |
| Deploy | 三层真机组网、推理流水线执行、带宽与 CPU 状态测量、结果回传 |

## 当前仓库结构

```text
DSCI_testbed/
├── Data/
│   ├── CIFAR10/                         # CIFAR-10 数据集目录
│   ├── Resnet50_accs.csv                # 不同阈值下的早退准确率表
│   ├── Resnet50_rates.csv               # 不同阈值下的早退退出率表
│   ├── Resnet50_layer_stats.csv         # 每层特征尺寸与 FLOPs
│   └── ...
│
├── Models/
│   ├── Weights/                         # 预训练权重
│   ├── ModelNet/                        # 模型结构定义
│   ├── Models/                          # 兼容保留的模型结构目录
│   └── Train_and_Evaluate/              # 模型训练、评估与阈值曲线脚本
│
├── Results/                             # 实验输出、图表和优化结果
│   ├── Exp1_Testbed/                    # 真机实验结果目录
│   ├── Exp2_Baseline/
│   ├── Exp3_Dynamic/
│   ├── Exp4_Convergence/
│   ├── Exp5_Ablation/
│   ├── Exp6_EE_Model/
│   ├── Optimize/
│   └── Test/
│
├── Scripts/
│   ├── Exp0_Motivation/
│   ├── Exp1_Testbed/                    # 真机实验脚本目录，当前待实现
│   ├── Exp2_Baseline/
│   ├── Exp3_Dynamic/
│   ├── Exp4_DSCI_Convergency/
│   ├── Exp5_Ablation/
│   └── Exp6_EE_Model/
│
├── Src/
│   ├── Objective/                       # 准确率、退出点、时延和目标函数计算
│   ├── Algorithm/Optimizer/             # DSCI、BF、GA 优化算法主实现
│   ├── Algo/                            # 兼容导入目录，部分模块代理到 Algorithm
│   ├── Deploy/                          # 真机部署执行模块目录，当前待实现
│   ├── Utils/
│   └── paras.py                         # 统一参数入口
│
└── README.md
```

> 注意：开发文档中的结果目录写作 `Result/`，当前代码仓库实际使用的是 `Results/`，并且 `Src/paras.py` 中的路径常量也指向 `Results/`。

## 核心算法说明

算法最终输出：

```python
best_sol = (X, Y, F_e, F_c)
```

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `X` | `(n, m)` | 每个用户的模型切分决策矩阵 |
| `Y` | `(n, m)` | 每个用户的早退置信度阈值矩阵 |
| `F_e` | `(n, 1)` | 每个用户的边缘算力分配，单位 GHz |
| `F_c` | `(n, 1)` | 每个用户的云端算力分配，单位 GHz |

`X[i]` 中应恰好有两个为 `1` 的位置，分别记为 `s1` 和 `s2`，满足 `0 <= s1 < s2 < m`。三段模型切片语义为：

```text
Device: [0,  s1)
Edge:   [s1, s2)
Cloud:  [s2, m)
```

所有层索引均使用 0-based 编号，区间均遵循 Python 左闭右开切片语义。

当前 ResNet50 配置中：

- 模型层数 `m = 128`；
- 早退出口集合 `E = [57, 103]`；
- `Y` 只有 `Y[i, 57]` 和 `Y[i, 103]` 对部署执行有实际意义，其余位置一般作为占位值。

## 参数入口

全局参数位于 `Src/paras.py`。主要参数包括：

| 参数 | 含义 | 当前来源 |
| --- | --- | --- |
| `n` | 并发用户 / 任务数量 | `NUM_USERS` |
| `m` | 模型层数 | `NUM_LAYERS` |
| `E` | 早退层集合 | `EARLY_EXIT_LAYERS` |
| `D` | 每层输出特征大小，单位 bytes | `Data/Resnet50_layer_stats.csv` |
| `C` | 每层计算量，近似 FLOPs | `Data/Resnet50_layer_stats.csv` |
| `F_u` | 每个用户端侧算力 | `USER_FREQs` |
| `f_e_max` | 边缘最大可分配算力 | `EDGE_MAX_FREQ` |
| `f_c_max` | 云端最大可分配算力 | `CLOUD_MAX_FREQ` |
| `b_e` | 端到边链路带宽参数 | `BANDWIDTH_EDGE` |
| `b_c` | 边到云链路带宽参数 | `BANDWIDTH_CLOUD` |
| `H_u` | 用户无线信道增益 | `CHANNEL_GAINS_USERS` |
| `rates` | 早退退出率表 | `Data/*_rates.csv` |
| `accs` | 早退准确率表 | `Data/*_accs.csv` |

真机实验中建议新增或适配每用户实测带宽 `B_u[i] = users[i]["BW_d2e"]`，并扩展 `Src/Objective/compute_latency.py`，避免将真实带宽强行折算为仿真信道增益 `H_u`。

## 可运行入口

安装依赖：

```bash
pip install numpy pandas torch matplotlib seaborn
```

运行 DSCI 优化实验：

```bash
python -m Src.Algorithm.Optimizer.DSCI.run_DSCI
```

该入口会读取 `Src/paras.py` 中的默认配置，训练 PPO Agent，并将最优解、训练日志和图表保存到 `Results/Optimize/DSCI/`。

运行基线实验：

```bash
python -m Scripts.Exp2_Baseline.run_SOTA_baseline
```

运行动态环境实验：

```bash
python -m Scripts.Exp3_Dynamic.run_resource_dynamic
python -m Scripts.Exp3_Dynamic.run_user_dynamic
```

运行收敛性实验：

```bash
python -m Scripts.Exp4_DSCI_Convergency.run_convergence
```

运行消融实验：

```bash
python -m Scripts.Exp5_Ablation.run_ablation
```

运行早退模型训练与分析脚本：

```bash
python -m Models.Train_and_Evaluate.resnet50_train
python -m Models.Train_and_Evaluate.resnet50_evaluate
python -m Models.Train_and_Evaluate.resnet50_thred_curve
```

## 真机实验接口设计

真机实验中，算法模块建议作为 HTTP Server 运行在 Edge 节点，部署模块作为 Client。每一轮实验包含两次请求：

```text
Deploy Client                         Algo Server
     |                                     |
     |-- POST /api/v1/decision ---------->|
     |   上报多用户状态 JSON               | state_adapter -> PPO -> resource optimizer
     |<------------- 决策 JSON ------------| decision_codec
     |                                     |
     |   执行多用户并发推理并采集测量数据    |
     |                                     |
     |-- POST /api/v1/measurements ------>|
     |   上报真实测量结果                  | reward_adapter -> PPO Buffer
     |<------------- 确认信号 -------------|
```

### 状态上报格式

```json
{
  "round_id": "round_0001",
  "model_name": "Resnet50",
  "users": [
    {
      "user_id": 0,
      "BW_d2e": 18.5,
      "f_u": 2.0,
      "cpu_util_device": 0.43
    }
  ],
  "edge": {
    "f_e_max": 20.0,
    "cpu_util": 0.62
  },
  "cloud": {
    "BW_e2c": 120.0,
    "f_c_max": 50.0,
    "cpu_util": 0.45
  }
}
```

### 决策下发格式

```json
{
  "decision_id": "round_0001",
  "model_name": "Resnet50",
  "num_users": 10,
  "num_layers": 128,
  "early_exit_layers": [57, 103],
  "layer_index_base": 0,
  "slice_semantics": "python_left_closed_right_open",
  "users": [
    {
      "user_id": 0,
      "partition_s1": 42,
      "partition_s2": 85,
      "device_layers": [0, 42],
      "edge_layers": [42, 85],
      "cloud_layers": [85, 128],
      "exit_thresholds": {
        "57": 0.83,
        "103": 0.91
      },
      "edge_compute_alloc": 1.7,
      "cloud_compute_alloc": 3.4,
      "edge_compute_quota": 0.085,
      "cloud_compute_quota": 0.068,
      "X_row": [],
      "Y_row": []
    }
  ]
}
```

部署执行应以 `device_layers`、`edge_layers`、`cloud_layers` 和 `exit_thresholds` 为准；`X_row` 与 `Y_row` 仅用于调试。

### 测量结果回传格式

```json
{
  "decision_id": "round_0001",
  "measurements": [
    {
      "user_id": 0,
      "T_device": 12.3,
      "T_trans_d2e": 6.2,
      "T_edge": 8.7,
      "T_trans_e2c": 1.8,
      "T_cloud": 5.1,
      "T_total": 34.1,
      "exit_layer": 103,
      "exit_location": "edge",
      "exit_confidence": 0.923,
      "prediction": 3,
      "ground_truth": 3,
      "is_correct": true
    }
  ]
}
```

若某用户在 Device 或 Edge 触发早退，后续未执行阶段的时延应填 `0`；异常或测量缺失应填 `null`，不要伪造耗时数据。

Reward 可按以下形式从真实测量结果计算：

```text
reward_i = alpha * is_correct_i - beta * T_total_i
round_reward = mean(reward_i)
```

## 真机部署模块规划

`Src/Deploy/` 当前是部署模块预留目录，建议按以下结构实现：

```text
Src/Deploy/
├── Device/
│   ├── model_device.py
│   ├── early_exit.py
│   ├── comm.py
│   └── run_device.py
├── Edge/
│   ├── model_edge.py
│   ├── resource_ctrl.py
│   └── run_edge.py
├── Cloud/
│   ├── model_cloud.py
│   ├── resource_ctrl.py
│   └── run_cloud.py
└── Monitor/
    ├── bandwidth.py
    ├── cpu_monitor.py
    └── state_reporter.py
```

早退判断必须跟随切分点动态归属：某早退层落在哪个节点的执行区间内，就由哪个节点完成判断，不应硬编码为固定节点。

## 真机接口层规划

建议在 `Src/Algo/Interface/` 中新增以下适配层：

```text
Src/Algo/Interface/
├── api_server.py         # HTTP Server：接收状态、下发决策、接收测量结果
├── state_adapter.py      # state JSON -> Paras
├── decision_codec.py     # (X, Y, F_e, F_c) -> 部署端 JSON
└── reward_adapter.py     # measurements JSON -> PPO reward
```

`decision_codec.py` 至少应包含以下校验：

- 每个用户的 `X_row` 恰好有两个切分点；
- `0 <= s1 < s2 < m`；
- 所有早退层都在 `[0, m)` 内；
- 有效早退阈值位于 `[0, 1]`；
- `sum(F_e) <= f_e_max`；
- `sum(F_c) <= f_c_max`。

## 开发优先级

短期建议优先打通最小闭环：

1. 在 `Src/Algo/Interface/` 实现状态适配、决策编码与 Flask API。
2. 扩展 `compute_latency.py` 支持每用户实测 `B_u[i]`。
3. 在 `Scripts/Exp1_Testbed/` 实现一轮 `状态 -> 决策 -> 推理 -> 回传` 的主控脚本。
4. 在 `Src/Deploy/` 实现 Device、Edge、Cloud 的分段推理与动态早退。
5. 先使用 `run_dsci_experiment(custom_paras_dict=...)` 每轮求解决策验证接口，再逐步替换为加载 PPO checkpoint 的在线决策模式。
