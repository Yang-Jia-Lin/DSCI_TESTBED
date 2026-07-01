# 简单测试
1. **Cloud**
```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID=v100-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Cloud.run_cloud --bundle-id resnet50-cifar10-ee-v1 --backend pytorch
```

2. **Edge**
```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="edge-pytorch-resnet50-cifar10"
python -m Src.Phase3_Runtime.Edge.run_edge --bundle-id resnet50-cifar10-ee-v1 --backend pytorch
```

3. **Scheduler**
```powershell
$env:DSCI_EXPECTED_USERS="1"
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 1
```

4. **Jetson**
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=nx-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Device.run_device --bundle-id resnet50-cifar10-ee-v1 --backend pytorch --round-id r50-cifar10-cloud-002 --user-id 0 --test-samples 1
```

---

> [!info] SEAM 原型系统
> - 实现 Device → Edge → Cloud 三级协同推理。
> - 核心思想：将一个带有早退出口的 DNN 模型按原子 Segment 切分，由 PPO 强化学习优化器（DSCI）为每个用户决定最优的切分点和早退阈值，使延迟与精度的加权（`Reward = α × accuracy − β × latency`）奖励最大化。。


# 一、系统概述

## 整体思路

一个 DNN 推理请求不必在单台设备上完成。DSCI Testbed 将模型拆分成连续的原子 Segment，分配给三级节点执行：

```textile
Device 执行 [0, b1)  →  Edge 执行 [b1, b2)  →  Cloud 执行 [b2, final)
```

在中间边界处插入"早退出口"：如果分类置信度已经足够高，就提前返回结果，不再继续传输和计算。

Scheduler 的任务是为每个用户找到最优的 `(b1, b2, exit_thresholds)`，在精度和端到端时延之间取得平衡。

## 三阶段架构

| 阶段 | 位置 | 做什么 |
|------|------|--------|
| **Phase 1** 离线准备 | 开发机 | 训练模型 → 生成 Manifest → Profile 各节点时延 |
| **Phase 2** 调度服务 | 在线 | 收集节点状态 → 运行优化器 → 下发切分决策 |
| **Phase 3** 运行时 | 在线 | Device/Edge/Cloud 各自执行分配到的 Segment |
```textile
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 2: Scheduler                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  AlgoService │  │   Round      │  │ DSCI (PPO) / GA / BF │ │
│  │  (决策缓存)   │◄─┤ Coordinator  │──┤     Optimizer         │ │
│  └──────┬───────┘  │ (多设备同步)  │  └───────────────────────┘ │
│         │          └──────────────┘                              │
│   Flask API Server (:8000)                                      │
└─────────┬───────────────────────────────────────────────────────┘
          │ HTTP (REST)
          │
┌─────────┴───────────────────────────────────────────────────────┐
│                     Phase 3: Runtime                            │
│                                                                 │
│  ┌──────────┐    socket/pickle    ┌──────────┐   socket/pickle  │
│  │  Device   │──────────────────►│   Edge    │────────────────►│
│  │ (端侧)    │◄──────────────────│  (边缘)   │◄────────────────│
│  │ :no port  │    response        │ :9001     │   response      │
│  └──────────┘                    │ :9002 HTTP │                 │
│                                  └──────────┘                  │
│                                                                │
│                                  ┌──────────┐                  │
│                                  │  Cloud   │                  │
│                                  │ (云端)    │                  │
│                                  │ :32266    │                  │
│                                  │ :32265 HTTP                 │
│                                  └──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```
## 通信流程

```textile
┌─────────┐              ┌───────────┐          ┌────────────┐       ┌─────────┐
│ Device  │              │ Scheduler │          │    Edge    │       │  Cloud  │
│ (×N台)  │              │  (:8000)  │          │(:9001/9002)│       │(:32265/6)│
└────┬────┘              └─────┬─────┘          └────┬───────┘       └────┬────┘
     │                         │                     │                  │
     │  1. iperf3 测带宽 ─────────────────────────►  │                  │
     │                         │                     │                  │
     │  2. POST /register ───►│                     │                  │
     │     (user_id, 设备状态)  │                     │                  │
     │                         │                     │                  │
     │  3. 心跳 POST ────────►│                     │                  │
     │                         │                     │                  │
     │                         │── 4. 全部到齐 ──────│                  │
     │                         │   GET /status ────►│                  │
     │                         │   GET /status ───────────────────────►│
     │                         │                     │                  │
     │                         │── 5. 运行优化器      │                  │
     │                         │   计算 (b1,b2,Y)    │                  │
     │                         │                     │                  │
     │  6. GET /decisions ───►│                     │                  │
     │     ← {b1,b2,thresholds}                     │                  │
     │                         │                     │                  │
     │ ── 7. 本地执行 [0,b1) ─│─────────────────────│──────────────────│
     │                         │                     │                  │
     │  8. socket: 中间张量 ─────────────────────►  │                  │
     │                         │                     │── 9. [b1,b2)     │
     │                         │                     │                  │
     │                         │                     │  10. 转发 ──────►│
     │                         │                     │                  │── 11. [b2,final)
     │                         │                     │  ◄── response ──│
     │  ◄── response ──────────────────────────────  │                  │
     │                         │                     │                  │
     │  12. POST /measurements►│                     │                  │
     │                         │── 13. 计算 Reward    │                  │
```

> **协议说明**：Device↔Scheduler 使用 HTTP REST；Device→Edge→Cloud 的张量传输使用 TCP Socket（4 字节大端长度头 + pickle 序列化）。

## 模型与早退

模型为 Multi-Exit ResNet，在 ResNet 残差层之后插入早退分类头：

| 架构 | 残差块分布 | Segments | Boundaries | 早退出口 |
|------|-----------|----------|------------|---------|
| ResNet-18 | (2,2,2,2) | 11 | 12 | after_layer2, after_layer3 |
| ResNet-50 | (3,4,6,3) | 19 | 20 | after_layer2, after_layer3 |
| ResNet-101 | (3,4,23,3) | 32 | 33 | layer3 内 5 个出口 |

推理时执行到早退边界 → 运行 exit head → 若 `confidence ≥ threshold` 则提前返回预测结果，不再传输后续张量。

## 内置模型包

通过 `--bundle-id` 或 `$env:DSCI_BUNDLE_ID` 选择：

| Bundle ID | 架构 | 数据集 | 类别数 |
|-----------|------|--------|-------|
| `resnet18-cifar10-ee-v1` | ResNet-18 | CIFAR-10 | 10 |
| `resnet50-cifar10-ee-v1` | ResNet-50 | CIFAR-10 | 10 |
| `resnet101-cifar10-ee-v1` | ResNet-101 | CIFAR-10 | 10 |
| `resnet18-imagenet100-ee-v1` | ResNet-18 | ImageNet-100 | 100 |
| `resnet50-imagenet100-ee-v1` | ResNet-50 | ImageNet-100 | 100 |
| `resnet101-imagenet100-ee-v1` | ResNet-101 | ImageNet-100 | 100 |

---

# 二、运行指南

## 2.1 可配置项总览

运行本系统时，需要通过**环境变量**和**命令行参数**指定以下内容：

| 需要指定的内容 | 环境变量 | 命令行参数 | 说明 |
|--------------|---------|-----------|------|
| 模型包 | `DSCI_BUNDLE_ID` | `--bundle-id` | 选择哪个模型，如 `resnet50-cifar10-ee-v1`。CLI 优先于环境变量 |
| 推理后端 | — | `--backend pytorch\|mnn` | 选择 PyTorch 或 MNN 执行推理 |
| Device Profile | `DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID` | — | Device 端的 Segment 时延 Profile |
| Edge Profile | `DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID` | — | Edge 端的 Segment 时延 Profile |
| Cloud Profile | `DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID` | — | Cloud 端的 Segment 时延 Profile |
| 用户数量 | `DSCI_EXPECTED_USERS` | `--expected-users` | Scheduler 等待多少台 Device 到齐 |
| 轮次 ID | — | `--round-id` | Device 注册时指定，同一批次用相同 ID |
| 用户 ID | — | `--user-id` | 每台 Device 的唯一整数标识 |

> MNN 后端的 Profile 使用 `DSCI_{DEVICE|EDGE|CLOUD}_MNN_SEGMENT_PROFILE_ID`。
> 完整环境变量列表见[第三部分「环境变量参考」](#环境变量参考)。

## 2.2 目录结构

```textile
DSCI_testbed/
├── Data/                         # 数据（不入 Git）
│   ├── Datasets/                     # 共享原始数据集
│   │   ├── CIFAR10/                  # torchvision 自动下载
│   │   └── ImageNet100/              # 手动准备，需 100 类 train/val
│   │       ├── train/<100 class dirs>/
│   │       └── val/<100 class dirs>/
│   │
│   ├── Bundles/<bundle_id>/          # 模型包
│   │   ├── weights.pth               # 训练好的模型权重
│   │   ├── manifest.json             # Partition Manifest
│   │   ├── exit_curves.csv           # 早退精度/比率查找表
│   │   ├── layer_stats.csv           # 仅仿真模式需要
│   │   ├── analysis/                 # 模型分析结果
│   │   └── mnn_segments/             # MNN 格式的 segment 文件
│   │
│   ├── Profiles/
│   │   ├── Compute/                  # 仿真算力 Profile (theta, equivalent_flops)
│   │   └── Segments/                 # 真机分段执行 Profile (calibrated latency)
│   │       └── <profile_id>/
│   │           ├── metadata.json     # worker_count, threads_per_worker 等
│   │           └── segments.csv      # 逐 segment 实测时延
│   │
│   └── Runtime/                      # 运行产物（可丢弃）
│       └── SolutionCache/            # DSCI 训练的最优解缓存
├── Scripts/                      # 论文实验脚本
├── Src/
│   ├── Phase1_Offline/           # 离线准备
│   │   ├── Training/             # 模型训练和早退头微调
│   │   ├── Profiling/            # Manifest 生成、Segment 分析、MNN 导出
│   │   └── LookupTables/         # 早退曲线生成
│   │
│   ├── Phase2_Scheduler/         # 在线调度
│   │   ├── Service/              # Flask API、Round Coordinator、Decision 编解码
│   │   ├── Optimizer/            # DSCI(PPO)、GA、BF 三种优化器
│   │   ├── Objective/            # 目标函数（时延计算、早退概率、总奖励）
│   │   └── Utils/                # 数据解析工具
│   │
│   ├── Phase3_Runtime/           # 在线运行时
│   │   ├── Device/               # Device 端推理客户端
│   │   ├── Edge/                 # Edge 端推理服务
│   │   ├── Cloud/                # Cloud 端推理服务
│   │   └── Shared/               # Worker Pool、Socket Server、Identity 校验
│   │
│   └── Shared/                   # 跨 Phase 共享模块
│       ├── Config/               # 部署拓扑、模型包规格、路径
│       ├── Data/                 # 数据集加载
│       ├── Models/               # Multi-Exit ResNet 实现
│       ├── Partitioning/         # Manifest、PyTorch Segment Executor
│       └── Profiles/             # Compute Profile / Segment Profile
└── Ducuments/                    # 适配文档
```

## 2.3 Phase 1：离线准备

在开始在线推理前，需要按顺序完成以下步骤：

```powershell
$bundle = "resnet50-cifar10-ee-v1"

# ① 训练主模型
python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle

# ② 微调早退头（冻结 backbone，只训练 exit_heads）
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle

# ③ 生成 Partition Manifest（模型切分映射 + 边界序列化字节数）
python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite

# ④ 生成早退曲线（阈值 → 精度/早退率 查找表）
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --overwrite

# ⑤ 在目标硬件上 Profile 逐 Segment 推理时延
python -m Src.Phase1_Offline.Profiling.profile_segments device-pytorch --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments edge-pytorch   --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments cloud-pytorch  --bundle-id $bundle
```

> **何时需要重新执行**：
> 模型权重变化 → 重新执行 3-6；
> Worker 配置变化 → 重新执行 5。


## 2.4 Phase 2 + 3：在线推理

在线部分需要启动 4 类进程，**按以下顺序**依次在不同终端中启动：

### 设置环境变量

```powershell
$env:DSCI_BUNDLE_ID = "resnet50-cifar10-ee-v1"
$env:DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID = "device-pytorch"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID   = "edge-pytorch"
$env:DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID  = "cloud-pytorch"
```

### 步骤 1：启动 Cloud

```powershell
python -m Src.Phase3_Runtime.Cloud.run_cloud --backend pytorch
```
加载 Segment Profile → 创建固定 Worker Pool → 监听 TCP `:32266`（推理）+ HTTP `:32265`（状态查询）。

### 步骤 2：启动 Edge

```powershell
python -m Src.Phase3_Runtime.Edge.run_edge --backend pytorch
```
同 Cloud，监听 TCP `:9001` + HTTP `:9002`。Edge 若需要转发给 Cloud，会连接 Cloud 的 `:32266`。

### 步骤 3：启动 Scheduler

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2
```
监听 HTTP `:8000`。启动时 **不会** 连接 Edge/Cloud，而是等到所有 Device 注册到齐后，才去查询 Edge/Cloud 的 `/status`。

### 步骤 4：启动 Device（每台一个进程）

```powershell
# 终端 A
python -m Src.Phase3_Runtime.Device.run_device --backend pytorch --round-id round-001 --user-id 0

# 终端 B
python -m Src.Phase3_Runtime.Device.run_device --backend pytorch --round-id round-001 --user-id 1
```

Device 启动后会自动完成：iperf3 测带宽 → 注册到 Scheduler → 心跳 → 等待决策 → 执行推理循环 → 提交测量结果。

### 轮次规则

- 同一 `round_id` 的 Device 组成一个批次
- Scheduler 等齐后只做一次联合优化
- 一次只允许一个活跃轮次，下一轮须用新 `round_id`

---

# 三、进阶选项

## API 版本

Scheduler 同时暴露 v1 和 v2 两套接口：

| 接口 | 适用场景 | 说明 |
|------|---------|------|
| **v1** | 单次决策 / 脚本调用 | 直接 POST 完整 State JSON 到 `/api/v1/decision`，立即返回决策 |
| **v2** | 多设备轮次协调 | Device 注册 → 心跳 → 屏障同步 → 按 user_id 领取决策 |

v1 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 服务健康状态 |
| POST | `/api/v1/decision` | 提交 State，获取决策 |
| POST | `/api/v1/measurements` | 提交测量结果 |

v2 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v2/rounds/{round_id}/devices/register` | 设备注册 |
| POST | `/api/v2/rounds/{round_id}/devices/{user_id}/heartbeat` | 心跳 |
| GET | `/api/v2/rounds/{round_id}/decisions/{user_id}` | 轮询决策（202=等待, 200=就绪） |
| POST | `/api/v2/rounds/{round_id}/measurements/{user_id}` | 提交测量 |
| GET | `/api/v2/rounds/{round_id}/status` | 轮次状态 |

## 决策模式

Scheduler 启动时可指定固定策略，跳过优化器直接返回预设决策：

```powershell
# 固定切分点（所有用户使用 b1=3, b2=10）
python -m Src.Phase2_Scheduler.Service.api_server --fixed-split 3 10

# 固定早退阈值
python -m Src.Phase2_Scheduler.Service.api_server --fixed-threshold 0.7

# 禁用后台 DSCI 训练
python -m Src.Phase2_Scheduler.Service.api_server --no-auto-train
```

也可在 State JSON 中通过 `decision_mode` 字段动态切换：

| 值 | 说明 |
|----|------|
| `dsci` / `auto` | 使用 DSCI (PPO) 优化（默认） |
| `device` | 全部在 Device 执行 |
| `edge` | 全部在 Edge 执行 |
| `cloud` | 全部在 Cloud 执行 |
| `device_early_exit` | Device 执行 + 启用早退 |
| `edge_early_exit` | Edge 执行 + 启用早退 |

## 推理后端

Edge/Cloud/Device 支持两种推理后端：

| 后端 | 参数 | 说明 |
|------|------|------|
| PyTorch | `--backend pytorch` | 默认，直接使用 PyTorch 执行 Segment |
| MNN | `--backend mnn` | 移动端轻量后端，需先用 `export_mnn_segments` 导出 |

## 双模式时延计算

Scheduler 在优化时有两种时延估算模式：

| 模式 | `resource_mode` 值 | 时延公式 | 适用场景 |
|------|-------------------|---------|---------|
| 真机模式 | `fixed_worker_pool` | `T = Σ calibrated_latency[segment]` | 真机部署（默认） |
| 仿真模式 | `simulation_resource_mode` | `T = FLOPs / throughput` | 论文仿真实验 |

## 端口与网络配置

默认端口（定义在 `Src/Shared/Config/deploy_config.py`）：

| 服务 | 端口 | 协议 | 用途 |
|------|------|------|------|
| Scheduler API | 8000 | HTTP | 注册 / 决策 / 测量 |
| Edge 推理 | 9001 | TCP Socket | 中间张量传输 |
| Edge 状态 | 9002 | HTTP | `/status` 查询 |
| Edge iperf | 5001 | iperf3 | 带宽测量 |
| Cloud 推理 | 32266 | TCP Socket | 中间张量传输 |
| Cloud 状态 | 32265 | HTTP | `/status` 查询 |
| Cloud iperf | 32264 | iperf3 | 带宽测量 |

当前默认所有主机为 `127.0.0.1`（本地测试）。真机部署需修改 `deploy_config.py` 中的 IP 地址。

## 环境变量参考

| 变量 | 说明 |
|------|------|
| `DSCI_BUNDLE_ID` | 模型包 ID |
| `DSCI_EXPECTED_USERS` | 预期 Device 数量 |
| `DSCI_{DEVICE\|EDGE\|CLOUD}_PYTORCH_SEGMENT_PROFILE_ID` | PyTorch 后端的 Profile ID |
| `DSCI_{DEVICE\|EDGE\|CLOUD}_MNN_SEGMENT_PROFILE_ID` | MNN 后端的 Profile ID |
| `DSCI_SEGMENT_PROFILE_ROOT` | 自定义 Profile 存储路径 |
| `DSCI_{EDGE\|CLOUD}_PROTOCOL_OVERHEAD_S` | 协议额外时延（秒） |

## 近期真机适配更新

以下内容用于 Device/Edge/Cloud 三端真机测试，尤其是慢速远程链路或 Tailscale 环境：

| 项目 | 用法 | 说明 |
|------|------|------|
| Device 带宽覆盖 | `--override-bw-d2e <Mbps>` | 跳过 iperf3，手动指定 Device -> Edge 带宽 |
| Edge 带宽覆盖 | `--override-bw-e2c <Mbps>` | Edge 状态上报时手动指定 Edge -> Cloud 带宽 |
| iperf 时长 | `--iperf-duration <seconds>` / `DSCI_IPERF_DURATION_S` | 默认 8 秒；远程链路建议 8-10 秒以上 |
| iperf 超时 | `--iperf-timeout <seconds>` / `DSCI_IPERF_TIMEOUT_MARGIN_S` | 避免低速链路下 iperf 过早超时 |
| iperf 路径 | `DSCI_IPERF_EXE` | 指定自定义 iperf3 可执行文件 |
| 固定策略测试 | `--decision-mode device\|edge\|cloud\|device_early_exit\|edge_early_exit\|cloud_early_exit` | Device 注册时请求预设策略，便于做基线测试 |
| 目标函数权重 | `DSCI_OBJECTIVE_ALPHA` / `DSCI_OBJECTIVE_BETA` | 调整 `alpha * accuracy - beta * latency` 中精度和时延权重 |
| 协议额外时延 | `DSCI_{DEVICE\|EDGE\|CLOUD}_PROTOCOL_OVERHEAD_S` | 传入 fixed-worker 延迟模型，用于校准真机协议开销 |

真机运行时 Device 会打印 `Decision summary`，其中包含：

```text
source=<decision_source>, objective=<objective>, b1=<partition_boundary_1>, b2=<partition_boundary_2>
```

常见 `decision_source`：

| 来源 | 含义 |
|------|------|
| `default` | 当前轮先返回默认解，后台启动 DSCI/PPO 训练 |
| `cached_dsci:exact` | 当前状态与缓存完全匹配，直接复用策略 |
| `cached_dsci:reuse:<distance>` | 状态距离较近，复用历史策略 |
| `cached_dsci:warm:<distance>` | 先返回 warm-start 策略，同时后台继续训练 |
| `fixed_split:<b1>:<b2>` | Scheduler 使用固定切分点 |
| `preset:<mode>:...` | Device 请求了预设策略 |

Scheduler 会把 PPO 训练事件写入：

```text
Data/Runtime/SolutionCache/training_events.jsonl
```

`GET /api/v1/health` 可查看 `training_status`、`update_epochs`、`last_training_mode`、`last_training_duration_s`、`training_events_path` 等字段，用于判断新策略何时训练完成。

真机 socket 传输已改为“4 字节大端长度头 + pickle payload”的双向长度前缀协议。Device 汇总输出中会包含 `T_device_edge_roundtrip_avg_ms`，用于观察 Device -> Edge 的真实传输/等待开销；当早退在 Device 侧发生时，该字段可能不会出现。

DSCI/PPO 的早退概率模型已修正：早退表给出的是各 early-exit head 的条件退出概率，未早退的剩余概率必须进入 final classifier。这个修正会显著影响慢网下的大 tensor offload 策略选择，避免把 final 样本误当成已经早退。

## 实验脚本

`Scripts/` 目录包含论文实验复现脚本：

| 目录 | 实验 |
|------|------|
| `Exp0_Motivation` | 动机实验 |
| `Exp1_Baseline` | 基线对比 |
| `Exp2_Scalable` | 可扩展性 |
| `Exp3_Ablation` | 消融实验 |
| `Exp4_Convergency_and_Overhead` | 收敛性与开销 |
| `Exp5_ParaSensitivity` | 参数敏感性 |

## 验证

```powershell
python -m compileall Src Scripts Tests -q   # 语法检查
python -m unittest discover -s Tests -v      # 单元测试
```
