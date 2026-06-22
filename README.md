# DSCI Testbed

**Distributed Split-Computing Inference** 原型系统，实现 Device → Edge → Cloud 三级协同推理。

系统核心思想：将一个带有早退出口的 ResNet 模型按原子 Segment 切分，由 PPO 强化学习
优化器（DSCI）为每个用户决定最优的切分点和早退阈值，使延迟与精度的加权奖励最大化。

---

## 目录

- [系统架构](#系统架构)
- [三阶段工作流](#三阶段工作流)
- [启动顺序与通信流程](#启动顺序与通信流程)
- [目录结构](#目录结构)
- [Data 目录](#data-目录)
- [模型包](#模型包)
- [Phase 1 离线准备](#phase-1-离线准备)
- [Phase 2 调度服务](#phase-2-调度服务)
- [Phase 3 运行时](#phase-3-运行时)
- [在线启动步骤](#在线启动步骤)
- [端口与协议](#端口与协议)
- [环境变量](#环境变量)
- [实验脚本](#实验脚本)
- [验证](#验证)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 2: Scheduler                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  AlgoService  │  │   Round      │  │ DSCI (PPO) / GA / BF │ │
│  │  (决策缓存)   │◄─┤ Coordinator  │──┤     Optimizer         │ │
│  └──────┬───────┘  │ (多设备同步)   │  └───────────────────────┘ │
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

---

## 三阶段工作流

### Phase 1: 离线准备

在任何真机部署之前，先在开发机上完成：

1. **训练模型** — 训练 Multi-Exit ResNet（带 `after_layer2` / `after_layer3` 两个早退头）
2. **生成 Partition Manifest** — 遍历模型的原子模块，建立 Boundary 和 Segment 映射，
   记录每个边界的 pickle 序列化字节数
3. **生成早退曲线** — 在验证集上遍历所有早退阈值组合，记录精度/早退率
4. **Profile 各节点** — 在目标硬件上逐 Segment 采样推理时延，写入 Segment Profile

### Phase 2: 调度服务（在线）

Scheduler 是系统的大脑。启动后：

1. 监听 HTTP 端口，等待 Device 注册
2. 收到所有 Device 注册后，向 Edge 和 Cloud 查询 `/status` 获取节点状态
3. 汇总全部状态信息，构建 `Paras` 参数对象
4. 调用 DSCI（PPO）/GA/BF 优化器，为每个用户计算最优 `(切分点1, 切分点2, 早退阈值)`
5. 将决策编码为 JSON，每个 Device 按 `user_id` 轮询领取
6. 收集推理结果，计算 Reward（`R = α × accuracy - β × T_total`）

### Phase 3: 运行时（在线）

Device、Edge、Cloud 三个节点各自运行推理服务：

1. **Device** 执行模型的 `[0, b1)` Segment → 如果早退成功则直接返回结果，
   否则通过 socket 将中间张量发送到 Edge
2. **Edge** 执行 `[b1, b2)` Segment → 如果早退成功则返回，否则将张量转发到 Cloud
3. **Cloud** 执行 `[b2, final)` Segment → 返回最终预测结果

---

## 启动顺序与通信流程

### 启动顺序（必须按此顺序）

```
1. Cloud   →  监听 socket(:32266) 和 HTTP status(:32265)
2. Edge    →  监听 socket(:9001) 和 HTTP status(:9002)
3. Scheduler →  监听 HTTP API(:8000)，连接 Edge/Cloud 的 status 端口
4. Device(s) →  主动连接 Scheduler 注册，然后开始推理循环
```

### 详细通信流程

```
┌─────────┐                ┌───────────┐            ┌─────────┐        ┌─────────┐
│ Device 0│                │ Scheduler │            │  Edge   │        │  Cloud  │
│ Device 1│                │  (:8000)  │            │ (:9001) │        │(:32266) │
└────┬────┘                └─────┬─────┘            └────┬────┘        └────┬────┘
     │                           │                       │                  │
     │  1. iperf3 测带宽          │                       │                  │
     │──────────────────────────────────────────────────►│                  │
     │  BW_d2e = XX Mbps          │                       │                  │
     │◄──────────────────────────────────────────────────│                  │
     │                           │                       │                  │
     │  2. POST /register        │                       │                  │
     │  {user_id, device_state}  │                       │                  │
     │─────────────────────────►│                       │                  │
     │   {"status": "registered"}│                       │                  │
     │◄─────────────────────────│                       │                  │
     │                           │                       │                  │
     │  3. 定时心跳               │                       │                  │
     │  POST /heartbeat          │                       │                  │
     │─────────────────────────►│                       │                  │
     │                           │                       │                  │
     │                           │  4. 所有 Device 到齐   │                  │
     │                           │  GET /status          │                  │
     │                           │─────────────────────►│                  │
     │                           │  {edge_state}        │                  │
     │                           │◄─────────────────────│                  │
     │                           │  GET /status                            │
     │                           │────────────────────────────────────────►│
     │                           │  {cloud_state}                          │
     │                           │◄────────────────────────────────────────│
     │                           │                       │                  │
     │                           │  5. 运行 DSCI 优化器   │                  │
     │                           │  → 计算 (b1, b2, Y)   │                  │
     │                           │                       │                  │
     │  6. GET /decisions/{uid}  │                       │                  │
     │─────────────────────────►│                       │                  │
     │   {b1, b2, thresholds}   │                       │                  │
     │◄─────────────────────────│                       │                  │
     │                           │                       │                  │
     │  7. 执行 [0, b1) segments │                       │                  │
     │  (Device 本地推理)         │                       │                  │
     │                           │                       │                  │
     │  8. socket: 发送中间张量   │                       │                  │
     │──────────────────────────────────────────────────►│                  │
     │                           │                       │ 9. 执行 [b1,b2)  │
     │                           │                       │    segments      │
     │                           │                       │                  │
     │                           │                       │ 10. socket 转发   │
     │                           │                       │─────────────────►│
     │                           │                       │                  │ 11. 执行
     │                           │                       │                  │ [b2,final)
     │                           │                       │  response        │
     │                           │                       │◄─────────────────│
     │  response (prediction)    │                       │                  │
     │◄──────────────────────────────────────────────────│                  │
     │                           │                       │                  │
     │  12. POST /measurements   │                       │                  │
     │  {T_total, is_correct}    │                       │                  │
     │─────────────────────────►│                       │                  │
     │                           │  13. 计算 Reward       │                  │
```

---

## 目录结构

```
DSCI_testbed/
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
│
├── Scripts/                      # 实验脚本 (Exp0-5)
├── Tests/                        # 单元测试
├── Data/                         # 数据资产（不提交到 Git）
└── Ducuments/                    # Bug 适配文档
```

---

## Data 目录

```
Data/
├── Datasets/                     # 共享原始数据集
│   ├── CIFAR10/                  # torchvision 自动下载
│   └── ImageNet100/              # 手动准备，需 100 类 train/val
│       ├── train/<100 class dirs>/
│       └── val/<100 class dirs>/
│
├── Bundles/<bundle_id>/          # 模型包
│   ├── weights.pth               # 训练好的模型权重
│   ├── manifest.json             # Partition Manifest
│   ├── exit_curves.csv           # 早退精度/比率查找表
│   ├── layer_stats.csv           # 仅仿真模式需要
│   ├── analysis/                 # 模型分析结果
│   └── mnn_segments/             # MNN 格式的 segment 文件
│
├── Profiles/
│   ├── Compute/                  # 仿真算力 Profile (theta, equivalent_flops)
│   └── Segments/                 # 真机分段执行 Profile (calibrated latency)
│       └── <profile_id>/
│           ├── metadata.json     # worker_count, threads_per_worker 等
│           └── segments.csv      # 逐 segment 实测时延
│
├── Runtime/                      # 运行产物（可丢弃）
│   └── SolutionCache/            # DSCI 训练的最优解缓存
│
└── Archive/                      # 历史记录，活动代码不读取
```

---

## 模型包

内置 4 个模型包，通过 `--bundle-id` 或环境变量 `DSCI_BUNDLE_ID` 选择：

| Bundle ID | 架构 | 数据集 | 类别数 | 早退出口 |
|-----------|------|--------|-------|---------|
| `resnet18-cifar10-ee-v1` | ResNet-18 (BasicBlock) | CIFAR-10 | 10 | after_layer2, after_layer3 |
| `resnet50-cifar10-ee-v1` | ResNet-50 (Bottleneck) | CIFAR-10 | 10 | after_layer2, after_layer3 |
| `resnet18-imagenet100-ee-v1` | ResNet-18 | ImageNet-100 | 100 | after_layer2, after_layer3 |
| `resnet50-imagenet100-ee-v1` | ResNet-50 | ImageNet-100 | 100 | after_layer2, after_layer3 |

Manifest 分段数：

| 架构 | Segments | Boundaries | 来源 |
|------|----------|------------|------|
| ResNet-18 | 11 | 12 | stem + 8 blocks + final_pool + fc |
| ResNet-50 | 19 | 20 | stem + 16 blocks + final_pool + fc |

模型包**仅在进程启动时切换**，不支持请求级热切换。

---

## Phase 1 离线准备

按以下顺序执行：

```powershell
$bundle = "resnet18-cifar10-ee-v1"

# 1. 训练主模型 (backbone + final classifier)
python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle

# 2. 微调早退头 (冻结 backbone，只训练 exit_heads)
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle

# 3. 生成 Partition Manifest (遍历模型模块，记录边界序列化字节)
python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite

# 4. 生成早退曲线 (阈值 → 精度/早退率 查找表)
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --overwrite

# 5. 在目标硬件上 Profile 各节点 Segment 时延
python -m Src.Phase1_Offline.Profiling.profile_segments device-pytorch --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments edge-pytorch   --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments cloud-pytorch  --bundle-id $bundle

# 6. (可选) 导出 MNN 格式 Segment，用于移动端部署
python -m Src.Phase1_Offline.Profiling.export_mnn_segments --bundle-id $bundle
```

> **何时需要重新执行**：模型权重变化 → 重新执行 3-6；Worker 配置变化 → 重新执行 5。

---

## Phase 2 调度服务

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| API Server | `api_server.py` | Flask HTTP 网关，暴露 v1/v2 两套接口 |
| AlgoService | `algo_service.py` | 决策缓存 + 后台 DSCI 训练 |
| RoundCoordinator | `round_coordinator.py` | 多设备注册、心跳、屏障同步 |
| Decision Codec | `decision_codec.py` | 将优化器的 `(X, Y, F_e, F_c)` 编码为 JSON |
| Paras | `paras.py` | 所有优化参数的统一入口 |
| DSCI Optimizer | `Optimizer/DSCI/` | PPO 强化学习优化器 |
| GA Optimizer | `Optimizer/GA/` | 遗传算法优化器 |
| BF Optimizer | `Optimizer/BF/` | 暴力穷举优化器 |

### 目标函数

```
Reward = α × accuracy - β × T_total

T_total = T_compute_device + T_d2e + T_compute_edge + T_e2c + T_compute_cloud
```

其中 `α=1.0, β=5.0` 为默认权重。

### API 端点

**v1 接口**（直接决策，单轮次）：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 服务健康状态 |
| POST | `/api/v1/decision` | 直接提交完整 State，获取决策 |
| POST | `/api/v1/measurements` | 提交测量结果 |

**v2 接口**（多设备轮次协调）：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v2/rounds/{round_id}/devices/register` | 设备注册 |
| POST | `/api/v2/rounds/{round_id}/devices/{user_id}/heartbeat` | 心跳 |
| GET | `/api/v2/rounds/{round_id}/decisions/{user_id}` | 轮询决策 (202=未就绪, 200=已就绪) |
| POST | `/api/v2/rounds/{round_id}/measurements/{user_id}` | 提交测量 |
| GET | `/api/v2/rounds/{round_id}/status` | 轮次状态 |

### 决策模式

可通过 `--fixed-split` / `--fixed-threshold` 或 State JSON 中的 `decision_mode` 切换：

| 模式 | 说明 |
|------|------|
| `dsci` / `auto` | DSCI (PPO) 后台训练，使用最新缓存结果 |
| `device` | 全 Device 执行 |
| `edge` | 全 Edge 执行 |
| `cloud` | 全 Cloud 执行 |
| `device_early_exit` | Device 执行 + 早退 |
| `edge_early_exit` | Edge 执行 + 早退 |
| `--fixed-split 3 10` | 固定切分点 |

---

## Phase 3 运行时

### Edge 服务

启动后开启两个监听：

1. **TCP Socket** (`:9001`) — 接收中间张量，执行 Segment 推理，返回结果
2. **HTTP Flask** (`:9002`) — `/status` 端点供 Scheduler 查询节点状态

使用 `FixedWorkerPool`（ProcessPoolExecutor + BoundedSemaphore）管理推理进程，
每个 Worker 进程启动时固定 `OMP_NUM_THREADS`，运行期间不动态修改。

### Cloud 服务

结构与 Edge 相同：

1. **TCP Socket** (`:32266`) — 接收 Edge 转发的中间张量
2. **HTTP Flask** (`:32265`) — `/status` 端点

### Device 客户端

Device 不监听任何端口，它是主动发起连接的客户端：

1. 用 `iperf3` 测量到 Edge 的带宽 (`BW_d2e`)
2. 向 Scheduler 注册自身状态
3. 开启心跳线程
4. 轮询等待决策
5. 收到决策后，循环执行推理：
   - 本地执行 `[0, b1)` Segment
   - 如果早退成功 → 直接记录结果
   - 否则通过 socket 将中间张量发送到 Edge (`:9001`)
   - 收到最终结果后计算 `T_total`
6. 全部样本完成后，汇总测量结果提交给 Scheduler

### 通信协议

Edge/Cloud 的 TCP Socket 使用**长度前缀 + pickle 序列化**：

```
发送：[4 字节大端长度头] + [pickle 序列化的 payload dict]
接收：[pickle 序列化的 response dict]（读到 EOF 为止）
```

每个连接处理一次请求后关闭，并发由 per-connection 线程处理。

---

## 在线启动步骤

### 前提条件

- 所有节点选择相同的 `bundle_id`
- 每个节点有匹配的 Segment Profile

### 设置环境变量

```powershell
$env:DSCI_BUNDLE_ID = "resnet50-cifar10-ee-v1"
$env:DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID = "device-pytorch"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID   = "edge-pytorch"
$env:DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID  = "cloud-pytorch"
```

### 按顺序启动（4 个终端）

```powershell
# 终端 1: Cloud (必须最先启动)
python -m Src.Phase3_Runtime.Cloud.run_cloud --backend pytorch

# 终端 2: Edge (Cloud 启动后)
python -m Src.Phase3_Runtime.Edge.run_edge --backend pytorch

# 终端 3: Scheduler (Edge/Cloud 启动后)
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2

# 终端 4+: Device(s) (Scheduler 启动后)
python -m Src.Phase3_Runtime.Device.run_device --backend pytorch --round-id round-001 --user-id 0
python -m Src.Phase3_Runtime.Device.run_device --backend pytorch --round-id round-001 --user-id 1
```

### 轮次规则

- 所有 Device 使用相同 `round_id` 注册
- Scheduler 等待 `--expected-users` 台 Device 到齐后**仅执行一次联合优化**
- 每台 Device 按自己的 `user_id` 轮询领取决策
- 同一时间只允许一个活跃轮次
- 下一轮必须使用新的 `round_id`

---

## 端口与协议

| 服务 | 端口 | 协议 | 方向 | 用途 |
|------|------|------|------|------|
| Cloud Feature | 32266 | TCP Socket (pickle) | Edge → Cloud | 中间张量传输 |
| Cloud Status | 32265 | HTTP (Flask) | Scheduler → Cloud | 节点状态查询 |
| Edge Feature | 9001 | TCP Socket (pickle) | Device → Edge | 中间张量传输 |
| Edge Status | 9002 | HTTP (Flask) | Scheduler → Edge | 节点状态查询 |
| Edge iperf | 5001 | iperf3 | Device → Edge | 带宽测量 |
| Cloud iperf | 32264 | iperf3 | Edge → Cloud | 带宽测量 |
| Scheduler API | 8000 | HTTP (Flask) | Device → Scheduler | 注册/决策/测量 |

> 端口默认值定义在 `Src/Shared/Config/deploy_config.py`，当前本地测试模式所有
> 主机为 `127.0.0.1`。真机部署时修改该文件中的 IP 地址。

---

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `DSCI_BUNDLE_ID` | 模型包 ID | `resnet50-cifar10-ee-v1` |
| `DSCI_EXPECTED_USERS` | 预期 Device 数量 | `2` |
| `DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID` | Device 的 Segment Profile ID | `device-pytorch` |
| `DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID` | Edge 的 Segment Profile ID | `edge-pytorch` |
| `DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID` | Cloud 的 Segment Profile ID | `cloud-pytorch` |
| `DSCI_DEVICE_MNN_SEGMENT_PROFILE_ID` | MNN 后端的 Device Profile | `device-mnn` |
| `DSCI_EDGE_MNN_SEGMENT_PROFILE_ID` | MNN 后端的 Edge Profile | `edge-mnn` |
| `DSCI_CLOUD_MNN_SEGMENT_PROFILE_ID` | MNN 后端的 Cloud Profile | `cloud-mnn` |
| `DSCI_SEGMENT_PROFILE_ROOT` | 自定义 Profile 存储路径 | `./my_profiles` |
| `DSCI_EDGE_PROTOCOL_OVERHEAD_S` | Edge 协议额外时延 (秒) | `0.001` |
| `DSCI_CLOUD_PROTOCOL_OVERHEAD_S` | Cloud 协议额外时延 (秒) | `0.001` |

---

## 实验脚本

`Scripts/` 目录包含论文实验复现脚本：

| 目录 | 实验 |
|------|------|
| `Exp0_Motivation` | 动机实验 |
| `Exp1_Baseline` | 基线对比实验 |
| `Exp2_Scalable` | 可扩展性实验 |
| `Exp3_Ablation` | 消融实验 |
| `Exp4_Convergency_and_Overhead` | 收敛性与开销实验 |
| `Exp5_ParaSensitivity` | 参数敏感性实验 |
| `Results/` | 实验结果输出 |

---

## 验证

```powershell
# 语法检查
python -m compileall Src Scripts Tests -q

# 单元测试
python -m unittest discover -s Tests -v
```

测试覆盖：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `test_round_coordinator.py` | 屏障同步、注册冲突、心跳超时、完整轮次流程、Reward 计算 |
| `test_runtime_identity.py` | v1/v2 决策选取、身份链校验 |
| `test_runtime_validation.py` | 边界合法性、bundle_id 校验 |

---

## 关键设计决策

### 双模式时延计算

- **仿真模式** (`simulation_resource_mode`): `T = C/f` (FLOPs ÷ 频率)，用于论文实验
- **真机模式** (`fixed_worker_pool`): `T = Σ calibrated_latency[segment]`，用于实际部署

### 固定 Worker Pool

Edge/Cloud 使用 `ProcessPoolExecutor` 启动固定数量的 Worker 进程，
每个 Worker 在启动时锁定 `OMP_NUM_THREADS`，运行期间不动态调整线程数。
提交队列使用 `BoundedSemaphore` 实现背压控制。

### 身份链 (Identity Chain)

每个推理请求携带 5 个身份字段贯穿 Device → Edge → Cloud：

```
round_id → user_id → request_id → decision_id → decision_version
```

返回时校验身份一致性，防止请求错乱。

### 早退机制

模型在 `layer2` 和 `layer3` 后设有早退出口。推理时：

1. 执行到早退边界 → 运行对应的 exit head → 获得 confidence
2. 如果 `confidence ≥ threshold` → 早退，直接返回预测结果
3. 否则继续执行后续 Segment
