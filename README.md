# DSCI Testbed

DSCI Testbed 是一个 Device、Edge、Cloud 协同推理原型。Device 收集实时状态并向调度服务请求决策，三个运行节点按照返回的切分点和早退阈值执行同一个多出口 ResNet50 的不同阶段。

当前版本只保留手动逐进程启动方式，不提供 PowerShell 一键启停脚本。

## 架构

```text
Src/
|-- Phase1_Offline/       # 训练、profiling、lookup table、离线评估
|-- Phase2_Scheduler/     # 调度 API、目标函数、优化器、调度数据模型
|-- Phase3_Runtime/       # Device、Edge、Cloud 在线推理
`-- Shared/               # 三个阶段均可复用的配置、模型、数据和 profile
```

三阶段依赖规则：

- `Shared` 不依赖任何 Phase。
- 每个 Phase 只能依赖自身与 `Shared`。
- `Scripts/` 是实验入口，不属于在线运行链路。

## 配置职责

| 文件 | 职责 |
| --- | --- |
| `Src/Shared/Config/deploy_config.py` | 节点地址、监听地址、端口和带宽 fallback |
| `Src/Shared/Config/model_config.py` | 模型静态元数据和模型产物命名规则，不包含文件路径 |
| `Src/Shared/Config/paths.py` | 项目级目录、模型产物路径和兼容文件解析 |
| `Src/Shared/Config/visualization.py` | 实验和报告共用的可视化常量 |
| `Src/Phase2_Scheduler/algo_config.py` | 目标函数权重与优化器超参数 |
| `Src/Phase2_Scheduler/paras.py` | 单轮调度问题的已校验输入数据 |

`Paras` 不属于 Shared。它将状态 JSON、设备 compute profile 和离线 lookup table 转换成目标函数与优化器需要的数组，因此只由 Phase 2 拥有。跨阶段共享的是状态 JSON、决策 JSON、模型配置和文件路径，不是 `Paras` 对象。

## 运行前准备

每台机器部署完整的 `Src/` 和 `Data/`。至少确认：

- `Data/Weights/resnet50_cifar10_multi_ee.pth` 存在。
- `Data/OfflineTables/` 中的 lookup table 完整。
- `Data/ComputeProfiles/` 包含当前 Device、Edge、Cloud 使用的 profile。
- `Src/Shared/Config/deploy_config.py` 中的节点 IP 和端口正确。
- `iperf3` 可直接从命令行执行；否则修改 `Src/Phase3_Runtime/Shared/bandwidth_iperf.py` 中的 `IPERF_EXE`。

为各运行节点设置与实际后端匹配的 profile ID。PowerShell 示例：

```powershell
$env:DSCI_DEVICE_PYTORCH_COMPUTE_PROFILE_ID="device-pytorch"
$env:DSCI_EDGE_PYTORCH_COMPUTE_PROFILE_ID="edge-pytorch"
$env:DSCI_CLOUD_PYTORCH_COMPUTE_PROFILE_ID="cloud-pytorch"
```

## 推荐启动顺序

从仓库根目录分别打开终端并按顺序启动。Cloud 和 Edge 必须在 Device 启动前就绪。

```text
1. Cloud:      iperf3 -s -p 32264
2. Cloud:      python -m Src.Phase3_Runtime.Cloud.run_cloud
3. Edge:       iperf3 -s -p 5001
4. Edge:       python -m Src.Phase3_Runtime.Edge.run_edge
5. Scheduler:  python -m Src.Phase2_Scheduler.Service.api_server
6. Device:     python -m Src.Phase3_Runtime.Device.run_device
```

调度 API 的测试覆盖参数：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --fixed-split 0 1
python -m Src.Phase2_Scheduler.Service.api_server --fixed-threshold 0.7
python -m Src.Phase2_Scheduler.Service.api_server --fixed-split 0 1 --fixed-threshold 0.7
```

Device 侧常用测试参数：

```powershell
python -m Src.Phase3_Runtime.Device.run_device --test-samples 100
python -m Src.Phase3_Runtime.Device.run_device --difficulty hard --test-samples 100
```

各进程当前使用前台阻塞方式运行。停止时在对应终端按 `Ctrl+C`。

## 端口表

| 端口 | 监听方 | 协议 | 用途 |
| --- | --- | --- | --- |
| `5001` | Edge | iperf3 | Device 测量 `BW_d2e` |
| `32264` | Cloud | iperf3 | Edge 测量 `BW_e2c` |
| `8000` | Scheduler | HTTP | `/api/v1/health`、`/api/v1/decision`、`/api/v1/measurements` |
| `9001` | Edge | TCP | 接收 Device 特征张量并返回最终结果 |
| `9002` | Edge | HTTP | `/status`，返回 Edge profile、计算能力和 `BW_e2c` |
| `32265` | Cloud | HTTP | `/status`，返回 Cloud profile 和计算能力 |
| `32266` | Cloud | TCP | 接收 Edge 转发的特征张量并返回最终结果 |

所有地址和端口由 `Src/Shared/Config/deploy_config.py` 统一定义。

## 在线数据流

```text
Device                         Edge                         Cloud
  |                              |                             |
  |-- iperf3 ------------------->| :5001                       |
  |   measure BW_d2e             |                             |
  |                              |-- iperf3 ------------------> | :32264
  |                              |   measure BW_e2c             |
  |-- GET /status -------------> | :9002                       |
  |-- GET /status --------------------------------------------> | :32265
  |                                                            |
  |-- POST state JSON --------> Scheduler :8000                 |
  |<-- decision JSON ---------- Scheduler :8000                 |
  |                                                            |
  |-- run model stage 0..s1                                    |
  |-- feature tensor ---------->| :9001                        |
  |                              |-- run stage s1+1..s2         |
  |                              |-- feature tensor ----------->| :32266
  |                              |                              |-- run stage s2+1..end
  |<----------------------------- final inference result -------|
```

Device 构造的状态 JSON 包含：

- Device、Edge、Cloud 的 `compute_profile_id` 和计算能力。
- Device 到 Edge、Edge 到 Cloud 的实测带宽。
- 模型名称与用户列表。

Scheduler 将状态转换为 `Paras`，返回每个用户的 `partition_s1`、`partition_s2`、早退阈值以及 Edge/Cloud 资源分配。运行节点不执行调度逻辑，只消费 decision JSON。

## 主要数据目录

```text
Data/
|-- Weights/                 # 模型权重
|-- ComputeProfiles/         # 各设备和后端的实测计算 profile
|-- OfflineTables/           # 早退率、准确率、层统计和难度标签
`-- Runtime/
    |-- SolutionCache/       # Scheduler 决策缓存
    `-- DeviceResults/       # Device 在线推理 CSV

Scripts/
`-- Results/                 # 离线算法和论文实验输出
```

## Phase 1：离线准备

常用命令：

```powershell
# 训练 backbone
python -m Src.Phase1_Offline.Training.train_model

# 微调早退头
python -m Src.Phase1_Offline.Training.finetune_exits

# 在实际运行设备上生成计算 profile
python -m Src.Phase1_Offline.Profiling.profile_device device-pytorch --backend pytorch --device cpu --threads 4

# 校验计算 profile
python -m Src.Phase1_Offline.Profiling.validate_compute_profile device-pytorch --backend pytorch --model-name Resnet50

# 生成早退 lookup table
python -m Src.Phase1_Offline.LookupTables.resnet50_thred_curve --overwrite

# 审计离线表
python -m Src.Phase1_Offline.LookupTables.audit_offline_tables
```

计算 profile 以 `Data/ComputeProfiles/<profile_id>/metadata.json` 和 `layers.csv` 保存。在线状态中使用的 profile ID 必须与实际设备、模型和推理后端匹配。

## 模型执行

PyTorch 运行路径中，三个节点加载同一份 `MultiEEResNet50` 权重，并通过 `forward_partial(x, start, end)` 执行指定阶段。

| Stage | 模型部分 |
| --- | --- |
| `0` | stem：`conv1 -> bn1 -> relu -> maxpool` |
| `1` | `layer1` |
| `2` | `layer2`，对应早退层 `57` |
| `3` | `layer3`，对应早退层 `103` |
| `4` | `layer4 -> avgpool -> flatten -> fc` |

MNN 入口保留在各运行节点目录中，但运行环境必须额外安装 MNN Python 包并准备对应模型文件。

## 验证

```powershell
python -m compileall Src Scripts Tests
python -m unittest discover -s Tests -v
python -m Src.Phase1_Offline.LookupTables.audit_offline_tables
python -m Src.Phase2_Scheduler.Service.api_server --help
python -m Src.Phase3_Runtime.Device.run_device --help
```

启动 Scheduler 后可检查：

```powershell
curl http://127.0.0.1:8000/api/v1/health
```
