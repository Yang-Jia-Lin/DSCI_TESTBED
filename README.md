# DSCI Testbed

DSCI Testbed 是一个 Device、Edge、Cloud 三节点协同推理原型。三个节点都加载同一个 `MultiEEResNet50` 完整模型，并根据算法服务返回的 decision JSON 只执行自己负责的模型阶段。

## Windows 一键启动

每个节点目录下都有启动脚本。日志写入项目根目录的 `Logs/`。脚本启动后台进程后会打印 PID、日志路径，以及预期端口是否已经监听。如果进程立即退出，脚本会打印错误日志尾部。

```powershell
Src\Deploy\Cloud\start_cloud.ps1
Src\Deploy\Edge\start_edge.ps1
Src\Deploy\Device\start_device.ps1
```

启动顺序为 Cloud、Edge、Device。Edge 脚本默认会同时启动 Algorithm API。如果 Algorithm API 已经在别处运行，可以这样启动 Edge：

```powershell
Src\Deploy\Edge\start_edge.ps1 -NoAlgo
```

Algorithm API test overrides can be passed through the Edge launcher:

```powershell
Src\Deploy\Edge\start_edge.ps1 -FixedSplit 0,1
Src\Deploy\Edge\start_edge.ps1 -FixedSplit 0,1 -FixedThreshold 0.7
Src\Deploy\Edge\start_edge.ps1 -FixedThreshold 0.7
```

停止脚本：

```powershell
Src\Deploy\Cloud\stop_cloud.ps1
Src\Deploy\Edge\stop_edge.ps1
Src\Deploy\Device\stop_device.ps1
```

## 手动启动

如果需要使用原来的多终端启动方式，按下面命令启动：

```bash
iperf3 -s -p 32264                              # Cloud: Edge -> Cloud 带宽测量
python -m Src.Deploy.Cloud.run_cloud            # Cloud: 状态接口 + 特征接收服务

iperf3 -s -p 5001                               # Edge: Device -> Edge 带宽测量
python -m Src.Deploy.Edge.run_edge              # Edge: 状态接口 + 特征接收服务
python -m Src.Algorithm.Interface.api_server    # Algorithm API
# Optional test override:
# python -m Src.Algorithm.Interface.api_server --fixed-split 0 1 --fixed-threshold 0.7

python -m Src.Deploy.Device.run_device          # Device
```

## 运行目录

```text
DSCI_testbed/
|-- Data/
|   |-- CIFAR10/
|   |-- Weights/
|   |   |-- resnet50_cifar10_multi_ee.pth
|   |-- OfflineTables/
|       |-- resnet50_cifar10_rates.csv
|       |-- resnet50_cifar10_accs.csv
|       |-- resnet50_cifar10_layer_stats.csv
|       |-- resnet50_cifar10_difficulty_raw.csv
|       |-- resnet50_cifar10_difficulty_labeled.csv
|       `-- resnet50_cifar10_confidence_stats.json
|-- Scripts/
|   |-- Exp1_Offline/
|   |-- Exp0_Motivation/
|   |-- Exp1_Testbed/
|   |-- Exp2_Baseline/
|   |-- Exp3_Dynamic/
|   |-- Exp4_DSCI_Convergency/
|   |-- Exp5_Ablation/
|   |-- Exp6_EE_Model/
|   |-- Results/
|-- Src/
    |-- Algorithm/
    |   |-- Interface/
    |   |   |-- api_server.py
    |   |   |-- algo_service.py
    |   |   |-- decision_codec.py
    |   |   |-- state_adapter.py
    |   |   `-- SolutionCache/
    |   `-- Optimizer/
    |-- Deploy/
    |   |-- Cloud/
    |   |-- Device/
    |   |   `-- Results/
    |   |-- Edge/
    |   |-- Shared/
    |   `-- deploy_config.py
    `-- Models/
```

通用实验输出放在 `Scripts/Results/`。离线 lookup table 和 CIFAR-10 难度分层表放在 `Data/OfflineTables/`。Device 侧推理 CSV 输出放在 `Src/Deploy/Device/Results/`，文件名类似 `test_results_YYYYMMDDHHMMSS.csv`。

## 关键路径

| 用途 | 路径 |
| --- | --- |
| 完整模型权重 | `Data/Weights/resnet50_cifar10_multi_ee.pth` |
| 离线 CSV lookup table | `Data/OfflineTables/` |
| 离线实验脚本 | `Scripts/Exp1_Offline/` |
| 脚本实验输出 | `Scripts/Results/` |
| Algorithm API 最新 DSCI 决策缓存 | `Src/Algorithm/Interface/SolutionCache/latest_solution.npz` |
| Algorithm API 最新缓存元数据 | `Src/Algorithm/Interface/SolutionCache/latest_solution_meta.json` |
| Algorithm API 历史决策快照 | `Src/Algorithm/Interface/SolutionCache/solution_YYYYMMDDHHMMSSmmm.*` |
| Device 推理 CSV 输出 | `Src/Deploy/Device/Results/` |

`SolutionCache` 保留 `latest_solution.*` 供快速加载，同时只保留最近 3 个带时间戳的历史决策快照。

## Offline 模块

`Scripts/Exp1_Offline/` 存放重复 testbed 或仿真实验之前需要先运行的离线预处理步骤。可复用输出统一放在 `Data/OfflineTables/`。

检查 canonical 离线表是否都在 `Data/OfflineTables/` 且符合 `{model_slug}_{dataset_slug}_{artifact}` 命名：

```powershell
python Scripts\Exp1_Offline\audit_offline_tables.py
```

### 1. 生成早退 lookup table

只有在 CIFAR-10 测试集、模型权重或阈值网格变化时才需要重新运行：

```powershell
python Scripts\Exp1_Offline\resnet50_thred_curve.py `
  --model_path Data\Weights\resnet50_cifar10_multi_ee.pth `
  --data_root Data\CIFAR10 `
  --output_dir Data\OfflineTables `
  --overwrite
```

输出文件：

| 文件 | 作用 |
| --- | --- |
| `Data/OfflineTables/resnet50_cifar10_rates.csv` | 阈值到 exit1、exit2 早退率的 lookup table，供 `Paras()` 使用 |
| `Data/OfflineTables/resnet50_cifar10_accs.csv` | 阈值到 exit1、exit2、整体精度的 lookup table |

脚本默认不会覆盖已有 canonical 表。若要刷新 `resnet50_cifar10_rates.csv` 和 `resnet50_cifar10_accs.csv`，必须显式传入 `--overwrite`。如果还需要保留带时间戳的审计副本，可以额外加 `--timestamped_copy`。

### 2. CIFAR-10 样本难度 profiling

使用完整 ResNet50 对 CIFAR-10 test set 跑一次 final head，记录每个样本的置信度和熵：

```powershell
python Scripts\Exp1_Offline\profile_difficulty.py `
  --model_path Data\Weights\resnet50_cifar10_multi_ee.pth `
  --data_root Data\CIFAR10 `
  --output_dir Data\OfflineTables
```

输出文件：

| 文件 | 作用 |
| --- | --- |
| `Data/OfflineTables/resnet50_cifar10_difficulty_raw.csv` | 原始逐样本记录，包括 `image_id`、标签、预测、置信度和熵 |
| `Data/OfflineTables/resnet50_cifar10_confidence_stats.json` | 置信度分位数和建议阈值 |

### 3. 分配难度标签

根据 profiling 统计结果选择阈值，然后生成可复用的 labeled table：

```powershell
python Scripts\Exp1_Offline\assign_difficulty_labels.py `
  --input_path Data\OfflineTables\resnet50_cifar10_difficulty_raw.csv `
  --output_path Data\OfflineTables\resnet50_cifar10_difficulty_labeled.csv `
  --easy_min 0.90 `
  --hard_max 0.60
```

`Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv` 是后续难度感知实验的唯一数据源。不要在每次实验前重新 profiling，直接复用这个 labeled CSV。

### 4. 按难度测试早退表现

运行本地早退测试，并按 easy、medium、hard 汇总 accuracy、local exit rate 和 avg exit layer：

```powershell
python Scripts\Exp1_Offline\test_with_difficulty.py `
  --model_path Data\Weights\resnet50_cifar10_multi_ee.pth `
  --table_path Data\OfflineTables\resnet50_cifar10_difficulty_labeled.csv `
  --data_root Data\CIFAR10 `
  --partition_idx 3 `
  --output_dir Scripts\Results\Exp1_Offline `
  --exit_threshold_57 0.80 `
  --exit_threshold_103 0.80
```

输出文件：

| 文件 | 作用 |
| --- | --- |
| `Scripts/Results/Exp1_Offline/resnet50_cifar10_difficulty_results_YYYYMMDDHHMMSS.csv` | 逐样本预测、置信度、难度标签、退出层和是否传到 Cloud |

可以用 `--difficulty easy`、`--difficulty medium`、`--difficulty hard` 或
`--difficulty easy medium` 只测试部分难度。只有当 `Data/CIFAR10/` 下没有数据集文件时才使用
`--download`。

### 5. 在其他脚本中复用难度 dataloader

常规 PyTorch 实验脚本仍然可以使用原来的 `get_test_data_loaders()`。默认不传难度参数时，行为和原来一致，每个 batch 仍然是 `(images, labels)`：

```python
from Src.Algorithm.Utils.utils_function import get_test_data_loaders

test_loader = get_test_data_loaders(
    root="Data/CIFAR10",
    batch_size=128,
)

for images, labels in test_loader:
    ...
```

如果要测试简单样本，只在构造 dataloader 时增加难度表路径和 `difficulty`，循环代码仍然可以保持 `(images, labels)` 不变：

```python
test_loader = get_test_data_loaders(
    root="Data/CIFAR10",
    batch_size=128,
    difficulty_table_path="Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv",
    difficulty="easy",
)

for images, labels in test_loader:
    ...
```

困难样本只需要把 `difficulty` 改为 `"hard"`；多个难度可以写成 `["easy", "medium"]`；不筛选则不传 `difficulty_table_path` 和 `difficulty`。

如果新脚本需要读取难度标签、profiling 置信度或原始 `image_id`，显式打开 metadata：

```python
test_loader = get_test_data_loaders(
    root="Data/CIFAR10",
    batch_size=128,
    difficulty_table_path="Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv",
    difficulty="easy",
    include_difficulty_metadata=True,
    include_image_id=True,
)

for images, labels, difficulties, confidences, image_ids in test_loader:
    ...
```

`get_data_loaders()` 也支持同样逻辑，但只作用于它返回的 `test_loader`，训练集和验证集不变：

```python
from Src.Algorithm.Utils.utils_function import get_data_loaders

train_loader, valid_loader, test_loader = get_data_loaders(
    root="Data/CIFAR10",
    batch_size=128,
    valid_size=0.1,
    random_seed=42,
    test_difficulty_table_path="Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv",
    test_difficulty="hard",
)
```

难度筛选的原理是：`resnet50_cifar10_difficulty_labeled.csv` 保存每个 CIFAR-10 test 样本的 `image_id` 和 `difficulty`。dataloader 先按 `difficulty` 过滤 CSV 行，再用 `image_id` 去索引原始 CIFAR-10 test set。它不会复制数据集，只是改变 test set 的采样索引。

### 6. 部署侧共享 dataloader 和 ONNX/MNN 说明

`Src/Deploy/Shared/dataloader.py` 是 CIFAR-10 test 和难度筛选的唯一实现。Deploy 侧的 `run_device.py`、算法侧的 `get_test_data_loaders()` / `get_data_loaders()`、以及 `Scripts/Exp1_Offline/` 的离线脚本都复用同一条路径。

该 loader 支持两种数据根目录：

```text
Data/CIFAR10
Data/CIFAR10/cifar-10-batches-py
```

默认样本格式保持部署兼容：`cifar10_test_loader()` 逐张返回 `(tensor, label)`，其中 tensor 为 `(1, 3, 227, 227)`，已按 CIFAR-10 mean/std 归一化。需要难度筛选时：

```powershell
python -m Src.Deploy.Device.run_device --difficulty hard --test-samples 100
```

不显式传 `--difficulty-table` 时，`--difficulty` 会默认读取 `Data/OfflineTables/resnet50_cifar10_difficulty_labeled.csv`。如果只想记录难度 metadata 但不过滤样本，可以传：

```powershell
python -m Src.Deploy.Device.run_device `
  --difficulty-table Data\OfflineTables\resnet50_cifar10_difficulty_labeled.csv
```

更换推理后端为 ONNX 或 MNN 后，难度表仍然可用，因为它只依赖 `image_id`、`difficulty` 和 CIFAR-10 test 样本顺序。需要单独适配的是最后一步 `tensor -> runtime input` 转换。

## 为什么必须启动 iperf3

iperf3 用于实时测量节点间带宽。测得的 `BW_d2e` 和 `BW_e2c` 会直接传给 Algorithm API，用于计算最优切分点。如果 iperf3 没有启动，`measure_bandwidth_iperf` 会返回 `0.0`，算法会基于错误带宽做决策，导致不合理的推理切分。

开始任何推理之前，必须先启动对应的 iperf3 server。

## 部署拓扑

| 角色 | 典型设备 | 说明 |
| --- | --- | --- |
| Device | Raspberry Pi | 发起推理，测量带宽，请求分区决策 |
| Edge | Linux 服务器 | 接收 Device 特征，执行中间阶段推理，并转发到 Cloud |
| Cloud | PC 或服务器 | 执行最后阶段推理并返回分类结果 |

## 各节点启动方式

### 1. Cloud 节点

启动两个进程，同一节点内的顺序不敏感：

```bash
# iperf3 server，供 Edge 测量 Edge -> Cloud 带宽
iperf3 -s -p 32264

# Cloud 推理服务
python -m Src.Deploy.Cloud.run_cloud
```

Cloud 使用的端口：

| 端口 | 用途 |
| --- | --- |
| `32264` | iperf3 server，Edge 连接它测量 `BW_e2c` |
| `32265` | Cloud 状态 HTTP API，路径为 `/status` |
| `32266` | 接收 Edge 发送的特征张量 |

### 2. Edge 节点

启动两个进程：

```bash
# iperf3 server，供 Device 测量 Device -> Edge 带宽
iperf3 -s -p 5001

# Edge 推理服务
python -m Src.Deploy.Edge.run_edge
```

Edge 使用的端口：

| 端口 | 用途 |
| --- | --- |
| `5001` | iperf3 server，Device 连接它测量 `BW_d2e` |
| `9001` | 接收 Device 发送的特征张量 |
| `9002` | Edge 状态 HTTP API，返回 `f_e_max` 和 `BW_e2c` |

如果 Cloud 地址或端口发生变化，需要修改 `Src/Deploy/deploy_config.py`。Edge 会从这个共享配置读取 `cloud_host`、`cloud_iperf_port` 和 `cloud_feature_port`。

### 3. Algorithm API

Algorithm API 可以运行在 Edge、Cloud 或任意 Device 可访问的机器上：

```bash
python -m Src.Algorithm.Interface.api_server
```

Test override options:

```bash
# Force every decision to use partition_s1=0 and partition_s2=1.
python -m Src.Algorithm.Interface.api_server --fixed-split 0 1

# Force every early-exit threshold in Y, for example layers 57 and 103.
python -m Src.Algorithm.Interface.api_server --fixed-threshold 0.7

# Combine both overrides.
python -m Src.Algorithm.Interface.api_server --fixed-split 0 1 --fixed-threshold 0.7
```

The same overrides can be supplied per request body without restarting the API:

```json
{
  "fixed_split": [0, 1],
  "fixed_threshold": 0.7
}
```

When unset, the API keeps the old behavior: cached/default DSCI decisions and the
existing `decision_mode` presets are used normally.

使用端口：

| 端口 | 用途 |
| --- | --- |
| `8000` | 决策和健康检查 HTTP API |

### 4. Device 节点

启动前，先在 `Src/Deploy/deploy_config.py` 中设置真实 IP：

```python
edge_host = "<Edge real IP>"
cloud_host = "<Cloud real IP>"
algo_host = "<Algorithm API real IP>"
```

如果部署到 Linux 或 Raspberry Pi，还需要在 `Src/Deploy/Shared/bandwidth_iperf.py` 中把 `IPERF_EXE` 改成系统里的 `iperf3`：

```python
# 将 Windows 路径：
# IPERF_EXE = "S:\\Tools\\Iperf\\iperf3.exe"
# 改成：
IPERF_EXE = "iperf3"
```

然后启动 Device：

```bash
python -m Src.Deploy.Device.run_device
```

## 推荐启动顺序

```text
1.  Cloud:   iperf3 -s -p 32264
2.  Cloud:   python -m Src.Deploy.Cloud.run_cloud
3.  Edge:    iperf3 -s -p 5001
4.  Edge:    python -m Src.Deploy.Edge.run_edge
5.  Any:     python -m Src.Algorithm.Interface.api_server
6.  Device:  python -m Src.Deploy.Device.run_device
```

Cloud 和 Edge 服务必须在 Device 启动前完全就绪，这样第一次推理请求时带宽测量和状态查询才能成功。

Windows 一键脚本对应的顺序是：

```text
1.  Cloud:   Src\Deploy\Cloud\start_cloud.ps1
2.  Edge:    Src\Deploy\Edge\start_edge.ps1
3.  Device:  Src\Deploy\Device\start_device.ps1
```

## 端口表

| 端口 | 所属节点 | 用途 |
| --- | --- | --- |
| `5001` | Edge | iperf3 server，Device -> Edge 带宽测量 |
| `32264` | Cloud | iperf3 server，Edge -> Cloud 带宽测量 |
| `8000` | Algorithm API | 决策和健康检查 HTTP API |
| `9001` | Edge | 接收 Device 发送的特征张量 |
| `9002` | Edge | Edge 状态 HTTP API，返回 `f_e_max` 和 `BW_e2c` |
| `32265` | Cloud | Cloud 状态 HTTP API |
| `32266` | Cloud | 接收 Edge 发送的特征张量 |

## 数据流

```text
Device                         Edge                          Cloud
  |                              |                              |
  |-- iperf3 ------------------->| :5001                        |
  |   测量 BW_d2e                |                              |
  |                              |-- iperf3 ------------------->| :32264
  |                              |   测量 BW_e2c                |
  |-- HTTP GET ----------------->| :9002                        |
  |   获取 Edge 状态             |                              |
  |-- HTTP GET ------------------------------------------------>| :32265
  |   获取 Cloud 状态            |                              |
  |-- POST (:8000) ------------->| Algorithm API                |
  |   获取分区决策               |                              |
  |-- 本地推理 stage 0..s1       |                              |
  |-- TCP ---------------------->| :9001                        |
  |   发送特征张量               |-- 推理 stage s1+1..s2        |
  |                              |-- TCP ---------------------->| :32266
  |                              |   转发特征张量               |-- 推理 stage s2+1..end
  |<-----------------------------|<-----------------------------|
  |   接收最终结果               |                              |
```

## 模型执行

每个节点都加载：

```text
MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True)
```

权重从 `Data/Weights/resnet50_cifar10_multi_ee.pth` 以 `state_dict` 形式加载。运行时使用 `forward_partial(x, start, end)` 执行指定阶段。

| Stage | 含义 |
| --- | --- |
| `0` | stem: `conv1 -> bn1 -> relu -> maxpool` |
| `1` | `layer1` |
| `2` | `layer2`，早退层为 `57` |
| `3` | `layer3`，早退层为 `103` |
| `4` | `layer4 -> avgpool -> flatten -> fc` |

## 部署注意事项

- Device 节点只需要模型推理、Algorithm API URL、Edge socket 连通性，以及 iperf/status 测量能力。
- Algorithm API 可以运行在 PC、Edge 节点、Cloud 节点，或任何 Device 能访问的机器上。
- Raspberry Pi 部署时，需要保持 `python -m ...` 模块命令一致，并检查以下内容：
  - `Src/Deploy/deploy_config.py` 中的 IP 和端口。
  - `Src/Deploy/Shared/bandwidth_iperf.py` 中的 `IPERF_EXE`。
  - PyTorch、ONNX 或 MNN 运行时是否可用。
  - `Data/Weights/resnet50_cifar10_multi_ee.pth` 是否存在。
  - `Data/OfflineTables/` 下 CSV 文件是否齐全。

## 验证

```bash
python -m compileall Src Scripts
curl http://127.0.0.1:8000/api/v1/health
```
