# Src README

`Src/` 是系统代码的主入口，按三阶段组织：Phase 1 离线准备、Phase 2 在线调度、Phase 3 分布式运行时。根目录 `README.md` 保持系统总览不变；本文件面向复现，给出代码导航和快速跑通路径。

#### 代码导航

| 目录 | 作用 | 详细文档 |
| --- | --- | --- |
| `Phase1_Offline/` | 训练模型、生成 manifest、生成早退曲线、测量 segment profile | [Phase1_README](Phase1_Offline/Phase1_README.md) |
| `Phase2_Scheduler/` | 将 Device/Edge/Cloud 状态转为切分点和早退阈值决策 | [Phase2_README](Phase2_Scheduler/Phase2_README.md) |
| `Phase3_Runtime/` | Cloud、Edge、Device 三端运行时和张量传输 | [Phase3_README](Phase3_Runtime/Phase3_README.md) |
| `Shared/` | 跨阶段共享的配置、模型、数据、manifest、profile 工具 | 本文件下方说明 |

论文实验新增的同步约束求解、共享资源模型、请求级并发屏障和 RequestTrace 见
[论文实验 Src 底层修改记录](PaperEvaluation_CORE_CHANGES_20260713.md)。

#### 复现顺序

1. 准备数据和模型包：检查 `Data/Datasets/`、`Data/Bundles/<bundle_id>/`，见 [DATA_README](../Data/DATA_README.md)。
2. 离线准备：生成或确认 `weights.pth`、`manifest.json`、`exit_curves.csv` 和各节点 profile，见 [Phase1_README](Phase1_Offline/Phase1_README.md)。
3. 启动运行时：按 Cloud、Edge、Scheduler、Device 顺序启动，见 [Phase3_README](Phase3_Runtime/Phase3_README.md)。
4. 查看调度逻辑：需要固定策略、禁用训练或理解 API 时，见 [Phase2_README](Phase2_Scheduler/Phase2_README.md)。
5. 复现实验图表：运行 `Scripts/` 下实验脚本，见 [Scripts_README](../Scripts/Scripts_README.md)。

#### 共享模块

| 目录 | 说明 |
| --- | --- |
| `Shared/Config/` | 部署 IP、端口、bundle 配置、路径配置 |
| `Shared/Data/` | 数据集注册与加载 |
| `Shared/Models/` | Multi-Exit ResNet、DeiT 等模型定义 |
| `Shared/Partitioning/` | manifest 读取、boundary 校验、PyTorch segment executor |
| `Shared/Profiles/` | Device/Edge/Cloud segment profile 的读写与校验 |
| `Shared/Utils/` | 训练日志、绘图、计时等通用工具 |

## 快速开始：两台 Device 联跑

本节按两台端设备写，复制命令时只需要保证：

- Scheduler 使用 `--expected-users 2`。
- 两台 Device 使用同一个 `--round-id`。下面的命令会用当前时间自动生成 `yyyymmdd-hhMM`，例如 `20260707-0948`。
- 两台 Device 使用不同的 `--user-id`，例如 `0` 和 `1`。
- 同一个 Scheduler 进程里，已经用过的 `round_id` 不能复用；重跑时重新执行命令会自动生成新的分钟级 `round_id`。
- 如果某个 `round_id + user_id` 已注册过，再用不同状态注册会返回 `409 Conflict`。

> 如果两台 Device 不是同一分钟启动，请先在第一台机器执行 `echo $ROUND_ID`，再把这个值复制到另一台机器执行 `export ROUND_ID=<同一个值>`。Scheduler 不需要因为换 `round_id` 重启；上一轮完成后，新 `round_id` 会自动开启下一轮。
> 不要写成 `ROUND_ID=$(date +%Y%m%d-%H%M) python ... --round-id "$ROUND_ID"`。shell 会先展开参数，此时 `"$ROUND_ID"` 可能还是空的，最终请求会变成 `/rounds//devices/register`。
> Edge 的 `iperf3 -s` 默认同一时间只能服务一个测试。两台 Device 同时启动时，后启动的一台可能看到 `server is busy running a test`，代码会自动重试；如果仍失败，可以错开启动，或设置 `export DSCI_IPERF_RETRY_SLEEP_S=12`。

### 离线准备

#### 修改 IP

检查：

```text
Src/Shared/Config/deploy_config.py
```

确认 `edge_host`、`cloud_host`、`algo_host` 指向当前 Edge、Cloud、Scheduler 机器。

#### Profile 新 Edge

新机器要起新名字防止覆盖，如 `edge-kaijielaptop-pytorch-resnet50-cifar10`。

```powershell
python -m Src.Phase1_Offline.Profiling.profile_segments edge-kaijielaptop-pytorch-resnet50-cifar10 --bundle-id resnet50-cifar10-ee-v1
```

#### Profile 新 Device

每台新 Device 都要有自己的 profile 名字，如 `device-nano1-pytorch-resnet50-cifar10`。

```bash
python -m Src.Phase1_Offline.Profiling.profile_segments device-nano1-pytorch-resnet50-cifar10 --bundle-id resnet50-cifar10-ee-v1
```

#### 复制 Profile 到 Scheduler 机器

> [!attention] 重要
> 所有新生成的 profile 目录都要复制到 Scheduler 所在机器的 `Data/Profiles/` 下。

### 运行顺序

#### 1. Cloud 终端 1：iperf cloud

```bash
iperf3 -s -p 32264
```

#### 2. Cloud 终端 2：run_cloud

```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID=cloud-v100-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Cloud.run_cloud \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch
```

#### 3. Edge 终端 1：iperf edge

```powershell
iperf3 -s -p 5001
```

#### 4. Edge 终端 2：run_edge

把 `DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID` 改成本机实际 profile。

```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="edge-jialindesktop-pytorch-resnet50-cifar10"
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id resnet50-cifar10-ee-v1 `
  --backend pytorch
```

#### 5. Scheduler 终端：api_server

两台 Device 联跑时必须是 `--expected-users 2`。

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2
```

#### 6. Device 0 终端

Jetson NX 示例。两台 Device 要在同一分钟内启动，或者先在两台机器上手动设置同一个 `ROUND_ID`。

```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nx1-pytorch-resnet50-cifar10
export ROUND_ID=$(date +%Y%m%d-%H%M)
echo $ROUND_ID # 两台 Device 都重新设置同一个新 round，复制
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch \
  --user-id 0 \
  --round-id "$ROUND_ID" \
  --test-samples 100
```

#### 7. Device 1 终端

第二台 Device 示例。把 profile 改成本机实际 profile；`--round-id` 必须和 Device 0 一样，`--user-id` 必须不同。

```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nano1-pytorch-resnet50-cifar10
export ROUND_ID= # 粘贴
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch \
  --user-id 1 \
  --round-id "$ROUND_ID" \
  --test-samples 100
```

### 409 Conflict 处理

出现 `409 Client Error: CONFLICT` 时，优先按下面顺序检查：

1. 是否两台 Device 都用了 `--user-id 0`。两台设备必须分别使用 `--user-id 0` 和 `--user-id 1`。
2. 是否 Scheduler 仍在等待两台 Device，但只启动了一台。两台联跑时 Scheduler 必须是 `--expected-users 2`。
3. 是否复用了已经注册或已经完成的 `round_id`。重跑请重新执行 `export ROUND_ID=$(date +%Y%m%d-%H%M)`，或手动设置新的值。
4. 是否上一轮还没结束。当前 Scheduler 一次只允许一个活跃 round；必要时重启 Scheduler 后使用新的 `round_id`。
