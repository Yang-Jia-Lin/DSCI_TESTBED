# Src README

`Src/` 是系统代码的主入口，按三阶段组织：Phase 1 离线准备、Phase 2 在线调度、Phase 3 分布式运行时。根目录 `README.md` 保持系统总览不变；本文件面向复现，给出代码导航和快速跑通路径。

## 代码导航

| 目录 | 作用 | 详细文档 |
| --- | --- | --- |
| `Phase1_Offline/` | 训练模型、生成 manifest、生成早退曲线、测量 segment profile | [Phase1_README](Phase1_Offline/Phase1_README.md) |
| `Phase2_Scheduler/` | 将 Device/Edge/Cloud 状态转为切分点和早退阈值决策 | [Phase2_README](Phase2_Scheduler/Phase2_README.md) |
| `Phase3_Runtime/` | Cloud、Edge、Device 三端运行时和张量传输 | [Phase3_README](Phase3_Runtime/Phase3_README.md) |
| `Shared/` | 跨阶段共享的配置、模型、数据、manifest、profile 工具 | 本文件下方说明 |

已有更深层 README：

- [Phase1_Offline/Training/README_train.md](Phase1_Offline/Training/README_train.md)：Multi-Exit ResNet 训练、早退头微调和分析图生成。
- [Phase2_Scheduler/Service/README_alg_service.md](Phase2_Scheduler/Service/README_alg_service.md)：Scheduler 服务中的 DSCI/PPO 自适应缓存和 warm-start 逻辑。

## 复现顺序

1. 准备数据和模型包：检查 `Data/Datasets/`、`Data/Bundles/<bundle_id>/`，见 [DATA_README](../Data/DATA_README.md)。
2. 离线准备：生成或确认 `weights.pth`、`manifest.json`、`exit_curves.csv` 和各节点 profile，见 [Phase1_README](Phase1_Offline/Phase1_README.md)。
3. 启动运行时：按 Cloud、Edge、Scheduler、Device 顺序启动，见 [Phase3_README](Phase3_Runtime/Phase3_README.md)。
4. 查看调度逻辑：需要固定策略、禁用训练或理解 API 时，见 [Phase2_README](Phase2_Scheduler/Phase2_README.md)。
5. 复现实验图表：运行 `Scripts/` 下实验脚本，见 [Scripts_README](../Scripts/Scripts_README.md)。

## 共享模块

| 目录 | 说明 |
| --- | --- |
| `Shared/Config/` | 部署 IP、端口、bundle 配置、路径配置 |
| `Shared/Data/` | 数据集注册与加载 |
| `Shared/Models/` | Multi-Exit ResNet、DeiT 等模型定义 |
| `Shared/Partitioning/` | manifest 读取、boundary 校验、PyTorch segment executor |
| `Shared/Profiles/` | compute profile 和 segment profile 的读写 |
| `Shared/Utils/` | 训练日志、绘图、计时等通用工具 |

## 快速开始

# 离线准备
## 修改IP
`Src/Shared/Config/deploy_config.py`

## Profile
###### 新edge
新机器要起新名字防止覆盖，如 `edge-kaijielaptop-pytorch-resnet50-cifar10`）
```powershell
python -m Src.Phase1_Offline.Profiling.profile_segments edge-kaijielaptop-pytorch-resnet50-cifar10 --bundle-id resnet50-cifar10-ee-v1
```
###### 新device
新机器要起新名字防止覆盖，如 `device-nano1-pytorch-resnet50-cifar10`）
```bash
python -m Src.Phase1_Offline.Profiling.profile_segments device-nano1-pytorch-resnet50-cifar10 --bundle-id resnet50-cifar10-ee-v1
```
###### 复制 `Data\Profiles\Segments` 下新生成的 Profile 到Scheduler机器
> [!attention] 重要
> 所有新生成的 profile 目录要复制到 Scheduler 所在机器

---

# 运行
## Cloud
###### 终端1（iperf cloud）
```bash
iperf3 -s -p 32264
```
###### 终端2（run_cloud）
```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID=cloud-v100-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Cloud.run_cloud \
	--bundle-id resnet50-cifar10-ee-v1 \
	--backend pytorch
```

## Edge
###### 终端1（iperf edge）
```powershell
iperf3 -s -p 5001
```
###### 终端2（run_edge）
更换 `PROFILE_ID` 为上面的新名字
```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="edge-jialindesktop-pytorch-resnet50-cifar10"
python -m Src.Phase3_Runtime.Edge.run_edge `
	--bundle-id resnet50-cifar10-ee-v1 `
	--backend pytorch
```
###### 终端3（scheduler）
```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 1
```

## Device
更换 `PROFILE_ID` 为上面的新名字
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nx1-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Device.run_device \
	--bundle-id resnet50-cifar10-ee-v1 \
	--backend pytorch \
	--user-id 0 \
	--test-samples 1 \
	--round-id 2
```
