# Data README

本目录保存复现实验和真机运行需要的数据、模型包、profile 与运行缓存。根目录 `README.md` 负责系统总览；这里专门说明 `Data/` 的有效结构、输入产物和复制规则。

## 目录契约

只有下面这些顶层目录属于当前代码的有效数据契约：

| 目录 | 用途 | 典型来源 |
| --- | --- | --- |
| `Datasets/` | 原始数据集与设备端测试包 | 手动准备、torchvision 下载、`create_test_package` 生成 |
| `Bundles/<bundle_id>/` | 每个模型包的权重、manifest、早退曲线、分析结果 | Phase 1 训练和离线分析生成 |
| `Profiles/<profile_id>/` | Device/Edge/Cloud 的 PyTorch/MNN 分段时延 profile | `profile_segments` 或 MNN profile 流程生成 |
| `Runtime/` | 当前运行缓存、调度解、设备输出、实验日志 | Phase 2/3 在线运行生成 |
| `Archive/` | 旧版或历史输入输出归档 | 仅人工参考，当前代码不应读取 |

历史顶层目录 `Weights`、`OfflineTables`、`PartitionManifests`、`SegmentProfiles`、`ComputeProfiles` 已不属于有效结构。新增模型文件应放入 `Bundles/<bundle_id>/`。

## 数据集

`Datasets/CIFAR10/` 用于 CIFAR-10，训练和查表命令可通过 `--download` 让 torchvision 下载。

`Datasets/ImageNet100/` 需要手动准备为：

```text
Data/Datasets/ImageNet100/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```

ImageNet100 加载器会检查类别目录数量是否为 100。

`Datasets/<dataset>/TestSets/` 保存设备端小规模测试包，用于 balanced、easy、hard 等样本评估。测试包通常由：

```powershell
python -m Src.Phase1_Offline.Datasets.create_test_package --bundle-id <bundle_id>
```

生成。

## 模型包

每个模型包独立放在：

```text
Data/Bundles/<bundle_id>/
```

常见文件如下：

| 文件或目录 | 说明 | 生成阶段 |
| --- | --- | --- |
| `weights.pth` | 训练好的 Multi-Exit 模型权重 | Phase 1 训练 |
| `manifest.json` | Segment、boundary、exit head 的切分描述 | Phase 1 manifest 生成 |
| `exit_curves.csv` | 阈值到精度、早退率的查找表 | Phase 1 LookupTables |
| `analysis/` | 训练日志、阈值曲线、分析图 | Phase 1 分析脚本 |
| `mnn_segments/` | MNN 后端使用的分段模型 | Phase 1 MNN 导出 |

常用 bundle ID 通过 `--bundle-id` 或环境变量 `DSCI_BUNDLE_ID` 指定，例如：

```powershell
$bundle = "resnet50-cifar10-ee-v1"
```

更多离线生成步骤见 [Phase1_README](../Src/Phase1_Offline/Phase1_README.md)。

## Segment Profile

真机运行时，Scheduler 需要知道 Device、Edge、Cloud 上每个 segment 的实测时延。Profile 位于：

```text
Data/Profiles/<profile_id>/
  metadata.json
  segments.csv
```

`profile_id` 应包含节点角色、机器名、后端、模型与数据集，避免覆盖已有 profile，例如：

```text
edge-kaijielaptop-pytorch-resnet50-cifar10
device-nano1-pytorch-resnet50-cifar10
cloud-v100-pytorch-resnet50-cifar10
```

新生成的 Device/Edge/Cloud profile 都要复制到 Scheduler 所在机器的 `Data/Profiles/` 下。否则 Scheduler 只能收到节点状态，无法用对应 profile 计算端到端时延。当前系统只支持 `fixed_worker_pool` segment profile；旧的逐层 Compute profile 和 `simulation_resource_mode` 不再属于有效数据契约。

## Runtime

`Runtime/` 是可丢弃的当前运行产物，通常不作为复现输入固定依赖：

| 目录 | 说明 |
| --- | --- |
| `Runtime/SolutionCache/` | DSCI/PPO 在线训练缓存和最近解 |
| `Runtime/ExperimentLogs/` | 真机或脚本运行日志 |
| `Runtime/DeviceResults/` | Device 端测试输出 |

复现实验时如果需要从干净状态开始，可以先备份或清理 `Runtime/` 中的临时缓存；模型包、数据集和 profile 不应放在 `Runtime/`。

## 相关文档

- [Src_README](../Src/Src_README.md)：系统代码复现入口。
- [Scripts_README](../Scripts/Scripts_README.md)：论文实验脚本和结果目录说明。
- [Phase1_README](../Src/Phase1_Offline/Phase1_README.md)：训练、manifest、exit curves 和 profile。
- [Phase2_README](../Src/Phase2_Scheduler/Phase2_README.md)：Scheduler、优化器和 API 服务。
- [Phase3_README](../Src/Phase3_Runtime/Phase3_README.md)：Device/Edge/Cloud 运行时。
