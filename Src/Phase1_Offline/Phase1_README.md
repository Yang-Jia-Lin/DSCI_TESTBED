# Phase 1 README

Phase 1 负责所有在线运行前的离线准备：训练或准备 Multi-Exit 模型，生成 partition manifest，生成早退查找表，测量各节点 segment profile，并在需要时导出 MNN segment。

## 输入与输出

| 步骤 | 输入 | 输出 |
| --- | --- | --- |
| 模型训练 | `Data/Datasets/<dataset>/` | `Data/Bundles/<bundle_id>/weights.pth` |
| 早退头微调 | `weights.pth`、训练/验证集 | `weights.pth`、`analysis/finetune_exits_log.csv` |
| Manifest 生成 | 模型结构、权重、bundle 配置 | `Data/Bundles/<bundle_id>/manifest.json` |
| Exit curves | 验证集、模型权重、manifest | `Data/Bundles/<bundle_id>/exit_curves.csv` |
| Segment profile | 目标硬件、模型权重、manifest | `Data/Profiles/Segments/<profile_id>/` |
| MNN 导出 | 模型权重、manifest | `Data/Bundles/<bundle_id>/mnn_segments/` |

数据目录说明见 [DATA_README](../../Data/DATA_README.md)。

## 常用命令

先选择模型包：

```powershell
$bundle = "resnet50-cifar10-ee-v1"
```

训练主模型：

```powershell
python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle
```

微调早退头：

```powershell
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle
```

生成 partition manifest：

```powershell
python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite
```

生成早退曲线：

```powershell
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --overwrite
```

测量 PyTorch segment profile：

```powershell
python -m Src.Phase1_Offline.Profiling.profile_segments device-nano1-pytorch-resnet50-cifar10 --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments edge-kaijielaptop-pytorch-resnet50-cifar10 --bundle-id $bundle
python -m Src.Phase1_Offline.Profiling.profile_segments cloud-v100-pytorch-resnet50-cifar10 --bundle-id $bundle
```

新机器的 `profile_id` 要起新名字，避免覆盖已有 profile。生成后把 `Data/Profiles/Segments/<profile_id>/` 复制到 Scheduler 所在机器。

## 子目录

| 目录 | 说明 |
| --- | --- |
| `Training/` | 训练主模型、微调早退头、绘制训练和早退分析图 |
| `LookupTables/` | 生成阈值到精度、早退率的查找表 |
| `Profiling/` | 生成 manifest、测量 segment profile、导出 MNN segments |
| `Datasets/` | 生成设备端小规模测试包 |

训练细节见 [Training/README_train.md](Training/README_train.md)。

## 什么时候需要重跑

| 变化 | 需要重跑 |
| --- | --- |
| 数据集或模型结构改变 | 训练、早退头微调、manifest、exit curves、profile |
| 权重改变 | manifest、exit curves、profile |
| 早退出口或切分边界改变 | manifest、exit curves、profile |
| Device/Edge/Cloud 硬件改变 | 对应节点的 segment profile |
| worker 数、线程数或后端改变 | 对应节点的 segment profile |

## 下游文档

- [Phase2_README](../Phase2_Scheduler/Phase2_README.md)：使用离线产物进行调度决策。
- [Phase3_README](../Phase3_Runtime/Phase3_README.md)：加载 profile 和模型包启动在线运行时。
