# DSCI Testbed

Device、Edge、Cloud 协同推理原型。当前主链路仅使用模型包、Partition
Manifest 和固定 Worker Pool；旧 5-stage Runtime 与旧模型接口已删除。

## Data 目录

活动代码只允许读取以下目录：

```text
Data/
|-- Datasets/                    # 共享原始数据集
|-- Bundles/<bundle_id>/         # 模型包资产
|-- Profiles/
|   |-- Compute/                 # 仿真算力 Profile
|   `-- Segments/                # 真机分段执行 Profile
|-- Runtime/                     # 可丢弃的当前运行产物
`-- Archive/                     # 只读历史记录，活动代码不得读取
```

模型包目录包含：

```text
Data/Bundles/<bundle_id>/
|-- weights.pth
|-- manifest.json
|-- exit_curves.csv
|-- layer_stats.csv              # 仅仿真模式需要
|-- analysis/                    # 与模型包相关的历史分析
`-- mnn_segments/
```

详细约束见 `Data/README.md`。

## 模型包

内置规格：

- `resnet18-cifar10-ee-v1`
- `resnet50-cifar10-ee-v1`
- `resnet18-imagenet100-ee-v1`
- `resnet50-imagenet100-ee-v1`

命令行 `--bundle-id` 优先于环境变量 `DSCI_BUNDLE_ID`。模型包仅在进程启动时
切换，不支持请求级热切换。

ImageNet100 使用本地 `ImageFolder`：

```text
Data/Datasets/ImageNet100/
|-- train/<100 class directories>/
`-- val/<100 class directories>/
```

项目不会下载 ImageNet100，类别数不是 100 时会拒绝运行。

## Phase 1

```powershell
$bundle = "resnet18-cifar10-ee-v1"

conda run -n DSCI python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle
conda run -n DSCI python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle
conda run -n DSCI python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite
conda run -n DSCI python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --overwrite
conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments device-pytorch --bundle-id $bundle
conda run -n DSCI python -m Src.Phase1_Offline.Profiling.export_mnn_segments --bundle-id $bundle
```

模型、出口或 Worker 配置变化后，必须重新生成对应 Manifest、出口曲线和
Segment Profile。

## 在线启动

三个节点必须选择相同 `bundle_id`，并配置与该模型包匹配的 Segment Profile：

```powershell
$env:DSCI_BUNDLE_ID = "resnet50-cifar10-ee-v1"
$env:DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID = "device-pytorch"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID = "edge-pytorch"
$env:DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID = "cloud-pytorch"

conda run -n DSCI python -m Src.Phase3_Runtime.Cloud.run_cloud --backend pytorch
conda run -n DSCI python -m Src.Phase3_Runtime.Edge.run_edge --backend pytorch
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server
conda run -n DSCI python -m Src.Phase3_Runtime.Device.run_device --backend pytorch
```

State JSON 必须包含 `bundle_id`；Decision 使用稳定出口 ID：

```json
{
  "bundle_id": "resnet50-cifar10-ee-v1",
  "exit_thresholds": {
    "after_layer2": 0.7,
    "after_layer3": 0.8
  }
}
```

旧 `model_name`、数字出口阈值、旧 Profile、旧离线表和旧缓存均不兼容。

## 验证

```powershell
conda run -n DSCI python -m compileall Src Scripts Tests -q
conda run -n DSCI python -m unittest discover -s Tests -v
```
