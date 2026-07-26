# Scripts 实验运行与结果说明

## 环境激活

- Linux
```bash
# conda
conda activate DSCI
# venv
source ./.venv/bin/activate
```
- Windows：
```powershell
# conda
conda activate DSCI
# venv
.\.venv\Scripts\Activate.ps1
```

---

## 当前实验目录与绘图入口

当前仓库保留 Exp0、Exp2、Exp3、Exp4 和 Exp5；Baseline 复现实验（Exp1）不在本仓库中。各实验的数据与图片均放在对应实验目录下，不再使用统一的 `Scripts\Results` 目录：

- `result_data`：原始实验记录、汇总数据，以及可由绘图脚本直接读取的最终统计数据。
- `result_figure`：最终生成的实验图片。
- `run`：需要额外运行实验或汇总脚本时使用；Exp2、Exp3 和 Exp5 直接使用设备实验数据，因此不设置 `run`。
- `plot`：一个实验包含多个独立绘图脚本时使用；只有一个绘图入口的实验将脚本直接放在实验目录下。

| 实验 | 运行/处理入口 | 绘图入口 | 绘图直接读取的数据 | 图片输出 |
| --- | --- | --- | --- | --- |
| Exp0 Motivation | `Exp0_Motivation\run` | `Exp0_Motivation\plot_all.py` | `Exp0_Motivation\result_data` 中的 CSV/JSON | `Exp0_Motivation\result_figure` |
| Exp2 Cross-Arch-Dataset | 无需额外运行脚本 | `Exp2_Cross-Arch-Dataset\plot_generalization.py` | `result_data\cross_arch_dataset_plot_data.csv` | `result_figure\cross_arch_dataset_accuracy.pdf`、`cross_arch_dataset_latency.pdf` |
| Exp3 Multi-Device | 无需额外运行脚本 | `Exp3_Multi-Device\plot_multi_device.py` | `result_data\multi_device_plot_data.csv` | `result_figure\multi_device.pdf` |
| Exp4 System-Overhead | `Exp4_System-Overhead\run` | `Exp4_System-Overhead\plot` | `result_data\summary\policy_update_summary.csv`、`convergence_summary.csv` | `Exp4_System-Overhead\result_figure` |
| Exp5 Ablation | 无需额外运行脚本 | `Exp5_Ablation\plot_ablation.py` | `result_data\ablation_plot_data.csv` | `result_figure\ablation.pdf` |

在仓库根目录下可分别执行：

```powershell
python Scripts\Exp0_Motivation\plot_all.py
python Scripts\Exp2_Cross-Arch-Dataset\plot_generalization.py
python Scripts\Exp3_Multi-Device\plot_multi_device.py
python Scripts\Exp4_System-Overhead\plot\plot_controlled_policy_update.py --experiment-dir Scripts\Exp4_System-Overhead\result_data
python Scripts\Exp5_Ablation\plot_ablation.py
```

Exp2、Exp3 和 Exp5 的绘图入口支持通过 `--data <CSV路径>` 指定其他绘图数据。也可以统一生成当前评估图片：

```powershell
python Scripts\EvaluationCommon\plot_evaluation_figures.py
```

论文中使用的最终统计结果及其数据来源见 [`实验结果汇总.md`](实验结果汇总.md)。以下原有内容保留为设备实验运行手册；最终绘图数据以各实验的 `result_data` 和汇总文档为准。

---

## 单Pi5（6组模型）
> [!IMPORTANT] 重复6个bundle 每个重复3次
> - `resnet50-cifar10`
> - `resnet50-imagenet100`
> - `resnet50-neucls64`
> - `vit-base-cifar10`
> - `vit-base-imagenet100`
> - `vit-base-neucls64`

Cloud 终端 1：
```bash
iperf3 -s -p 32264
```

Cloud 终端 2：
```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-cloud-5880"
python -m Src.Phase3_Runtime.Cloud.run_cloud \
  --bundle-id __BUNDLE_ID__ \
  --backend pytorch
```

Edge 终端1
```powershell
iperf3 -s -p 5001
```

Edge 终端2
```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-edge-kaijie-laptop"
```

```powershell
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id __BUNDLE_ID__ `
  --backend pytorch `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

Edge 终端3 Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --dynamic-bandwidth
```

### 冷启动
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
export ROUND_ID=cold-1dev-__BUNDLE_ID__-pi5-$(date +%Y%m%d-%H%M%S)
```

```bash
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

### 热启动 重复3次
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
export ROUND_ID=warm-1dev-__BUNDLE_ID__-pi5-$(date +%Y%m%d-%H%M%S)
```

```bash
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

## 多台递增（resnet50-cifar10）

|   N | 同时运行的端                               |
| --: | ------------------------------------ |
|   1 | Pi 5（已有）                             |
|   2 | Pi 5 + Pi 4-A                        |
|   3 | Pi 5 + Pi 4-A + Pi 4-B               |
|   4 | Pi 5 + Pi 4-A + Pi 4-B + Jetson Nano |
### N=2
停止旧 Scheduler，启动2用户训练策略：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 2 `
  --dynamic-bandwidth
```

#### 冷启动
生成 ROUND_ID：
```bash
export ROUND_ID=cold-2dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```

#### 热启动 重复3次
生成 ROUND_ID：
```bash
export ROUND_ID=warm-2dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```


### N=3
停止旧 Scheduler，启动3用户训练策略：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3 `
  --dynamic-bandwidth
```
#### 冷启动
生成 ROUND_ID：
```bash
export ROUND_ID=cold-3dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```

Pi 4-2，`user-id=2`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 2
```

#### 热启动 重复3次
生成 ROUND_ID：
```bash
export ROUND_ID=warm-3dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```

Pi 4-2，`user-id=2`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 2
```

### N=4
停止旧 Scheduler，启动3用户训练策略：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 4 `
  --dynamic-bandwidth
```
#### 冷启动
生成 ROUND_ID：
```bash
export ROUND_ID=cold-4dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```

Pi 4-2，`user-id=2`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 2
```

Nano，`user-id=3`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-nano"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 3
```

#### 热启动 重复3次
生成 ROUND_ID：
```bash
export ROUND_ID=warm-4dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

```bash
export ROUND_ID=
```

Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 0
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 1
```

Pi 4-2，`user-id=2`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 2
```

Nano，`user-id=3`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-nano"
python -m Src.Phase3_Runtime.Device.run_device \
  --dynamic-bandwidth --test-package-mode balanced --test-package-full \
  --bundle-id __BUNDLE_ID__ \
  --round-id "$ROUND_ID" \
  --test-samples 100 \
  --user-id 3
```


## 消融（resnet50-imagenet100）
| 配置            |
| ------------- |
| A. End only   |
| B. Edge only  |
| C. Cloud only |
| D. Split only |
| E. EE only    |
| F. ours（已有）   |
> [!IMPORTANT] 重复6个策略 每个重复3次
> - End only
> - Edge only
> - Cloud only
> - Split only
> - EE only
> - ours（已有）

### A. End only
> 含义：完整模型全部在 Pi 5 执行，不切分，关闭早退。

Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --fixed-split 19 19 `
  --fixed-threshold 1.0 `
  --no-auto-train `
  --dynamic-bandwidth
```

Pi 5
将 `REP=1` 分别修改为 `1、2、3`，运行三次。
```bash
export BUNDLE=resnet50-imagenet100
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"

for REP in 1 2 3; do
  export ROUND_ID="ablate-end-only-${BUNDLE}-pi5-r${REP}-$(date +%Y%m%d-%H%M%S)"

  python -m Src.Phase3_Runtime.Device.run_device \
    --dynamic-bandwidth --test-package-mode balanced --test-package-full \
    --bundle-id "$BUNDLE" \
    --round-id "$ROUND_ID" \
    --test-samples 100 \
    --user-id 0
done
```

### B. Edge only
> 含义：Pi 5 不执行模型，完整模型在 Edge 执行，不进入 Cloud，不早退。

Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --fixed-split 0 19 `
  --fixed-threshold 1.0 `
  --no-auto-train `
  --dynamic-bandwidth
```

Pi 5
```bash
export BUNDLE=resnet50-imagenet100
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"

for REP in 1 2 3; do
  export ROUND_ID="ablate-edge-only-${BUNDLE}-pi5-r${REP}-$(date +%Y%m%d-%H%M%S)"

  python -m Src.Phase3_Runtime.Device.run_device \
    --dynamic-bandwidth --test-package-mode balanced --test-package-full \
    --bundle-id "$BUNDLE" \
    --round-id "$ROUND_ID" \
    --test-samples 100 \
    --user-id 0
done
```

### C. Cloud only
> 含义：输入经过 Edge 转发到 Cloud，完整模型在 Cloud 执行，不早退。

Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --fixed-split 0 0 `
  --fixed-threshold 1.0 `
  --no-auto-train `
  --dynamic-bandwidth
```

Pi 5
```bash
export BUNDLE=resnet50-imagenet100
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"

for REP in 1 2 3; do
  export ROUND_ID="ablate-cloud-only-${BUNDLE}-pi5-r${REP}-$(date +%Y%m%d-%H%M%S)"

  python -m Src.Phase3_Runtime.Device.run_device \
    --dynamic-bandwidth --test-package-mode balanced --test-package-full \
    --bundle-id "$BUNDLE" \
    --round-id "$ROUND_ID" \
    --test-samples 100 \
    --user-id 0
done
```

### D. Split only
Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --ablation-mode split-only `
  --dynamic-bandwidth
```

Pi 5
```bash
export BUNDLE=resnet50-imagenet100
export MODE=split-only
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"

for REP in 1 2 3; do
  export ROUND_ID="ablate-${MODE}-${BUNDLE}-pi5-r${REP}-$(date +%Y%m%d-%H%M%S)"

  python -m Src.Phase3_Runtime.Device.run_device \
    --dynamic-bandwidth --test-package-mode balanced --test-package-full \
    --bundle-id "$BUNDLE" \
    --round-id "$ROUND_ID" \
    --test-samples 100 \
    --user-id 0
done
```

### E. EE only
Scheduler
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --ablation-mode ee-only `
  --dynamic-bandwidth
```

Pi 5
```bash
export BUNDLE=resnet50-imagenet100
export MODE=ee-only
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"

for REP in 1 2 3; do
  export ROUND_ID="ablate-${MODE}-${BUNDLE}-pi5-r${REP}-$(date +%Y%m%d-%H%M%S)"

  python -m Src.Phase3_Runtime.Device.run_device \
    --dynamic-bandwidth --test-package-mode balanced --test-package-full \
    --bundle-id "$BUNDLE" \
    --round-id "$ROUND_ID" \
    --test-samples 100 \
    --user-id 0
done
```
