# 实验 1：vit-base-cifar10 真实网络场景运行（baseline对比结果）

## 一、模型、数据集、设备、指标、参数准备

| 架构       | 数据集      | Bundle ID          |
| -------- | -------- | ------------------ |
| ViT-Base | CIFAR-10 | `vit-base-cifar10` |

| user-id | 角色               | 机器             | Profile 后缀           |
| ------: | ---------------- | -------------- | -------------------- |
|         | Cloud            | v100           | `cloud-v100`         |
|         | Edge + Scheduler | kaijie-laptop  | `edge-kaijie-laptop` |
|       0 | Device           | pi4-1          | `device-pi4-1`       |
|       1 | Device           | pi4-2          | `device-pi4-2`       |
|       2 | Device           | nano           | `device-nano`        |
|       3 | Device           | pi5            | `device-pi5`         |

- **指标**：四台 Device 全部请求的 Accuracy、端到端 `T_total` 和 observed Utility。
- **参数**：
	- $\alpha$ = baseline最优
	- $\beta$ = 自动搜索
- **其他**
	- tensor transport dtype = float32
	- expected users = 4
	- test package = balanced
	- formal repeats = 3
- **激活环境**
- pi4-1、pi4-2、pi5、v100：
```bash
conda activate DSCI
```
- nano：
```bash
source .venv/bin/activate
```
- kaijie-laptop：
```powershell
conda activate DSCI
```

---

## 二、第一轮运行：冷启动生成策略

#### 1. 云

###### 终端 1：Cloud Iperf

```bash
iperf3 -s -p 32264
```

###### 终端 2：Cloud Runtime

```bash
export BUNDLE=vit-base-cifar10
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-cloud-v100"
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
python -m Src.Phase3_Runtime.Cloud.run_cloud --bundle-id "$BUNDLE" --backend pytorch
```

###### 检查

```bash
curl http://127.0.0.1:32265/status
```

#### 2. 边 Kaijie-laptop

###### 终端 1：Edge Iperf

```powershell
iperf3 -s -p 5001
```

###### 终端 2：Edge Runtime

```powershell
$bundle = "vit-base-cifar10"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID = "$bundle-edge-kaijie-laptop"
$env:DSCI_TENSOR_TRANSPORT_DTYPE = "float32"
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id $bundle `
  --backend pytorch
```

###### 终端 3：Scheduler（使用参数搜索）

```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 4 `
  --target-accuracy 0.90 `
  --force-retrain
```

#### 3. 依次运行 pi4-1、pi4-2、nano、pi5 推理测试，使调度器开始冷启动训练

在一台机器上只生成一次共同 ROUND_ID：

```bash
export ROUND_ID=cold-4dev-vit-base-cifar10-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

输出 echo 原样粘贴。

###### pi4-1

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### pi4-2

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 1 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### nano

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-nano"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 2 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### pi5

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 3 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

---

## 三、调度器训练结束后直接重新测试

在一台机器上只生成一次共同 ROUND_ID：

```bash
export ROUND_ID=hot-4dev-vit-base-cifar10-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

输出 echo 原样粘贴。

###### pi4-1

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### pi4-2

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 1 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### nano

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-nano"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 2 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### pi5

```bash
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="vit-base-cifar10-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id vit-base-cifar10 --backend pytorch \
  --user-id 3 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

> 上面的步骤重复 3 次，取均值和方差。

填表：![100](Others/Assets/01-Research/实验%20Bseline-SEAS%20手册/Asset-202607181322.png)

---

如果能更换网络重新测试，则再重复两次步骤二和步骤三，测试其他速度。

---

---

# 实验 2：六模型真实网络场景运行

## 一、模型、数据集、设备、指标、参数准备

目的：证明 SEAM 在不同模型与任务上均具有一致有效性，而非针对单一场景调参。

| 架构        | 数据集          | Bundle ID              |
| --------- | ------------ | ---------------------- |
| ResNet-50 | CIFAR-10     | `resnet50-cifar10`     |
| ResNet-50 | NEU-CLS-64   | `resnet50-neucls64`    |
| ResNet-50 | ImageNet-100 | `resnet50-imagenet100` |
| ViT-Base  | CIFAR-10     | `vit-base-cifar10`     |
| ViT-Base  | NEU-CLS-64   | `vit-base-neucls64`    |
| ViT-Base  | ImageNet-100 | `vit-base-imagenet100` |

| user-id | 角色               | 机器             | Profile 后缀           |
| ------: | ---------------- | -------------- | -------------------- |
|         | Cloud            | v100           | `cloud-v100`         |
|         | Edge + Scheduler | kaijie-laptop  | `edge-kaijie-laptop` |
|       0 | Device           | pi4-1          | `device-pi4-1`       |
|       1 | Device           | pi4-2          | `device-pi4-2`       |
|       2 | Device           | nano           | `device-nano`        |
|       3 | Device           | pi5            | `device-pi5`         |

| 架构 | Static fixed split | Static threshold |
| --- | --- | ---: |
| ResNet-50 | `(4, 8)` | `0.7` |
| ViT-Base | `(4, 7)` | `0.7` |

- 对比策略 `Static`：固定切分点和固定早退阈值，作为简单下界。
- **指标**：四台 Device 全部请求的 Accuracy、端到端 `T_total` 和 observed Utility。
- **参数**：
	- $\alpha$ = baseline最优
	- $\beta$ = 自动搜索
- **其他**
	- tensor transport dtype = float32
	- expected users = 4
	- test package = balanced
	- formal repeats = 3
- **激活环境**
- pi4-1、pi4-2、pi5、v100：
```bash
conda activate DSCI
```
- nano：
```bash
source .venv/bin/activate
```
- kaijie-laptop：
```powershell
conda activate DSCI
```

---

## 二、每个模型第一轮运行：冷启动生成策略

每次从下面六个 Bundle ID 中选择一个，六个模型依次完整执行步骤二和步骤三：

```text
resnet50-cifar10
resnet50-neucls64
resnet50-imagenet100
vit-base-cifar10
vit-base-neucls64
vit-base-imagenet100
```

#### 1. 云

###### 终端 1：Cloud Iperf

```bash
iperf3 -s -p 32264
```

###### 终端 2：Cloud Runtime

```bash
export BUNDLE=resnet50-cifar10
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-cloud-v100"
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
python -m Src.Phase3_Runtime.Cloud.run_cloud --bundle-id "$BUNDLE" --backend pytorch
```

###### 检查

```bash
curl http://127.0.0.1:32265/status
```

#### 2. 边 Kaijie-laptop

###### 终端 1：Edge Iperf

```powershell
iperf3 -s -p 5001
```

###### 终端 2：Edge Runtime

```powershell
$bundle = "resnet50-cifar10"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID = "$bundle-edge-kaijie-laptop"
$env:DSCI_TENSOR_TRANSPORT_DTYPE = "float32"
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id $bundle `
  --backend pytorch
```

###### 终端 3：Scheduler（使用参数搜索）

```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 4 `
  --target-accuracy 0.90 `
  --force-retrain
```

#### 3. 依次运行 pi4-1、pi4-2、nano、pi5 推理测试，使调度器开始冷启动训练

在一台机器上只生成一次共同 ROUND_ID：

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID="cold-4dev-${BUNDLE}-$(date +%Y%m%d-%H%M%S)"
echo "$ROUND_ID"
```

输出 echo 原样粘贴。

###### pi4-1

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### pi4-2

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 1 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### nano

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-nano"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 2 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

###### pi5

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 3 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 --no-request-barrier
```

---

## 三、每个模型调度器训练结束后直接重新测试

在一台机器上只生成一次共同 ROUND_ID：

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID="hot-4dev-${BUNDLE}-$(date +%Y%m%d-%H%M%S)"
echo "$ROUND_ID"
```

输出 echo 原样粘贴。

###### pi4-1

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### pi4-2

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 1 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### nano

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-nano"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 2 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

###### pi5

```bash
export BUNDLE=resnet50-cifar10
export ROUND_ID=
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id "$BUNDLE" --backend pytorch \
  --user-id 3 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 100 --decision-timeout 900 --no-request-barrier
```

> 每个模型的步骤三重复 3 次，取均值和方差。完成后将所有终端中的 `BUNDLE` 改为下一个 Bundle ID，再重复步骤二和步骤三。
