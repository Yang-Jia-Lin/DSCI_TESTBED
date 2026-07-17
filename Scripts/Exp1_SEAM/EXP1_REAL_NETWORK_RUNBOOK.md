# 实验 1：六模型真实网络场景运行指南

## 1. 实验定义

目的：证明 SEAM 在不同模型与任务上均具有一致有效性，而非针对单一场景调参。

| 架构 | 数据集 | Bundle ID |
| --- | --- | --- |
| ResNet-50 | CIFAR-10 | `resnet50-cifar10` |
| ResNet-50 | NEU-CLS-64 | `resnet50-neucls64` |
| ResNet-50 | ImageNet-100 | `resnet50-imagenet100` |
| ViT-Base | CIFAR-10 | `vit-base-cifar10` |
| ViT-Base | NEU-CLS-64 | `vit-base-neucls64` |
| ViT-Base | ImageNet-100 | `vit-base-imagenet100` |

对比方法：

1. `Static`：固定切分点和固定早退阈值，作为简单下界；
2. `SEAM`：Scheduler 的 DSCI 联合决策。

指标：四台 Device 全部请求的 Accuracy、端到端 `T_total` 和 observed Utility。

本指南只运行一个真实网络场景：不使用 `tc`、不设置带宽档位、不把带宽作为实验自变量。每个 bundle 开始前用 iperf3 测量真实吞吐中位数，并把这个实测快照固定用于该 bundle 的 Static 与 SEAM。实际张量仍通过未限速的真实网络传输。

> [!IMPORTANT]
> 这里的 `--override-bw-*` 不是伪造或限制网络，而是把刚实测的中位数固定为 Scheduler 状态输入，防止重复 iperf 的微小波动破坏 SEAM 的 exact cache。

## 2. 固定拓扑

| user-id | 角色 | 机器 | Profile 后缀 |
| ---: | --- | --- | --- |
| 0 | Device | Jetson NX | `device-nx` |
| 1 | Device | Jetson Nano | `device-nano` |
| 2 | Device | jialin-desktop | `device-jialin-desktop` |
| 3 | Device | jialin-laptop | `device-jialin-laptop` |
| - | Edge + Scheduler | kaijie-laptop | `edge-kaijie-laptop` |
| - | Cloud | V100 | `cloud-v100` |

固定启动顺序：

```text
Cloud iperf -> Cloud runtime -> Edge iperf -> 测量网络快照
-> Edge runtime -> Scheduler -> 四台 Device
```

下文假设仓库路径为：

```text
Linux:   ~/Desktop/DSCI_SEAS
Windows: $HOME\Desktop\DSCI_SEAS
```

## 3. 固定参数与实验数量

```text
alpha = 1
beta = 20
tensor transport dtype = float32
expected users = 4
request barrier = enabled
test package = balanced / test / 10 samples per class / seed 42
formal repeats = 3
```

| 架构 | Static fixed split | Static threshold |
| --- | --- | ---: |
| ResNet-50 | `(4, 8)` | `0.7` |
| ViT-Base | `(4, 7)` | `0.7` |

```text
6 bundles × 2 methods × 3 repeats = 36 个正式 round
SEAM 另有 6 个 rep0 预热 round，不计入结果
```

Utility 固定定义：

```text
observed_utility = 1 × is_correct - 20 × T_total_seconds
```

## 4. 实验前一次性检查

### 4.1 六台机器确认同一代码版本

Linux：

```bash
cd ~/Desktop/DSCI_SEAS
git rev-parse HEAD
git status --short
```

Windows：

```powershell
Set-Location "$HOME\Desktop\DSCI_SEAS"
git rev-parse HEAD
git status --short
```

六台机器 commit 必须一致，实验期间不要 `git pull`。

### 4.2 检查部署 IP

所有机器的 `Src/Shared/Config/deploy_config.py` 必须使用同一真实路径：

```text
edge_host  = kaijie-laptop 的实验网络 IP
algo_host  = kaijie-laptop 的实验网络 IP
cloud_host = V100 IP
```

不要混用热点、局域网和 Tailscale IP。

### 4.3 检查 bundle 和 profile

所有机器检查 bundle：

```bash
python -c "from Src.Shared.Config.model_config import get_bundle; ids=['resnet50-cifar10','resnet50-neucls64','resnet50-imagenet100','vit-base-cifar10','vit-base-neucls64','vit-base-imagenet100']; [print(get_bundle(x).bundle_id) for x in ids]"
```

NX 检查自己的 6 份 profile：

```bash
for b in resnet50-cifar10 resnet50-neucls64 resnet50-imagenet100 vit-base-cifar10 vit-base-neucls64 vit-base-imagenet100; do
  test -f "Data/Profiles/${b}-device-nx/metadata.json" || echo "MISSING ${b}-device-nx"
  test -f "Data/Profiles/${b}-device-nx/segments.csv" || echo "MISSING ${b}-device-nx/segments.csv"
done
```

Nano 将 `device-nx` 改成 `device-nano`。Windows Device 用下列命令，并按机器修改后缀：

```powershell
$role = "device-jialin-desktop" # 另一台改为 device-jialin-laptop
$bundles = @("resnet50-cifar10","resnet50-neucls64","resnet50-imagenet100","vit-base-cifar10","vit-base-neucls64","vit-base-imagenet100")
$bundles | ForEach-Object {
  $p = "Data\Profiles\$_-$role"
  if (-not (Test-Path "$p\metadata.json") -or -not (Test-Path "$p\segments.csv")) {
    Write-Host "MISSING $p" -ForegroundColor Red
  }
}
```

kaijie-laptop 的 Scheduler 必须拥有当前 bundle 的 4 个 Device profile、Edge profile 和 Cloud profile：

```powershell
$bundle = "resnet50-cifar10"
$profiles = @("$bundle-device-nx","$bundle-device-nano","$bundle-device-jialin-desktop","$bundle-device-jialin-laptop","$bundle-edge-kaijie-laptop","$bundle-cloud-v100")
$profiles | ForEach-Object { if (-not (Test-Path "Data\Profiles\$_\metadata.json")) { Write-Host "MISSING $_" -ForegroundColor Red } }
```

任意 `MISSING` 都必须先处理，不能用其他机器 profile 代替。

### 4.4 检查端口

```text
Edge:  TCP 5001 iperf, 8000 Scheduler, 9001 tensor, 9002 status
Cloud: TCP 32264 iperf, 32265 status, 32266 tensor
```

所有机器确认：

```bash
iperf3 --version
```

## 5. 每个 bundle 的 Cloud、测速与 Edge

下面以 `resnet50-cifar10` 为例。切换 bundle 时，只改 bundle，不改机器后缀。

### 5.1 V100 终端 1：Cloud iperf

```bash
cd ~/Desktop/DSCI_SEAS
conda activate DSCI
iperf3 -s -p 32264
```

### 5.2 V100 终端 2：Cloud runtime

```bash
cd ~/Desktop/DSCI_SEAS
conda activate DSCI
export BUNDLE=resnet50-cifar10
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="${BUNDLE}-cloud-v100"
export DSCI_PYTORCH_DEVICE=cuda:0
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
python -m Src.Phase3_Runtime.Cloud.run_cloud --bundle-id "$BUNDLE" --backend pytorch
```

检查：

```bash
curl http://127.0.0.1:32265/status
```

### 5.3 kaijie-laptop 终端 1：Edge iperf

```powershell
Set-Location "$HOME\Desktop\DSCI_SEAS"
conda activate DSCI
iperf3 -s -p 5001
```

### 5.4 测量本 bundle 的真实网络快照

不使用 `tc`。每条链路连续测 3 次，取最后一行 sender 的 `Mbits/sec` 中位数。

kaijie-laptop 测 Edge -> Cloud：

```powershell
$cloudHost = "<CLOUD_HOST>"
1..3 | ForEach-Object { iperf3 -c $cloudHost -p 32264 -t 10 }
$bwE2C = <三次结果的中位数_Mbps>
```

四台 Device 必须依次测试，避免 `server is busy`。

NX / Nano 分别执行并填写各自结果：

```bash
EDGE_HOST=<EDGE_HOST>
for i in 1 2 3; do iperf3 -c "$EDGE_HOST" -p 5001 -t 10; done
export BW_D2E=<本机三次结果的中位数_Mbps>
```

jialin-desktop / jialin-laptop 分别执行：

```powershell
$edgeHost = "<EDGE_HOST>"
1..3 | ForEach-Object { iperf3 -c $edgeHost -p 5001 -t 10 }
$bwD2E = <本机三次结果的中位数_Mbps>
```

立即记录：

```text
bundle / timestamp / BW_nx_d2e / BW_nano_d2e
BW_jialin_desktop_d2e / BW_jialin_laptop_d2e / BW_edge_cloud
```

本 bundle 的 smoke、rep0 和 6 个正式 round 全部复用这五个值。

### 5.5 kaijie-laptop 终端 2：Edge runtime

```powershell
Set-Location "$HOME\Desktop\DSCI_SEAS"
conda activate DSCI
$bundle = "resnet50-cifar10"
$bwE2C = <本bundle实测E2C中位数_Mbps>
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID = "$bundle-edge-kaijie-laptop"
$env:DSCI_PYTORCH_DEVICE = "cpu"
$env:DSCI_TENSOR_TRANSPORT_DTYPE = "float32"
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id $bundle `
  --backend pytorch `
  --override-bw-e2c $bwE2C
```

另开终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:9002/status | ConvertTo-Json -Depth 6
Invoke-RestMethod http://<CLOUD_HOST>:32265/status | ConvertTo-Json -Depth 6
```

两边的 bundle、manifest、model hash、backend 必须一致；Edge 的 `BW_e2c` 必须等于 `$bwE2C`。

