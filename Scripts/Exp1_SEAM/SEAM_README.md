# Exp1 SEAM 六模型真机实验操作手册

本文档用于在 5 台机器上完成 SEAM 主体性能实验：3 台 Device（Jetson NX、Jetson Nano、Raspberry Pi 5）、1 台 Windows Edge/Scheduler 和 1 台 V100 Cloud。实验比较：

- **SEAM**：联合优化模型切分点与 3 个早退阈值；
- **Static**：固定切分点和统一早退阈值 `0.7`；
- **模型与任务**：ResNet-50 / ViT-Base × CIFAR-10 / NEU-CLS-64 / ImageNet-100；
- **网络档位**：每台 Device 独立限制为低/中/高 `10/50/100 Mbps`；
- **指标**：Accuracy、端到端 Latency、Utility。

受控限速只复现带宽，不复现 WiFi/5G/网线各自的 RTT、抖动和丢包。因此论文表头应写“低/中/高带宽（10/50/100 Mbps）”，不能把三档直接写成真实 WiFi、5G、网线。

---

## 0. 当前仓库审计结论

截至 2026-07-15，本仓库的状态如下。

| 检查项 | 结论 |
| --- | --- |
| 六组新模型权重 | 已存在，且 `weights.pth` 与 manifest 中的 SHA-256 全部一致 |
| 模型扫描 / partition manifest | 已完成，六组均有 `manifest.json` 和 `layer_stats.csv` |
| 早退曲线 | 已完成，六组均有 `exit_curves.csv` |
| ResNet-50 出口 | 3 个：`after_layer1/2/3`，boundary `4/8/14`，final `19` |
| ViT-Base 出口 | 3 个：`after_block3/6/9`，boundary `4/7/10`，final `15` |
| 新模型硬件 profile | **尚未生成**；六组模型 × 五台机器共需 30 份 |
| 仓库现有 4 份 profile | 全部属于旧二出口 `resnet50-cifar10-ee-v1`，**不可复用** |
| 当前 Windows `DSCI` 环境 | `torch 2.9.1+cpu`、无 CUDA、缺少 `timm`；ViT 运行前必须补依赖 |

结论：**不要重新训练，也不要无条件重新扫描模型。当前首先要做的是五机环境预检和生成 30 份 segment profile。** 只有模型结构、出口位置或权重发生变化时，才重新生成 manifest、exit curves 和 profile。

六个正式 bundle ID 必须原样使用：

```text
resnet50-cifar10
resnet50-neucls64
resnet50-imagenet100
vit-base-cifar10
vit-base-neucls64
vit-base-imagenet100
```

每条运行命令都显式写 `--bundle-id`，不要依赖默认值，也不要使用带 `-ee-v1` 的旧 ID。

---

## 1. 固定实验拓扑和角色

| 机器 | 角色 | user_id | PyTorch device | worker | threads/worker | profile 前缀 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Jetson NX | Device | 0 | `cuda:0` | 1 | 1 | `device-nx-pytorch-` |
| Jetson Nano | Device | 1 | `cuda:0` | 1 | 1 | `device-nano-pytorch-` |
| Raspberry Pi 5 | Device | 2 | `cpu` | 1 | 4 | `device-pi5-pytorch-` |
| Windows 本机 | Edge + Scheduler | - | `cpu` | 1 | 10 | `edge-windows-pytorch-` |
| V100 服务器 | Cloud | - | `cuda:0` | 1 | 1 | `cloud-v100-pytorch-` |

当前 `Src/Shared/Config/deploy_config.py` 使用：

```text
Edge host     = 100.72.193.11
Scheduler host= 100.72.193.11
Cloud host    = 172.16.6.101
```

端口：

| 服务 | 主机 | 端口 | 协议 |
| --- | --- | ---: | --- |
| Scheduler API | Windows | 8000 | HTTP |
| Edge feature | Windows | 9001 | TCP |
| Edge status | Windows | 9002 | HTTP |
| Edge iperf | Windows | 5001 | iperf3 |
| Cloud feature | V100 | 32266 | TCP |
| Cloud status | V100 | 32265 | HTTP |
| Cloud iperf | V100 | 32264 | iperf3 |

特别注意：三台 Device 不仅要访问 Edge 和 Scheduler，还会直接查询 Cloud 的 `http://172.16.6.101:32265/status`。若 Device 无法路由到该地址，会在注册前失败。

---

## 2. 五台机器的一次性环境预检

以下命令都在仓库根目录执行。Linux 示例先进入实际仓库目录：

```bash
cd /path/to/DSCI_testbed
```

Windows：

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
```

### 2.1 确认代码版本一致

五台机器分别执行：

```bash
git rev-parse HEAD
git status --short
```

Windows 使用相同命令。五台机器的 commit 必须一致；正式实验前工作区应不存在来源不明的修改。

### 2.2 检查 Python、PyTorch 和 CUDA

五台机器分别执行：

```bash
conda run -n DSCI python -c "import os,platform,torch; print(platform.platform()); print('cpu_count=',os.cpu_count()); print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('cuda_count=',torch.cuda.device_count()); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

期望：

- NX、Nano、V100：`cuda=True`，且能打印正确 GPU 名称；
- Pi5、Windows：允许 `cuda=False`；
- 如果 NX、Nano、V100 显示 `cuda=False`，立即停止，不要让 `--device auto` 静默退回 CPU。

### 2.3 安装模型依赖，但不要重装 torch

现有依赖文件不包含 torch，可以在五台机器各自已有的 DSCI 环境中执行：

```bash
conda run -n DSCI python -m pip install -r Src/Phase1_Offline/Training/requirements-v100.txt
```

安装后验证：

```bash
conda run -n DSCI python -c "import torch,torchvision,timm,flask,requests,pandas,numpy; print('imports OK'); print('timm=',timm.__version__)"
```

必须看到 `timm=1.0.28`。Jetson 上不要执行任何会替换 NVIDIA/JetPack PyTorch 的安装命令；若 pip 计划卸载 torch，应取消并先修正环境。

### 2.4 检查系统工具

NX、Nano、Pi5：

```bash
command -v iperf3
command -v curl
command -v tc
```

V100：

```bash
command -v iperf3
command -v curl
```

Windows PowerShell：

```powershell
Get-Command iperf3
Get-Command curl.exe
Get-Command ssh
Get-Command scp
```

### 2.5 五机连通性检查

Windows Edge/Scheduler：

```powershell
Test-NetConnection 172.16.6.101 -Port 32264
Test-NetConnection 172.16.6.101 -Port 32265
Test-NetConnection 172.16.6.101 -Port 32266
```

三台 Device 分别执行：

```bash
nc -vz 100.72.193.11 8000
nc -vz 100.72.193.11 9001
nc -vz 100.72.193.11 9002
nc -vz 100.72.193.11 5001
nc -vz 172.16.6.101 32265
```

Cloud/Edge runtime 尚未启动时端口可能显示拒绝连接，但不能是路由超时。第 8 节启动服务后必须重新执行并全部成功。

---

## 3. 六组离线产物复核

### 3.1 Windows 上复核 manifest、权重哈希和三个出口

```powershell
conda run -n DSCI python -c "from Src.Shared.Config.paths import bundle_paths; from Src.Shared.Partitioning.manifest import load_partition_manifest,validate_model_file; ids=('resnet50-cifar10','resnet50-neucls64','resnet50-imagenet100','vit-base-cifar10','vit-base-neucls64','vit-base-imagenet100'); [(validate_model_file(load_partition_manifest(i),bundle_paths(i).weight_path),print(i,list(load_partition_manifest(i).exit_ids),list(load_partition_manifest(i).exit_boundary_ids),'hash OK')) for i in ids]"
```

每行必须显示 3 个出口及 `hash OK`。

### 3.2 检查 exit curve 列

```powershell
$bundles = @(
  "resnet50-cifar10", "resnet50-neucls64", "resnet50-imagenet100",
  "vit-base-cifar10", "vit-base-neucls64", "vit-base-imagenet100"
)
foreach ($bundle in $bundles) {
  Write-Host "==== $bundle ===="
  Get-Content -LiteralPath "Data\Bundles\$bundle\exit_curves.csv" -TotalCount 1
}
```

ResNet 表头必须包含 `after_layer1/2/3_rate` 和对应 accuracy；ViT 必须包含 `after_block3/6/9_rate` 和对应 accuracy。

### 3.3 检查正式 balanced 测试包

仓库中应存在：

```text
Data/Datasets/CIFAR10/TestSets/cifar10__test__balanced__10pc__seed42
Data/Datasets/NEU-CLS-64/TestSets/neucls64__test__balanced__10pc__seed42
Data/Datasets/ImageNet100/TestSets/imagenet100__test__balanced__10pc__seed42
```

样本总数分别为：

- CIFAR-10：10 类 × 10 = 100；
- NEU-CLS-64：6 类 × 10 = 60；
- ImageNet-100：100 类 × 10 = 1000。

三台 Device 必须使用完全相同的 package 内容和 seed。

### 3.4 什么时候才重新扫描

只有以下任一项变化时才执行：

```powershell
conda run -n DSCI python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id bundle_id_here --overwrite
conda run -n DSCI python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id bundle_id_here --overwrite
```

- `weights.pth` 改变；
- 模型 segment 定义改变；
- 出口数量、出口位置或分类头改变。

本次六个 bundle 不需要执行上述命令。

---

## 4. 生成 30 份硬件 segment profile

统一参数：

```text
worker_count = 1
warmup       = 10
runs         = 100
backend      = pytorch
```

profile 会绑定 `bundle_id + manifest_id + model_hash + backend + worker/thread`。任何一项不一致都会被运行时拒绝。

### 4.1 NX：生成 6 份 CUDA profile

在 NX 执行：

```bash
cd /path/to/DSCI_testbed
bundles=(
  resnet50-cifar10 resnet50-neucls64 resnet50-imagenet100
  vit-base-cifar10 vit-base-neucls64 vit-base-imagenet100
)
for bundle in "${bundles[@]}"; do
  conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments \
    "device-nx-pytorch-${bundle}" \
    --bundle-id "${bundle}" \
    --device cuda:0 \
    --worker-count 1 \
    --threads-per-worker 1 \
    --warmup 10 \
    --runs 100 \
    --overwrite || exit 1
done
```

### 4.2 Nano：生成 6 份 CUDA profile

在 Nano 执行：

```bash
cd /path/to/DSCI_testbed
bundles=(
  resnet50-cifar10 resnet50-neucls64 resnet50-imagenet100
  vit-base-cifar10 vit-base-neucls64 vit-base-imagenet100
)
for bundle in "${bundles[@]}"; do
  conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments \
    "device-nano-pytorch-${bundle}" \
    --bundle-id "${bundle}" \
    --device cuda:0 \
    --worker-count 1 \
    --threads-per-worker 1 \
    --warmup 10 \
    --runs 100 \
    --overwrite || exit 1
done
```

### 4.3 Pi5：生成 6 份 CPU profile

在 Pi5 执行：

```bash
cd /path/to/DSCI_testbed
bundles=(
  resnet50-cifar10 resnet50-neucls64 resnet50-imagenet100
  vit-base-cifar10 vit-base-neucls64 vit-base-imagenet100
)
for bundle in "${bundles[@]}"; do
  conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments \
    "device-pi5-pytorch-${bundle}" \
    --bundle-id "${bundle}" \
    --device cpu \
    --worker-count 1 \
    --threads-per-worker 4 \
    --warmup 10 \
    --runs 100 \
    --overwrite || exit 1
done
```

### 4.4 Windows Edge：生成 6 份 CPU profile

在 Windows PowerShell 执行：

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
$bundles = @(
  "resnet50-cifar10", "resnet50-neucls64", "resnet50-imagenet100",
  "vit-base-cifar10", "vit-base-neucls64", "vit-base-imagenet100"
)
foreach ($bundle in $bundles) {
  conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments `
    "edge-windows-pytorch-$bundle" `
    --bundle-id $bundle `
    --device cpu `
    --worker-count 1 `
    --threads-per-worker 10 `
    --warmup 10 `
    --runs 100 `
    --overwrite
  if ($LASTEXITCODE -ne 0) { throw "Edge profile failed: $bundle" }
}
```

### 4.5 V100 Cloud：生成 6 份 CUDA profile

在 V100 执行：

```bash
cd /path/to/DSCI_testbed
bundles=(
  resnet50-cifar10 resnet50-neucls64 resnet50-imagenet100
  vit-base-cifar10 vit-base-neucls64 vit-base-imagenet100
)
for bundle in "${bundles[@]}"; do
  conda run -n DSCI python -m Src.Phase1_Offline.Profiling.profile_segments \
    "cloud-v100-pytorch-${bundle}" \
    --bundle-id "${bundle}" \
    --device cuda:0 \
    --worker-count 1 \
    --threads-per-worker 1 \
    --warmup 10 \
    --runs 100 \
    --overwrite || exit 1
done
```

### 4.6 把远端 24 份 profile 复制到 Windows Scheduler

Windows 自己已有 6 份 Edge profile。再从四台远端机器各拉取 6 份。将示例中的 SSH 用户、IP 和 `/path/to/DSCI_testbed` 换成实际值：

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
scp -r nx_user@nx_ip:/path/to/DSCI_testbed/Data/Profiles/device-nx-pytorch-* .\Data\Profiles\
scp -r nano_user@nano_ip:/path/to/DSCI_testbed/Data/Profiles/device-nano-pytorch-* .\Data\Profiles\
scp -r pi_user@pi5_ip:/path/to/DSCI_testbed/Data/Profiles/device-pi5-pytorch-* .\Data\Profiles\
scp -r cloud_user@v100_ip:/path/to/DSCI_testbed/Data/Profiles/cloud-v100-pytorch-* .\Data\Profiles\
```

不要删除各节点本地的 profile：NX/Nano/Pi5/Cloud runtime 各自也要读取本机对应目录。

### 4.7 Scheduler 上验收 30 份 profile

```powershell
conda run -n DSCI python -c "from Scripts.EvaluationCommon.config import EXPECTED_BUNDLES; from Scripts.EvaluationCommon.artifacts import check_bundle_readiness; [(print(x.bundle_id,x.status,x.device_nx_profiles,x.device_nano_profiles,x.device_pi5_profiles,x.edge_profiles,x.cloud_profiles,x.notes)) for x in map(check_bundle_readiness,EXPECTED_BUNDLES)]"
```

六行都必须满足：

```text
status=ready
NX=1 Nano=1 Pi5=1 Edge=1 Cloud=1
notes=ok
```

逐份 metadata 还应满足：

```powershell
conda run -n DSCI python -c "import json,pathlib; roots=pathlib.Path('Data/Profiles').glob('*/metadata.json'); [(lambda d,p: print(p.parent.name,d['bundle_id'],d['manifest_id'],d['backend'],d['worker_count'],d['threads_per_worker'],sorted(d['exit_head_latencies_s'])))(json.loads(p.read_text()),p) for p in roots if any(x in p.parent.name for x in ('device-nx-','device-nano-','device-pi5-','edge-windows-','cloud-v100-'))]"
```

每份 `exit_head_latencies_s` 必须有 3 个键。

---

## 5. 统一实验参数

正式实验固定：

```text
alpha = 1
beta  = 20
tensor transport dtype = float32
expected users = 3
request barrier = enabled
test package = balanced / test / 10 samples per class / seed 42
repeats = 3
```

Scheduler 和三台 Device 都必须在启动 Python **之前**设置 alpha/beta。否则 Scheduler 的 predicted utility 与 Device 的 observed utility 口径不同。

Linux Device：

```bash
export DSCI_OBJECTIVE_ALPHA=1
export DSCI_OBJECTIVE_BETA=20
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
```

Windows Scheduler：

```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
```

静态下界：

| 架构 | fixed split | fixed threshold | 出口分布 |
| --- | --- | ---: | --- |
| ResNet-50 | `(4, 8)` | 0.7 | Device boundary 4、Edge boundary 8、Cloud boundary 14 |
| ViT-Base | `(4, 7)` | 0.7 | Device boundary 4、Edge boundary 7、Cloud boundary 10 |

---

## 6. 三台 Device 的真实 D2E 限速

三台 Device 分别独立限速，并使用不同 link ID：

| Device | link ID |
| --- | --- |
| NX | `nx-d2e` |
| Nano | `nano-d2e` |
| Pi5 | `pi5-d2e` |

### 6.1 自动识别到 Edge 的网卡

三台 Device 分别执行：

```bash
EDGE_IP=100.72.193.11
IFACE=$(ip route get "$EDGE_IP" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "D2E interface=${IFACE}"
test -n "$IFACE"
```

### 6.2 设置某一档带宽

低档：

```bash
BW=10
sudo tc qdisc replace dev "$IFACE" root tbf rate "${BW}mbit" burst 256kbit latency 400ms
sudo tc -s qdisc show dev "$IFACE"
```

中档只把 `BW=50`，高档只把 `BW=100`，其余命令相同。

### 6.3 正式轮次前逐台验证

Windows 先启动：

```powershell
iperf3 -s -p 5001
```

三台 Device **依次**执行，避免 iperf busy：

```bash
iperf3 -c 100.72.193.11 -p 5001 -t 10
```

实际吞吐应接近当前目标。若显著低于目标，说明底层链路已经更慢；该档不得开始正式轮次。Device 正式命令还会传 `--override-bw-d2e "$BW"`，这是让 Scheduler 使用与真实 tc 限速一致的数值，并不是替代 tc。

### 6.4 全部实验结束后恢复

三台 Device 分别执行：

```bash
sudo tc qdisc del dev "$IFACE" root
```

若提示不存在 root qdisc，可忽略；随后用 `tc qdisc show dev "$IFACE"` 确认 TBF 已移除。

---

## 7. 每个 bundle 的服务启动顺序

每切换一个 bundle，Cloud 和 Edge runtime 都要停止并用新 profile 重启。对同一 bundle 的三个 D2E 档位，Cloud/Edge 可以保持运行；Edge 启动时只测一次 E2C，训练预热与正式重复期间不要重启 Edge，否则 E2C 状态可能变化，导致 cache 不再 exact。

固定顺序：

1. V100：Cloud iperf；
2. V100：Cloud runtime；
3. Windows：Edge iperf；
4. Windows：Edge runtime；
5. Windows：Scheduler；
6. NX、Nano、Pi5：使用同一 round ID 启动 Device。

以下命令中的 `$BUNDLE` / `$bundle` 必须是同一个正式 bundle ID。

### 7.1 V100 终端 1：Cloud iperf

```bash
iperf3 -s -p 32264
```

### 7.2 V100 终端 2：Cloud runtime

```bash
cd /path/to/DSCI_testbed
export BUNDLE=resnet50-cifar10
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="cloud-v100-pytorch-${BUNDLE}"
export DSCI_PYTORCH_DEVICE=cuda:0
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
conda run -n DSCI python -m Src.Phase3_Runtime.Cloud.run_cloud \
  --bundle-id "$BUNDLE" \
  --backend pytorch
```

启动后，在 V100 本机检查：

```bash
curl http://127.0.0.1:32265/status
```

### 7.3 Windows 终端 1：Edge iperf

```powershell
iperf3 -s -p 5001
```

### 7.4 Windows 终端 2：Edge runtime

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
$bundle = "resnet50-cifar10"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID = "edge-windows-pytorch-$bundle"
$env:DSCI_PYTORCH_DEVICE = "cpu"
$env:DSCI_TENSOR_TRANSPORT_DTYPE = "float32"
conda run -n DSCI python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id $bundle `
  --backend pytorch
```

Edge 会调用 V100 的 `iperf3 :32264` 测 E2C。启动日志必须打印正数 `BW_e2c`，随后检查：

```powershell
Invoke-RestMethod http://127.0.0.1:9002/status | ConvertTo-Json -Depth 6
Invoke-RestMethod http://172.16.6.101:32265/status | ConvertTo-Json -Depth 6
```

两份状态的 bundle、manifest、model hash、backend 必须一致。

---

## 8. Device 通用命令

### 8.1 round ID 规则

格式：

```text
<method>_<bundle>_bw<10|50|100>_rep<0|1|2|3>_<timestamp>
```

- `rep0` 只用于 SEAM 训练预热，结果丢弃；
- 正式重复使用 `rep1/rep2/rep3`；
- 同一轮三台 Device 使用完全相同的 round ID；
- `user_id` 分别固定为 0/1/2；
- 完成或失败的 round ID 都不能复用。

在 NX 生成并复制给 Nano、Pi5：

```bash
METHOD=seam
BUNDLE=resnet50-cifar10
BW=10
REP=0
ROUND_ID="${METHOD}_${BUNDLE}_bw${BW}_rep${REP}_$(date +%Y%m%d-%H%M%S)"
echo "$ROUND_ID"
```

### 8.2 NX：user 0

```bash
cd /path/to/DSCI_testbed
export BUNDLE=resnet50-cifar10
export BW=10
export ROUND_ID="paste_the_shared_round_id_here"
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="device-nx-pytorch-${BUNDLE}"
export DSCI_PYTORCH_DEVICE=cuda:0
export DSCI_OBJECTIVE_ALPHA=1
export DSCI_OBJECTIVE_BETA=20
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
conda run -n DSCI python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id "$BUNDLE" \
  --backend pytorch \
  --user-id 0 \
  --round-id "$ROUND_ID" \
  --test-package-mode balanced \
  --test-package-split test \
  --test-package-samples-per-class 10 \
  --test-package-seed 42 \
  --override-bw-d2e "$BW" \
  --d2e-link-id nx-d2e \
  --d2e-capacity-mbps "$BW" \
  --decision-mode dsci \
  --decision-timeout 300
```

### 8.3 Nano：user 1

```bash
cd /path/to/DSCI_testbed
export BUNDLE=resnet50-cifar10
export BW=10
export ROUND_ID="paste_the_shared_round_id_here"
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="device-nano-pytorch-${BUNDLE}"
export DSCI_PYTORCH_DEVICE=cuda:0
export DSCI_OBJECTIVE_ALPHA=1
export DSCI_OBJECTIVE_BETA=20
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
conda run -n DSCI python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id "$BUNDLE" \
  --backend pytorch \
  --user-id 1 \
  --round-id "$ROUND_ID" \
  --test-package-mode balanced \
  --test-package-split test \
  --test-package-samples-per-class 10 \
  --test-package-seed 42 \
  --override-bw-d2e "$BW" \
  --d2e-link-id nano-d2e \
  --d2e-capacity-mbps "$BW" \
  --decision-mode dsci \
  --decision-timeout 300
```

### 8.4 Pi5：user 2

```bash
cd /path/to/DSCI_testbed
export BUNDLE=resnet50-cifar10
export BW=10
export ROUND_ID="paste_the_shared_round_id_here"
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="device-pi5-pytorch-${BUNDLE}"
export DSCI_PYTORCH_DEVICE=cpu
export DSCI_OBJECTIVE_ALPHA=1
export DSCI_OBJECTIVE_BETA=20
export DSCI_TENSOR_TRANSPORT_DTYPE=float32
conda run -n DSCI python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id "$BUNDLE" \
  --backend pytorch \
  --user-id 2 \
  --round-id "$ROUND_ID" \
  --test-package-mode balanced \
  --test-package-split test \
  --test-package-samples-per-class 10 \
  --test-package-seed 42 \
  --override-bw-d2e "$BW" \
  --d2e-link-id pi5-d2e \
  --d2e-capacity-mbps "$BW" \
  --decision-mode dsci \
  --decision-timeout 300
```

正式实验不要加 `--no-request-barrier`。三台设备会在每个样本前等待统一 release，屏障等待不计入 `T_total`。

---

## 9. 先做单样本 smoke test

完整矩阵之前只使用：

```text
bundle = resnet50-cifar10
bandwidth = 10 Mbps
samples = 1 per Device
```

1. 按第 6 节给三台 Device 设置 `BW=10`；
2. 按第 7 节启动 Cloud 和 Edge；
3. Windows 启动固定静态 Scheduler：

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3 `
  --fixed-split 4 8 `
  --fixed-threshold 0.7 `
  --no-auto-train
```

4. 三台 Device 的第 8 节命令末尾临时增加：

```text
--test-samples 1
```

5. 验收：

- Scheduler 收到 user `0/1/2`；
- 三台设备的决策都包含 `after_layer1/2/3` 三个阈值；
- `partition_boundary_1=4`、`partition_boundary_2=8`；
- 三台设备各生成 1 条 measurement；
- `request_trace.total_latency` 与 `T_total` 一致；
- 已知互斥分项加 `unattributed_overhead` 闭合到总时延；
- 三台设备均成功提交 measurements。

smoke 未通过时不要继续正式矩阵。

---

## 10. Static 正式实验

每个 bundle、每个带宽档分别跑 3 个正式 round。Cloud/Edge 命令中的 bundle/profile 必须同步切换。

### 10.1 ResNet Static Scheduler

```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3 `
  --fixed-split 4 8 `
  --fixed-threshold 0.7 `
  --no-auto-train
```

### 10.2 ViT Static Scheduler

```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3 `
  --fixed-split 4 7 `
  --fixed-threshold 0.7 `
  --no-auto-train
```

### 10.3 每档的操作

1. 三台 Device 设置同一个 `BW=10/50/100`，并完成 iperf 验证；
2. 启动相应 Static Scheduler；
3. 生成 `static_<bundle>_bw<BW>_rep1_<timestamp>`；
4. 三台 Device 同时运行完整 balanced 包；
5. 重复 `rep2`、`rep3`，每次使用新 round ID；
6. 检查 summary 中的固定 split、三个 `0.7` 阈值、样本数和 `decision_source`。

---

## 11. SEAM cache 预热和正式实验

当前在线 Scheduler 第一次看到新状态时会先返回 `default`，同时在后台训练 DSCI。**第一次返回的不是正式 SEAM 决策，必须作为 rep0 丢弃。**

每个 `bundle × bandwidth` 都执行一次预热，共 18 次。

### 11.1 启动允许后台训练的 Scheduler

Windows：

```powershell
Set-Location D:\Coding\Python\DSCI_testbed
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3
```

### 11.2 运行 rep0 预热轮

三台 Device 使用：

```text
seam_<bundle>_bw<BW>_rep0_<timestamp>
```

其余参数与第 8 节一致。rep0 只负责向 Scheduler 提供确定状态并触发训练，产生的 Accuracy/Latency/Utility 不进入论文结果。

### 11.3 等待训练完成

预热轮完成后不要关闭 Scheduler。另开 Windows PowerShell：

```powershell
do {
  Start-Sleep -Seconds 10
  $health = Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
  $health | Select-Object training_status,has_cached_solution,cache_entries,last_error,last_training_duration_s
} while ($health.training_status -eq "running")

if ($health.training_status -ne "idle") { throw "SEAM training failed: $($health.last_error)" }
if (-not $health.has_cached_solution) { throw "No cached SEAM solution" }
```

同时确认：

```text
Data/Runtime/SolutionCache/latest_solution.npz
Data/Runtime/SolutionCache/latest_solution_meta.json
```

已更新。

### 11.4 重启 Scheduler，正式阶段禁止再训练

按 `Ctrl+C` 停止刚才的 Scheduler，保持 Cloud、Edge 和三档网络状态不变，然后启动：

```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="20"
$env:DSCI_TENSOR_TRANSPORT_DTYPE="float32"
conda run -n DSCI python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 3 `
  --no-auto-train
```

### 11.5 运行 rep1/rep2/rep3

依次生成：

```text
seam_<bundle>_bw<BW>_rep1_<timestamp>
seam_<bundle>_bw<BW>_rep2_<timestamp>
seam_<bundle>_bw<BW>_rep3_<timestamp>
```

每轮结束后，三台 Device 的 summary 都必须满足：

```text
decision.decision_source = cached_dsci:exact
```

以下来源均不得计入正式 SEAM 结果：

```text
default
cached_dsci:warm:*
cached_dsci:reuse:*
```

出现非 exact 通常表示 E2C 状态、带宽、profile、worker 数、bundle 或 link ID 与预热轮不同。保持当前 Cloud/Edge 进程不变，修正状态后重新做 rep0 训练。

Scheduler 最多保留有限数量历史 cache；完成一个 `bundle × bandwidth` 的三个正式重复后再切换下一个条件。若以后回跑较早条件且找不到 exact cache，重新做该条件的 rep0。

---

## 12. 完整实验矩阵和数量

正式矩阵：

```text
6 bundles × 3 bandwidths × 2 methods × 3 repeats = 108 个正式 round
```

SEAM 额外需要：

```text
6 bundles × 3 bandwidths = 18 个 rep0 训练预热 round
```

建议顺序：

1. `resnet50-cifar10`：10 → 50 → 100 Mbps；每档先 Static 3 次，再 SEAM rep0 + 正式 3 次；
2. `resnet50-neucls64`：同上；
3. `resnet50-imagenet100`：同上；
4. `vit-base-cifar10`：同上；
5. `vit-base-neucls64`：同上；
6. `vit-base-imagenet100`：同上。

每切换 bundle 重启 Cloud/Edge；每切换带宽重新设置三台 Device 的 tc 并逐台验证。

---

## 13. 结果文件和集中收集

每台 Device 本地生成：

```text
Data/Runtime/DeviceResults/<round_id>/user_<id>_measurements.jsonl
Data/Runtime/DeviceResults/<round_id>/user_<id>_summary.json
Data/Runtime/DeviceResults/<round_id>/user_<id>_inference_results.csv
```

Windows 集中目录约定：

```text
Scripts/Results/Exp1_SEAM/Raw/<bundle>/bw_<10|50|100>/<static|seam>/rep_<1|2|3>/<nx|nano|pi5>/
```

示例：

```powershell
$bundle = "resnet50-cifar10"
$bw = 10
$method = "seam"
$rep = 1
$round = "paste_the_full_round_id_here"
$root = "Scripts\Results\Exp1_SEAM\Raw\$bundle\bw_$bw\$method\rep_$rep"
New-Item -ItemType Directory -Force "$root\nx","$root\nano","$root\pi5" | Out-Null
scp -r nx_user@nx_ip:/path/to/DSCI_testbed/Data/Runtime/DeviceResults/$round/* "$root\nx\"
scp -r nano_user@nano_ip:/path/to/DSCI_testbed/Data/Runtime/DeviceResults/$round/* "$root\nano\"
scp -r pi_user@pi5_ip:/path/to/DSCI_testbed/Data/Runtime/DeviceResults/$round/* "$root\pi5\"
```

拷贝后立即记录：

```text
bundle / method / bandwidth / repeat / round_id
三个 device profile ID
edge profile ID / cloud profile ID
实际 E2C iperf 值
三个 D2E tc 目标和验证值
alpha / beta
git commit
```

---

## 14. 指标汇总口径

### 14.1 Accuracy

不能简单平均不同样本数的 summary。统一使用：

```text
accuracy = (NX correct + Nano correct + Pi5 correct)
           / (NX samples + Nano samples + Pi5 samples)
```

### 14.2 Latency

使用三台设备全部请求的 `T_total`：

```text
latency_mean_s = sum(all request T_total) / total request count
latency_mean_ms = latency_mean_s × 1000
```

不要把屏障等待计入 `T_total`；不要使用 Scheduler 求解时间替代端到端 latency。

### 14.3 Utility

请求级定义：

```text
observed_utility = 1 × is_correct - 20 × T_total_seconds
```

整轮：

```text
utility_mean = mean(all request observed_utility)
```

应复算并与 JSONL 中 `observed_utility` 对比，误差只允许来自浮点精度。

### 14.4 三次重复

每个 `bundle × bandwidth × method` 分别对三次 round 的 Accuracy、Latency、Utility 报告：

```text
mean ± sample standard deviation
```

论文表格同时保留 Static 与 SEAM，不要把两种方法混到同一个均值。

### 14.5 RequestTrace 闭合

每条请求应满足：

```text
device_compute
+ d2e_transport
+ edge_queue
+ edge_segment_compute
+ edge_exit_head_compute
+ edge_exit_check
+ e2c_transport
+ cloud_queue
+ cloud_segment_compute
+ cloud_exit_head_compute
+ cloud_exit_check
+ unattributed_overhead
= total_latency ≈ T_total
```

`unattributed_overhead` 包含序列化、socket 协议和尚未独立计时的 Python 开销，不能直接解释为网络传输。

---

## 15. 每轮结束检查清单

- [ ] 三台 Device 的 round ID 完全一致，user ID 为 0/1/2；
- [ ] 三台 Device 的 `samples` 等于该数据集完整 balanced 包大小；
- [ ] `alpha=1`、`beta=20`；
- [ ] `tensor_transport_dtype=float32`；
- [ ] profile 的 bundle、manifest、hash 与 Cloud/Edge/三台 Device 一致；
- [ ] 决策中恰好有 3 个 `exit_thresholds`；
- [ ] Static split/threshold 与架构约定一致；
- [ ] SEAM 正式轮为 `cached_dsci:exact`；
- [ ] `T_total`、accuracy、utility 非空；
- [ ] RequestTrace 闭合；
- [ ] 三台设备的结果已复制到 Windows 集中目录；
- [ ] rep0 未混入正式统计。

---

## 16. 常见问题与处理

| 现象 | 原因与处理 |
| --- | --- |
| `Selected bundle_id does not match segment profile` | 使用了旧 `*-ee-v1` profile 或选错 bundle。重新设置本机对应的新 profile ID。 |
| `Profile manifest/model hash mismatch` | profile 来自不同权重或扫描结果。确认五机权重一致后在该硬件重新 profile。 |
| `No module named timm` | ViT 依赖未安装。按 2.3 节安装并验证，不能只同步权重。 |
| NX/Nano/V100 日志显示 CPU | CUDA PyTorch 不可用或 `DSCI_PYTORCH_DEVICE` 未设。停止实验，先修复环境。 |
| Scheduler 一直等待 | `--expected-users` 不是 3、三台 Device round ID 不同、某台未注册或心跳超时。 |
| `409 Conflict` | 复用了完成/失败的 round ID，或同一 user ID 用不同状态重复注册。生成全新 ID。 |
| `decision_source=default` | 这是首次状态或 exact cache 不存在，只能作为 rep0；等待训练后重启 `--no-auto-train`。 |
| `cached_dsci:warm/reuse` | 当前状态与 cache 不完全一致。检查 E2C、BW、link ID、profile、worker、bundle，然后重新预热。 |
| `iperf3: server is busy` | 三台 Device 同时测量。正式命令使用 override；预检时三台依次运行 iperf。 |
| tc 后吞吐没有变化 | `IFACE` 选错、qdisc 未替换或实际走了另一条路由。重新用 `ip route get` 和 `tc -s` 检查。 |
| 100 Mbps 档实测远低于 100 | 底层物理链路不足，不能把该轮标作 100 Mbps。先改善链路或降低档位定义。 |
| Device 访问 Cloud status 超时 | Device 必须能直达 `172.16.6.101:32265`；检查路由、VPN/Tailscale 和防火墙。 |
| Scheduler 找不到远端 profile | Scheduler 会按 Device 上报的 ID在自己的 `Data/Profiles` 加载；把远端 profile 复制到 Windows。 |
| 三台 samples 不一致 | 测试包或 `--test-samples` 不一致。正式轮移除 `--test-samples` 并统一 balanced 10pc 包。 |
| ViT 在 Nano/Pi5 内存不足 | 降低其他进程占用；不能擅自改 worker/profile。若仍失败，记录为硬件不可运行并停止该配置。 |

---

## 17. 本实验明确不做的事情

- 不重新训练模型或早退头；
- 不使用旧二出口 profile；
- 不使用 MNN backend；
- 不把 tc 带宽档称为真实 WiFi/5G/网线；
- 不把 SEAM rep0/default 决策计入正式结果；
- 不关闭多 Device 请求屏障；
- 不用 predicted latency 替代 Phase3 的 observed `T_total`；
- 不在本 README 中提供批量启动器或绘图脚本。
