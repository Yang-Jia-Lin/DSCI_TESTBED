# 新设备准备
> [!info]
> 本部分只在接入新机器时执行。已经部署完成的机器可直接进入“Pipeline 运行”。

#### 0. Git 准备
###### 1. 安装 Git，并确认终端能执行
```bash
git --version
```

###### 2. 配置全局用户信息（每台机器只需要配置一次）
```bash
git config --global user.name "yangjialin"
git config --global user.email "jialinyang6688@gmail.com"
```

###### 3. 生成 SSH Key（已有 `id_ed25519` 时跳过）
```bash
# Linux
ssh-keygen -t ed25519 -C "jialinyang6688@gmail.com"
cat ~/.ssh/id_ed25519.pub
```

```powershell
# Windows
ssh-keygen -t ed25519 -C "jialinyang6688@gmail.com"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

###### 4. 将完整公钥添加到 GitHub
```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

###### 5. 测试连接
```bash
ssh -T git@github.com
```

如果当前网络封锁 GitHub SSH 22 端口，可改用 443。在 Linux 的
`~/.ssh/config` 或 Windows 的 `$env:USERPROFILE\.ssh\config` 写入：
```text
Host github.com
  HostName ssh.github.com
  User git
  Port 443
```

然后重新执行：
```bash
ssh -T git@github.com
```

#### 1. 拉取代码
Linux：
```bash
cd ~/Desktop
git clone git@github.com:Yang-Jia-Lin/DSCI_TESTBED.git DSCI_SEAS
cd ~/Desktop/DSCI_SEAS
```

Windows PowerShell：
```powershell
Set-Location "$HOME\Desktop"
git clone git@github.com:Yang-Jia-Lin/DSCI_TESTBED.git DSCI_SEAS
Set-Location "$HOME\Desktop\DSCI_SEAS"
```

已有仓库时只更新：
```bash
git pull --ff-only
```

#### 2. 拷贝六个模型 Bundle
将 `six_bundles_weights.tar.gz` 放到仓库根目录后执行。

Linux：
```bash
cd ~/Desktop/DSCI_SEAS
tar -xzf six_bundles_weights.tar.gz
```

Windows PowerShell：
```powershell
Set-Location "$HOME\Desktop\DSCI_SEAS"
tar -xzf .\six_bundles_weights.tar.gz
```

确认六个目录均存在：
```text
Data/Bundles/resnet50-cifar10/
Data/Bundles/resnet50-imagenet100/
Data/Bundles/resnet50-neucls64/
Data/Bundles/vit-base-cifar10/
Data/Bundles/vit-base-imagenet100/
Data/Bundles/vit-base-neucls64/
```

每个 bundle 至少检查：
```text
weights.pth
manifest.json
exit_curves.csv
```


#### 3. 安装环境
###### Windows Conda
```powershell
conda create -n DSCI python=3.10 -y
conda activate DSCI
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m Src.Phase3_Runtime.Device.run_device --help
```

###### Linux Conda
```bash
conda create -n DSCI python=3.10 -y
conda activate DSCI
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m Src.Phase3_Runtime.Cloud.run_cloud --help
```

###### Raspberry Pi 5、Pi 4-1、Pi 4-2（64-bit Linux）conda
```bash
conda create -n DSCI python=3.10 -y
conda activate DSCI
python -m pip install --upgrade pip
python -m pip install torch torchvision
sed 's/scikit-learn==1.3.2/scikit-learn==1.4.2/' requirements.txt > /tmp/requirements-pi.txt
python -m pip install -r /tmp/requirements-pi.txt
python -m pip check
uname -m
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m Src.Phase3_Runtime.Device.run_device --help
```

###### Jetson Nano、Jetson NX Venv
> Jetson 的 torch 必须与 JetPack 匹配。不要用普通 `pip install torch` 覆盖 NVIDIA
> 版本。若系统环境已经能导入 CUDA torch，创建继承系统包的虚拟环境：

```bash
python3 -m venv --system-site-packages ~/venvs/DSCI
source ~/venvs/DSCI/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson-nano.txt
python -m pip check
cat /etc/nv_tegra_release
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m Src.Phase3_Runtime.Device.run_device --help
```

#### 4. 激活环境
Linux：
```bash
# conda
conda activate DSCI
# venv
source ./venvs/DSCI/bin/activate
```

Windows：
```powershell
# conda
conda activate DSCI
# venv
.\.venv\Scripts\Activate.ps1
```

#### 5. 连接端设备到边缘服务器
- 方法1：tailscale [Tailscale覆盖网络](04-Cards/Tailscale覆盖网络.md)
- 方法2：公网IP直接连接
- 方法3：边缘设备打开热点

Linux 查看地址：
```bash
ip addr
tailscale ip -4
```

Windows PowerShell：
```powershell
ipconfig
tailscale ip -4
```

#### 6. 配置IP `Src/Shared/Config/deploy_config.py`
核对：
1. 端能否访问 `edge_host` 和 `algo_host` 的IP
2. 边能否访问 `cloud_host` 的IP
```text
edge_host       当前 Edge/Scheduler 地址
algo_host       当前 Scheduler 地址，通常与 edge_host 相同
cloud_host      V100 地址

edge_feature_port   9001
edge_status_port    9002
cloud_feature_port  32266
cloud_status_port   32265
algo_server_port    8000
```

需要放通的端口：

| 机器           |      端口 | 用途                    |
| -------------- | --------: | ----------------------- |
| V100           | TCP 32264 | iperf3                  |
| V100           | TCP 32265 | Cloud 状态 HTTP         |
| V100           | TCP 32266 | Cloud 推理数据          |
| Edge/Scheduler |  TCP 5001 | iperf3                  |
| Edge/Scheduler |  TCP 8000 | Scheduler HTTP 控制接口 |
| Edge/Scheduler |  TCP 9001 | Edge 推理数据           |
| Edge/Scheduler |  TCP 9002 | Edge 状态 HTTP          |

V100 Linux 使用防火墙时：
```bash
sudo ufw allow 32264/tcp
sudo ufw allow 32265/tcp
sudo ufw allow 32266/tcp
```

Edge/Scheduler Windows 管理员 PowerShell：
```powershell
New-NetFirewallRule -DisplayName "DSCI Scheduler 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "DSCI Edge 9001" -Direction Inbound -Protocol TCP -LocalPort 9001 -Action Allow
New-NetFirewallRule -DisplayName "DSCI Edge Status 9002" -Direction Inbound -Protocol TCP -LocalPort 9002 -Action Allow
New-NetFirewallRule -DisplayName "DSCI Edge iperf 5001" -Direction Inbound -Protocol TCP -LocalPort 5001 -Action Allow
```

#### 7. 安装并检查 Iperf3
所有 Device、Edge 和 Cloud 都需要能够执行 `iperf3`。

Linux：
```bash
sudo apt update
sudo apt install -y iperf3
iperf3 --version
```

Windows PowerShell：
```powershell
Get-Command iperf3
iperf3 --version
```

如果 Windows 的 `iperf3.exe` 不在 `PATH`，可以指定完整路径：
```powershell
$env:DSCI_IPERF_EXE="C:\Tools\iperf3\iperf3.exe"
```

端口用途：

| 测量链路      | iperf 客户端 | iperf 服务端 |      端口 |
| ------------- | ------------ | ------------ | --------: |
| Device → Edge | Device       | Edge         |  TCP 5001 |
| Edge → Cloud  | Edge         | Cloud/V100   | TCP 32264 |

验收命令如下。

Edge 上先启动服务端：
```powershell
iperf3 -s -p 5001
```

在每台 Device 上测试端到边：
```bash
iperf3 -c <EDGE_HOST> -p 5001 -t 10
```

Cloud/V100 上启动服务端：
```bash
iperf3 -s -p 32264
```

然后在 Edge 上测试边到云：
```powershell
iperf3 -c <CLOUD_HOST> -p 32264 -t 10
```


#### 8. 生成该机器的六个模型 Profile
Profile 统一使用：`<bundle-id>-<role>-<machine>` 形式

| 角色   | 机器                   | ROLE_MACHINE            |
| ------ | ---------------------- | ----------------------- |
| Cloud  | V100 Linux             | `cloud-v100`            |
| Edge   | jialin-desktop Windows | `edge-jialin-desktop`   |
| Edge   | jialin-laptop Windows  | `edge-jialin-laptop`    |
| Edge   | kaijie-laptop Windows  | `edge-kaijie-laptop`    |
| Device | Raspberry Pi 5         | `device-pi5`            |
| Device | Raspberry Pi 4-1       | `device-pi4-1`          |
| Device | Raspberry Pi 4-2       | `device-pi4-2`          |
| Device | Jetson Nano            | `device-nano`           |
| Device | Jetson NX              | `device-nx`             |
| Device | jialin-desktop Windows | `device-jialin-desktop` |
| Device | jialin-laptop Windows  | `device-jialin-laptop`  |

##### Linux：V100、Pi、Nano、NX
先按上表设置本机唯一后缀，例如 Pi 5：
```bash
export ROLE_MACHINE=cloud-5880
```

依次生成六个 profile：
```bash
python -m Src.Phase1_Offline.Profiling.profile_segments \
  "resnet50-cifar10-${ROLE_MACHINE}" --bundle-id resnet50-cifar10 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments \
  "resnet50-imagenet100-${ROLE_MACHINE}" --bundle-id resnet50-imagenet100 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments \
  "resnet50-neucls64-${ROLE_MACHINE}" --bundle-id resnet50-neucls64 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments \
  "vit-base-cifar10-${ROLE_MACHINE}" --bundle-id vit-base-cifar10 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments \
  "vit-base-imagenet100-${ROLE_MACHINE}" --bundle-id vit-base-imagenet100 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments \
  "vit-base-neucls64-${ROLE_MACHINE}" --bundle-id vit-base-neucls64 \
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite
```

##### Windows：三个 Edge 候选和两个 Windows Device
先按上表设置角色，例如 jialin-desktop 作为 Edge：
```powershell
$ROLE_MACHINE="edge-kaijie-laptop"
```

依次生成六个 profile：
```powershell
python -m Src.Phase1_Offline.Profiling.profile_segments `
  "resnet50-cifar10-$ROLE_MACHINE" --bundle-id resnet50-cifar10 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments `
  "resnet50-imagenet100-$ROLE_MACHINE" --bundle-id resnet50-imagenet100 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments `
  "resnet50-neucls64-$ROLE_MACHINE" --bundle-id resnet50-neucls64 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments `
  "vit-base-cifar10-$ROLE_MACHINE" --bundle-id vit-base-cifar10 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments `
  "vit-base-imagenet100-$ROLE_MACHINE" --bundle-id vit-base-imagenet100 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite

python -m Src.Phase1_Offline.Profiling.profile_segments `
  "vit-base-neucls64-$ROLE_MACHINE" --bundle-id vit-base-neucls64 `
  --device auto --worker-count 1 --threads-per-worker 1 --warmup 5 --runs 30 --overwrite
```

#### 9. 汇总 Profile 到 Scheduler
> Scheduler 所在机器必须拥有本轮 Device、Edge、Cloud 的完整 profile 目录。
###### 方案 1：Git 同步
```bash
git add .
git commit -m "nano profile结果更新"
git push
```
然后在 Scheduler：
```bash
git pull --ff-only
```

###### 方案 2：直接复制到 Scheduler
```bash
scp -r Data/Profiles/<PROFILE_ID> <SCHEDULER_USER>@<SCHEDULER_HOST>:~/Desktop/DSCI_SEAS/Data/Profiles/
```
复制后在 Scheduler 上确认：
```powershell
Get-ChildItem .\Data\Profiles\<PROFILE_ID>
```

---

# Pipeline 运行

> [!IMPORTANT]
> 从本节开始，全文只使用一个 bundle 占位符：`__BUNDLE_ID__`。
> 在编辑器复制本节到新文档，然后全局查找 `__BUNDLE_ID__`，一次性替换为六个正式 bundle 之一，例如
> `resnet50-cifar10`。不要替换 profile 的角色/机器后缀。
>
> 当前六个正式 bundle：
> - `resnet50-cifar10`
> - `resnet50-imagenet100`
> - `resnet50-neucls64`
> - `vit-base-cifar10`
> - `vit-base-imagenet100`
> - `vit-base-neucls64`

#### 0. 进入仓库激活环境
Linux：
```bash
# conda
conda activate DSCI
```

```bash
# venv
source ./.venv/bin/activate
```

Windows：
```powershell
# conda
conda activate DSCI
```

```powershell
# venv
.\.venv\Scripts\Activate.ps1
```

#### 1. 启动 Cloud
终端 1：
```bash
iperf3 -s -p 32264
```

终端 2：
```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-cloud-v100"
python -m Src.Phase3_Runtime.Cloud.run_cloud \
  --bundle-id __BUNDLE_ID__ \
  --backend pytorch
```

#### 2. 启动 Edge
###### 终端1
```powershell
iperf3 -s -p 5001
```

###### 终端2
选择本轮实际使用的一台的ID，只选择一个：
```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-edge-jialin-desktop"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-edge-jialin-laptop"
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-edge-kaijie-laptop"
```
启动：
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
在任意 Device 检查：
```bash
curl http://<EDGE_HOST>:9002/status
```

#### 3. 启动 Scheduler
烟雾测试：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --fixed-split 3 10 `
  --fixed-threshold 1.0 `
  --no-auto-train `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

正常测试：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 1 `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

#### 4. 测试 1 个 Device
每次只选择下面一台 Device。Scheduler 必须使用 `--expected-users 1`。
###### Linux 做端
选择其中对应的设备：只选择一组！
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
export ROUND_ID=smoke-__BUNDLE_ID__-pi5-$(date +%Y%m%d-%H%M%S)

export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
export ROUND_ID=smoke-__BUNDLE_ID__-pi4-1-$(date +%Y%m%d-%H%M%S)

export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
export ROUND_ID=smoke-__BUNDLE_ID__-pi4-2-$(date +%Y%m%d-%H%M%S)

export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-nano"
export ROUND_ID=smoke-__BUNDLE_ID__-nano-$(date +%Y%m%d-%H%M%S)

export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-nx"
export ROUND_ID=smoke-__BUNDLE_ID__-nx-$(date +%Y%m%d-%H%M%S)

```

```bash
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id __BUNDLE_ID__ --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" \
  --test-package-mode balanced --test-package-samples-per-class 10 \
  --test-samples 3 --decision-timeout 900 \
  --dynamic-bandwidth \
  --bandwidth-ewma-alpha 0.3 \
  --bandwidth-change-threshold 0.20 \
  --bandwidth-min-reschedule-interval 30 \
  --bandwidth-stale-after 300 \
  --iperf-calibration-duration 3
```

###### Windows 做端
```powershell
$env:DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-jialin-desktop"
$env:ROUND_ID="smoke-__BUNDLE_ID__-jialin-desktop-$(Get-Date -Format yyyyMMdd-HHmmss)"

$env:DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-jialin-laptop"
$env:ROUND_ID="smoke-__BUNDLE_ID__-jialin-laptop-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

```powershell
python -m Src.Phase3_Runtime.Device.run_device `
  --bundle-id __BUNDLE_ID__ --backend pytorch `
  --user-id 0 --round-id "$env:ROUND_ID" `
  --test-package-mode balanced --test-package-samples-per-class 10 `
  --test-samples 3 --decision-timeout 900 `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```


#### 5. 多台 Device 联跑
> 示例4台 Device 参加，只使用其中一部分时，将 `--expected-users` 改为实际数量，并重新分配互不重复的 `user-id`。

停止旧 Scheduler，启动4用户训练策略：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 4 `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

> [!WARNING]
> 不要在4台机器分别运行 `date` 或 `Get-Date`，否则会生成七个不同 round。

在一台机器上只生成一次共同 ROUND_ID：
```bash
export ROUND_ID=smoke-7dev-__BUNDLE_ID__-$(date +%Y%m%d-%H%M%S)
echo "$ROUND_ID"
```

将输出原样粘贴
Linux：
```bash
export ROUND_ID=
```
Windows PowerShell：
```powershell
$env:ROUND_ID=""
```

各设备同时执行对应命令：
Pi 5，`user-id=0`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi5"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id __BUNDLE_ID__ --backend pytorch \
  --user-id 0 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 \
  --dynamic-bandwidth --bandwidth-ewma-alpha 0.3 --bandwidth-change-threshold 0.20 \
  --bandwidth-min-reschedule-interval 30 --bandwidth-stale-after 300 --iperf-calibration-duration 3
```

Pi 4-1，`user-id=1`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-1"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id __BUNDLE_ID__ --backend pytorch \
  --user-id 1 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 \
  --dynamic-bandwidth --bandwidth-ewma-alpha 0.3 --bandwidth-change-threshold 0.20 \
  --bandwidth-min-reschedule-interval 30 --bandwidth-stale-after 300 --iperf-calibration-duration 3
```

Pi 4-2，`user-id=2`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-pi4-2"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id __BUNDLE_ID__ --backend pytorch \
  --user-id 2 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 \
  --dynamic-bandwidth --bandwidth-ewma-alpha 0.3 --bandwidth-change-threshold 0.20 \
  --bandwidth-min-reschedule-interval 30 --bandwidth-stale-after 300 --iperf-calibration-duration 3
```

Nano，`user-id=3`：
```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID="__BUNDLE_ID__-device-nano"
python -m Src.Phase3_Runtime.Device.run_device --bundle-id __BUNDLE_ID__ --backend pytorch \
  --user-id 3 --round-id "$ROUND_ID" --test-package-mode balanced \
  --test-package-samples-per-class 10 --test-samples 3 --decision-timeout 900 \
  --dynamic-bandwidth --bandwidth-ewma-alpha 0.3 --bandwidth-change-threshold 0.20 \
  --bandwidth-min-reschedule-interval 30 --bandwidth-stale-after 300 --iperf-calibration-duration 3
```

至此，基础流程就已跑通，可以开始正式运行实验了。

---

# 高级：调度器使用
> [!warning]
> 以下命令中的 `N` 须替换为本轮实际 Device 数量。

#### 0. 自然运行
使用当前环境变量中的手动权重；没有可复用缓存时先返回默认解并后台训练：
```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="1"
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users N `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

#### 1. 固定切分（不训练）
固定 boundary 3 和 10，并关闭早退，确保请求经过 Device、Edge、Cloud：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users N `
  --fixed-split 3 10 `
  --fixed-threshold 1.0 `
  --no-auto-train `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

#### 2. 强制训练冷启动（从头训练）
```powershell
$env:DSCI_OBJECTIVE_ALPHA="1"
$env:DSCI_OBJECTIVE_BETA="1"
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users N `
  --force-retrain `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```
首个 round 会返回 `default:force_retrain` 并触发后台 cold training。等待 Health 中
`training_status=idle` 后，使用新的 ROUND_ID 再跑正式推理。

#### 3. 自动搜索参数（从头训练）
第一次搜索：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users N `
  --target-accuracy 0.90 `
  --force-retrain `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```
首个 round 返回 `constraint=pending`，后台最多训练 5 个 alpha 候选。检查：
```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health |
  Select-Object objective_mode,constraint_search_status,constraint_candidates_completed,selected_alpha,selected_beta,achieved_expected_accuracy,achieved_expected_latency
```
达到 `constraint_search_status=satisfied` 或 `unmet` 后，使用新的 ROUND_ID 正式推理。

如果需要重启 Scheduler，必须继续带相同目标，但不要再加 `--force-retrain`：
```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users N `
  --target-accuracy 0.90 `
  --dynamic-bandwidth `
  --bandwidth-ewma-alpha 0.3 `
  --bandwidth-change-threshold 0.20 `
  --bandwidth-min-reschedule-interval 30 `
  --bandwidth-stale-after 300 `
  --iperf-calibration-duration 3
```

#### 4. 检查连接
Scheduler 本机：
```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

Device 到 Scheduler：
```bash
ping -c 4 <SCHEDULER_HOST>
curl --connect-timeout 5 http://<SCHEDULER_HOST>:8000/api/v1/health
nc -vz -w 5 <SCHEDULER_HOST> 8000
```

Edge 到 Cloud：
```powershell
curl.exe http://<CLOUD_HOST>:32265/status
Test-NetConnection <CLOUD_HOST> -Port 32266
```

Device 到 Edge：
```bash
curl --connect-timeout 5 http://<EDGE_HOST>:9002/status
nc -vz -w 5 <EDGE_HOST> 9001
```

---

# 常见错误

###### `Selected bundle_id does not match segment profile`

环境变量选择了其他 bundle 的 profile。确认命令中的 `__BUNDLE_ID__`、profile ID 和
`metadata.json` 内的 `bundle_id` 完全一致；profile ID 区分大小写。

###### `Profile manifest_id does not match partition manifest` 或 Model Hash 错误

各机器上的 `weights.pth`、`manifest.json` 或 profile 不属于同一批 bundle。重新同步
整个 `Data/Bundles/__BUNDLE_ID__/`，再使用 `--overwrite` 重新生成 profile。

###### Scheduler 报找不到某个 Profile
该 profile 只存在于远端执行机器。将完整`Data/Profiles/<PROFILE_ID>/` 复制到 Scheduler，目录中必须同时存在 `metadata.json` 和 `segments.csv`。

###### Device 卡在等待决策
1. Scheduler 的 `--expected-users` 是否等于本轮实际 Device 数；
2. 所有 Device 是否使用完全相同的 ROUND_ID；
3. `user-id` 是否互不重复；
4. Scheduler 能否访问 Edge `9002/status` 和 Cloud `32265/status`；
5. 所有 Device 是否都已注册并进入同一个 request barrier。

###### `409 Conflict`
当前 Scheduler 仍有未完成 round，或重复使用了 round ID。每次重跑生成新的 ROUND_ID；
若上一轮因网络错误没有提交 measurements，重启 Scheduler 后再使用新 ROUND_ID。

###### `source=default` 且 `constraint=disabled`
重启 Scheduler 时漏掉了 `--target-accuracy`。若要复用自动搜索缓存，必须使用与训练时
完全相同的 `--target-accuracy` 和 `--accuracy-tolerance`，但不要加 `--force-retrain`。

###### 重启后 Health 显示 `constraint_search_status=idle`
缓存可能已经加载，但 Scheduler 尚未收到本轮 Device 状态。检查
`has_cached_solution=true`；设备注册后若状态一致，应返回 `cached_dsci:exact`。

###### Heartbeat、request Barrier 或 8000 端口超时
`8000` 是 Scheduler HTTP 控制接口；`9001` 和 `32266` 是张量数据通道。依次测试
ping、`curl /api/v1/health` 和 `nc 8000`。ping 延迟和波动很大时属于网络/Tailscale
问题，不是 PPO 或缓存错误。

###### Tailscale 很慢或 Iperf3 失败
烟雾测试可以使用：
```text
Edge:   --override-bw-e2c 50
Device: --override-bw-d2e 10
```
这只绕过测速，不代表真实链路已经被设置为 10/50 Mbps。正式带宽实验应移除 override
或使用实验规定的覆盖值，并在论文中记录。

###### Windows 环境变量设置后没有生效
PowerShell 的 `$env:...` 只在当前终端及其子进程生效。必须在启动对应 Edge、Scheduler
或 Device 的同一个 PowerShell 窗口中设置。

###### 自动准确率搜索耗时很长
自动模式最多训练 5 个 PPO 候选，首次运行可能需要几十分钟。不要看到单个候选的
`[Polish]` 就停止 Scheduler；以 `constraint_search_complete` 事件或 Health 的
`satisfied/unmet` 为完成标准。
