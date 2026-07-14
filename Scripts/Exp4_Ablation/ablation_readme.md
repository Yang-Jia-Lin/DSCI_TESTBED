# Exp2 Ablation 真机实验操作说明

本文档面向两台端设备联跑的真机消融实验。核心思路是：Cloud、Edge runtime 只启动一次；每个消融方法单独启动一次 Scheduler；两台 Device 使用同一个 `round_id`、不同 `user_id`，共同完成同一轮实验。

## 1. 建议对比方法

推荐先跑主表 7 组，能够分别回答“纯本地、纯云、纯早退、纯切分、切分+早退、调度器优化”各自的贡献。

| case | 目的 | Scheduler 参数 | 说明 |
| --- | --- | --- | --- |
| `local_no_ee` | 纯本地推理 | `--fixed-split 19 19 --fixed-threshold 1.0 --no-auto-train` | 全模型都在 Device，本组作为本地 baseline。 |
| `cloud_no_ee` | 纯云推理 | `--fixed-split 0 0 --fixed-threshold 1.0 --no-auto-train` | Device 不算模型，输入经 Edge 转发到 Cloud。 |
| `edge_no_ee` | 纯边缘推理 | `--fixed-split 0 19 --fixed-threshold 1.0 --no-auto-train` | Device 不算模型，Edge 完成全模型。 |
| `local_ee` | 纯早退收益 | `--fixed-split 19 19 --fixed-threshold 0.7 --no-auto-train` | 所有计算仍在 Device，只打开 early exit。 |
| `split_de_no_ee` | 纯 Device-Edge 切分 | `--fixed-split 7 19 --fixed-threshold 1.0 --no-auto-train` | Device 算到 boundary 7，Edge 完成剩余。 |
| `split_dec_no_ee` | 纯三层切分 | `--fixed-split 7 13 --fixed-threshold 1.0 --no-auto-train` | Device、Edge、Cloud 都参与，无 early exit。 |
| `ours_schedule` | 调度器联合优化 | `--no-auto-train` 或不加固定参数 | 使用 Scheduler 的 DSCI/cache 决策，得到 split + threshold。 |

可选扩展组：

| case | Scheduler 参数 | 用途 |
| --- | --- | --- |
| `edge_ee` | `--fixed-split 0 19 --fixed-threshold 0.7 --no-auto-train` | 看 early exit 放在 Edge 上是否有效。 |
| `cloud_ee` | `--fixed-split 0 0 --fixed-threshold 0.7 --no-auto-train` | 看云端 early exit 的收益，通常主要体现计算减少，不减少上行传输。 |
| `split_dec_ee` | `--fixed-split 7 13 --fixed-threshold 0.7 --no-auto-train` | 静态三层切分 + early exit，用来和 `ours_schedule` 比较联合优化是否更好。 |

当前 bundle `resnet50-cifar10-ee-v1` 的 final boundary 是 `19`，early-exit boundary 是 `8` 和 `14`。`--fixed-threshold 1.0` 近似表示关闭 early exit；`0.7` 是真机消融的默认统一阈值，可根据需要再 sweep `0.5/0.7/0.9`。

## 2. 运行前检查

1. 在所有机器上确认代码、数据和模型包一致。
2. 检查 `Src/Shared/Config/deploy_config.py`：
   - `edge_host` 指向 Edge 机器。
   - `cloud_host` 指向 Cloud 机器。
   - `algo_host` 指向 Scheduler 机器。
3. Scheduler 机器必须拥有所有 profile：
   - `Data/Profiles/<device0_profile>/`
   - `Data/Profiles/<device1_profile>/`
   - `Data/Profiles/<edge_profile>/`
   - `Data/Profiles/<cloud_profile>/`
4. 两台 Device 使用不同 profile、不同 `--user-id`，但每个 case 使用同一个 `--round-id`。
5. 建议每个 case 使用相同样本数，例如 `--test-samples 500`。调试时可先用 `--test-samples 20`。

## 3. 启动 Cloud

Cloud 终端 1：启动 iperf。

```bash
iperf3 -s -p 32264
```

Cloud 终端 2：启动 Cloud runtime。

```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID=cloud-v100-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Cloud.run_cloud \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch
```

## 4. 启动 Edge

Edge 终端 1：启动 iperf。

```powershell
iperf3 -s -p 5001
```

Edge 终端 2：启动 Edge runtime。把 profile 改成本机实际值。

```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="edge-jialindesktop-pytorch-resnet50-cifar10"
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id resnet50-cifar10-ee-v1 `
  --backend pytorch
```

Cloud 和 Edge runtime 启动后，每个消融 case 不需要重启它们。

## 5. 每个 case 的通用流程

以下步骤对每个 case 重复执行。

### 5.1 启动对应 Scheduler

Scheduler 必须使用 `--expected-users 2`。例如跑 `local_no_ee`：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 2 `
  --fixed-split 19 19 `
  --fixed-threshold 1.0 `
  --no-auto-train
```

跑完一个 case 后，按 `Ctrl+C` 停止 Scheduler，再用下一组参数重新启动。

`ours_schedule` 的 Scheduler 推荐先用：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server `
  --expected-users 2 `
  --no-auto-train
```

如果你希望让 Scheduler 在后台自动训练/更新缓存，则去掉 `--no-auto-train`。为了可复现，正式记录结果时建议固定一种设置并写进实验日志。

### 5.2 设置本轮 round_id

在 Device 0 上生成新的 `ROUND_ID`，并复制给 Device 1。建议把 case 名放进 round_id，后续整理结果更方便。

```bash
export CASE=local_no_ee
export ROUND_ID=${CASE}_$(date +%Y%m%d-%H%M%S)
echo $ROUND_ID
```

Device 1 使用复制过来的同一个值：

```bash
export CASE=local_no_ee
export ROUND_ID=local_no_ee_20260709-153000
```

不要复用已经完成过的 `ROUND_ID`。同一个 Scheduler 进程里，一个完成或失败的 round_id 不能再次注册。

### 5.3 启动 Device 0

把 profile 改成 Device 0 的实际 profile。

```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nx1-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch \
  --user-id 0 \
  --round-id "$ROUND_ID" \
  --test-samples 10
```

### 5.4 启动 Device 1

把 profile 改成 Device 1 的实际 profile。`--round-id` 必须和 Device 0 完全一致。

```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nano1-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch \
  --user-id 1 \
  --round-id "$ROUND_ID" \
  --test-samples 10
```

两台 Device 都注册后，Scheduler 会进入 `optimizing/ready`，随后两个 Device 开始推理。若其中一台启动慢，另一台会等待决策。

## 6. 全部 case 的 Scheduler 命令清单

按下面顺序跑即可。每跑完一组，停止 Scheduler，重启下一组 Scheduler，然后两台 Device 用新的 `ROUND_ID` 再跑一轮。

```powershell
# 1. local_no_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 19 19 --fixed-threshold 1.0 --no-auto-train

# 2. cloud_no_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 0 0 --fixed-threshold 1.0 --no-auto-train

# 3. edge_no_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 0 19 --fixed-threshold 1.0 --no-auto-train

# 4. local_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 19 19 --fixed-threshold 0.7 --no-auto-train

# 5. split_de_no_ee
# python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 7 19 --fixed-threshold 1.0 --no-auto-train

# 6. split_dec_no_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 7 13 --fixed-threshold 1.0 --no-auto-train

# 7. ours_schedule
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --no-auto-train
```

可选扩展：

```powershell
# edge_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 0 19 --fixed-threshold 0.7 --no-auto-train

# cloud_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 0 0 --fixed-threshold 0.7 --no-auto-train

# split_dec_ee
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 2 --fixed-split 7 13 --fixed-threshold 0.7 --no-auto-train
```

## 7. 每轮结束后检查什么

每台 Device 会在本机生成：

```text
Data/Runtime/DeviceResults/<round_id>/user_<user_id>_measurements.jsonl
Data/Runtime/DeviceResults/<round_id>/user_<user_id>_summary.json
Data/Runtime/DeviceResults/<round_id>/user_<user_id>_inference_results.csv
```

每个 case 至少检查：

1. 两台 Device 都有 `summary.json` 和 `inference_results.csv`。
2. `summary.json` 里的 `decision.user.partition_boundary_1/2` 与本 case 设计一致。
3. `summary.json` 里的 `decision.decision_source` 与预期一致：
   - 固定组通常包含 `fixed_split` 和 `threshold`。
   - `ours_schedule` 通常是 `default`、`cached_dsci:*` 或 DSCI 相关来源。
4. `samples` 等于设定的 `--test-samples`。
5. `latency.T_total_avg_ms`、`accuracy` 非空。
6. `measurements.jsonl` 中 `exit_location` 分布符合预期：
   - `local_no_ee` 主要是 `device`。
   - `edge_no_ee` 主要是 `edge`。
   - `cloud_no_ee` 主要是 `cloud`。
   - early-exit 组可能在更早节点结束。

建议把两台 Device 的结果目录拷贝到统一目录，例如：

```text
Scripts/Results/Exp2_Ablation/RealDevice/<case>/<round_id>/device0/
Scripts/Results/Exp2_Ablation/RealDevice/<case>/<round_id>/device1/
```

## 8. 记录实验日志模板

每跑完一个 case，记录下面信息：

```text
case:
round_id:
scheduler_command:
device0_profile:
device1_profile:
edge_profile:
cloud_profile:
test_samples:
device0_accuracy:
device1_accuracy:
device0_T_total_avg_ms:
device1_T_total_avg_ms:
notes:
```

如果要报告双设备整体结果，建议使用：

```text
accuracy_mean = (device0_accuracy + device1_accuracy) / 2
latency_mean_ms = (device0_T_total_avg_ms + device1_T_total_avg_ms) / 2
latency_sum_ms = device0_T_total_avg_ms + device1_T_total_avg_ms
```

`latency_mean_ms` 反映单请求平均体验；`latency_sum_ms` 更接近两用户总成本。

## 9. 常见问题

### Device 一直等决策

检查 Scheduler 是否用了 `--expected-users 2`，两台 Device 是否同一个 `ROUND_ID`，以及两个 `--user-id` 是否分别是 `0` 和 `1`。

### 409 Conflict

通常是复用了 `ROUND_ID`，或者同一个 `round_id + user_id` 用不同状态注册过。换一个新的 `ROUND_ID` 后重跑该 case。

### iperf busy

两台 Device 同时测 Device-to-Edge 带宽时，Edge 的 iperf 服务可能忙。可以错开两台 Device 启动时间，或在 Device 上设置：

```bash
export DSCI_IPERF_RETRY_SLEEP_S=12
```

### 固定阈值只能设置一个值

当前 Scheduler CLI 的 `--fixed-threshold` 会给所有 early-exit 出口设置同一个阈值。如果要复现每个出口不同阈值，需要走 `ours_schedule` 的优化结果，或扩展 Scheduler CLI 支持 per-exit threshold。

### 想固定带宽，减少每轮波动

Device 侧可以用 `--override-bw-d2e`，Edge 侧可以用 `--override-bw-e2c`。正式实验如果使用 override，必须在日志里记录固定值。

```bash
python -m Src.Phase3_Runtime.Device.run_device \
  --bundle-id resnet50-cifar10-ee-v1 \
  --backend pytorch \
  --user-id 0 \
  --round-id "$ROUND_ID" \
  --test-samples 500 \
  --override-bw-d2e 100
```

```powershell
python -m Src.Phase3_Runtime.Edge.run_edge `
  --bundle-id resnet50-cifar10-ee-v1 `
  --backend pytorch `
  --override-bw-e2c 500
```
