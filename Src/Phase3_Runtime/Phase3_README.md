# Phase 3 README

Phase 3 是在线运行时，包含 Cloud、Edge、Device 三端。Device 从 Scheduler 获取决策后，先执行本地 segment，再把中间张量通过 TCP socket 传给 Edge 或 Cloud；Edge/Cloud 根据决策继续执行后续 segment 或返回结果。

## 运行前检查

1. 已按 [Phase1_README](../Phase1_Offline/Phase1_README.md) 准备好模型包和 segment profile。
2. `Src/Shared/Config/deploy_config.py` 中的 IP 和端口符合当前部署机器。
3. Scheduler 机器已拥有所有 Device/Edge/Cloud profile。
4. Cloud、Edge、Scheduler、Device 之间的端口可访问。
5. `iperf3` 已安装，用于 Device 测量 Device-to-Edge 与 Edge-to-Cloud 带宽。

## 启动顺序

推荐顺序是 Cloud、Edge、Scheduler、Device。

### Cloud

Cloud 侧需要先启动 iperf 服务，再启动推理服务：

```bash
iperf3 -s -p 32264
```

```bash
export DSCI_CLOUD_PYTORCH_SEGMENT_PROFILE_ID=cloud-v100-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Cloud.run_cloud \
	--bundle-id resnet50-cifar10-ee-v1 \
	--backend pytorch
```

Cloud 推理 socket 默认端口为 `32266`，状态 HTTP 端口为 `32265`。

### Edge

Edge 侧先启动 iperf，再启动 Edge runtime：

```powershell
iperf3 -s -p 5001
```

```powershell
$env:DSCI_EDGE_PYTORCH_SEGMENT_PROFILE_ID="edge-jialindesktop-pytorch-resnet50-cifar10"
python -m Src.Phase3_Runtime.Edge.run_edge `
	--bundle-id resnet50-cifar10-ee-v1 `
	--backend pytorch
```

Edge 推理 socket 默认端口为 `9001`，状态 HTTP 端口为 `9002`。

### Scheduler

Scheduler 属于 Phase 2，但运行时启动顺序中应放在 Edge 之后、Device 之前：

```powershell
python -m Src.Phase2_Scheduler.Service.api_server --expected-users 1
```

更多调度配置见 [Phase2_README](../Phase2_Scheduler/Phase2_README.md)。

### Device

Device 侧指定 profile、bundle、后端、用户 ID 和轮次 ID：

```bash
export DSCI_DEVICE_PYTORCH_SEGMENT_PROFILE_ID=device-nx1-pytorch-resnet50-cifar10
python -m Src.Phase3_Runtime.Device.run_device \
	--bundle-id resnet50-cifar10-ee-v1 \
	--backend pytorch \
	--user-id 0 \
	--test-samples 1 \
	--round-id 2
```

多 Device 运行时，每台 Device 使用唯一 `--user-id`，同一批次使用相同 `--round-id`，Scheduler 的 `--expected-users` 要与 Device 数量一致。

## 子目录

| 目录 | 说明 |
| --- | --- |
| `Cloud/` | Cloud 推理服务和状态服务入口 |
| `Edge/` | Edge 推理服务、状态服务和转发逻辑 |
| `Device/` | Device 端注册、决策轮询、推理循环和测量上报 |
| `Shared/` | socket server、tensor codec、worker pool、segment executor、状态上报等公共运行时代码 |

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| Device 一直等决策 | `--expected-users` 是否与 Device 数量一致，所有 Device 是否使用同一 `round_id` |
| Scheduler 找不到 profile | 新生成的 profile 是否已复制到 Scheduler 的 `Data/Profiles/` |
| Edge/Cloud 状态查询失败 | `deploy_config.py` 中 IP 是否正确，HTTP 状态端口是否开放 |
| 张量转发失败 | TCP socket 端口是否开放，Cloud/Edge runtime 是否已启动 |
| 带宽测量失败 | `iperf3` 服务端是否已启动，端口是否对应 |

## 相关文档

- [Src_README](../Src_README.md)：系统代码复现入口和快速开始。
- [DATA_README](../../Data/DATA_README.md)：数据、模型包和 profile 结构。
- [Phase2_README](../Phase2_Scheduler/Phase2_README.md)：Scheduler 和 API 服务。
## 论文实验请求同步与 Trace

Device 默认在每个样本执行前调用请求级屏障：

```text
POST /api/v2/rounds/{round_id}/requests/{request_seq}/ready/{user_id}
```

所有设备就绪后获得同一个 `release_at`，屏障等待不计入 `T_total`；请求记录
保存 `barrier_ready_at_utc`、`barrier_release_at_utc`、`actual_start_at_utc`
和 `start_skew_s`。
仅兼容旧流程时可使用 Device 参数 `--no-request-barrier`。

推理响应中的 `request_trace` 会累积 Device/Edge/Cloud 的执行 segment、
最终出口、置信度、worker queue、segment/exit-head/exit-check 计算、两段传输、
端到端时延和 `unattributed_overhead`。各互斥分项加 residual 必须等于
`total_latency`。
