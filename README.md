```bash
iperf3 -s -p 32264                              # Cloud: Edge->Cloud bandwidth
python -m Src.Deploy.Cloud.run_cloud            # Cloud: status + feature service
iperf3 -s -p 5001                               # Edge: Device->Edge bandwidth
python -m Src.Deploy.Edge.run_edge              # Edge: status + feature service
python -m Src.Algorithm.Interface.api_server    # Algorithm API
python -m Src.Deploy.Device.run_device          # Device
```

# DSCI Testbed

DSCI Testbed is a three-node collaborative inference prototype for Device, Edge, and Cloud execution.
Each node loads the same full `MultiEEResNet50` model and executes only the stages assigned by the
algorithm decision JSON.

## Runtime Layout

```text
DSCI_testbed/
|-- Data/
|   |-- CIFAR10/
|   |-- Weights/
|   |   `-- full_model.pth
|   `-- OfflineTables/
|       |-- Resnet50_rates.csv
|       |-- Resnet50_accs.csv
|       `-- Resnet50_layer_stats.csv
|-- Scripts/
|   |-- Exp0_Motivation/
|   |-- Exp1_Testbed/
|   |-- Exp2_Baseline/
|   |-- Exp3_Dynamic/
|   |-- Exp4_DSCI_Convergency/
|   |-- Exp5_Ablation/
|   |-- Exp6_EE_Model/
|   `-- Results/
`-- Src/
    |-- Algorithm/
    |   |-- Interface/
    |   |   |-- api_server.py
    |   |   |-- algo_service.py
    |   |   |-- decision_codec.py
    |   |   |-- state_adapter.py
    |   |   `-- SolutionCache/
    |   `-- Optimizer/
    |-- Configs/
    |-- Deploy/
    |   |-- Cloud/
    |   |-- Device/
    |   |   `-- Results/
    |   |-- Edge/
    |   |-- monitor/
    |   `-- shared/
    `-- Models/
```

Generated experiment outputs go under `Scripts/Results/`. Device-side test CSV files go under
`Src/Deploy/Device/Results/` with a timestamped name such as `test_results_YYYYMMDDHHMMSS.csv`.

---

## Key Paths

| Purpose | Path |
| --- | --- |
| Full model weights | `Data/Weights/full_model.pth` |
| Offline CSV lookup tables | `Data/OfflineTables/` |
| Script experiment outputs | `Scripts/Results/` |
| API cached latest DSCI decision | `Src/Algorithm/Interface/SolutionCache/latest_solution.npz` |
| API cached latest metadata | `Src/Algorithm/Interface/SolutionCache/latest_solution_meta.json` |
| API cached timestamp history | `Src/Algorithm/Interface/SolutionCache/solution_YYYYMMDDHHMMSSmmm.*` |
| Device inference CSV outputs | `Src/Deploy/Device/Results/` |

`SolutionCache` keeps `latest_solution.*` for fast startup and the latest 3 timestamped decision snapshots.

---

## Why iperf3 Is Required

iperf3 is used to **measure real-time bandwidth between nodes**. The measured values (`BW_d2e` and
`BW_e2c`) are passed directly to the Algorithm API, which uses them to compute the optimal partition
points. If iperf3 is not running, `measure_bandwidth_iperf` returns `0.0`, causing the algorithm to
make partition decisions based on incorrect bandwidth and producing unreasonable inference splits.

**iperf3 must be running before any inference begins.**

---

## Deployment Topology

| Role | Typical Device | Description |
| --- | --- | --- |
| **Device** | Raspberry Pi | Initiates inference, measures bandwidth, requests partition decision |
| **Edge** | Linux server | Receives features from Device, runs middle inference, forwards to Cloud |
| **Cloud** | PC / server | Runs final inference and returns the classification result |

---

## Per-Node Setup and Startup

### 1. Cloud Node

Start two processes (order within node does not matter):

```bash
# iperf3 server - Edge measures Edge->Cloud bandwidth against this
iperf3 -s -p 32264

# Cloud inference service
python -m Src.Deploy.Cloud.run_cloud
```

Ports used by Cloud:

| Port | Purpose |
| --- | --- |
| `32264` | iperf3 server (Edge connects to measure BW_e2c) |
| `32265` | Cloud status HTTP API (`/status`) |
| `32266` | Feature tensor input from Edge |

---

### 2. Edge Node

Start two processes:

```bash
# iperf3 server - Device measures Device->Edge bandwidth against this
iperf3 -s -p 5001

# Edge inference service
python -m Src.Deploy.Edge.run_edge
```

Ports used by Edge:

| Port | Purpose |
| --- | --- |
| `5001` | iperf3 server (Device connects to measure BW_d2e) |
| `9001` | Feature tensor input from Device |
| `9002` | Edge status HTTP API (`/status`, returns `f_e_max` and `BW_e2c`) |

> **Configuration required** - In `Src/Deploy/Edge/run_edge.py`, make the cloud-facing settings match the Cloud node:
> - `CLOUD_IP` must be the real Cloud IP.
> - `CLOUD_IPERF_PORT = 32264`.
> - `CLOUD_FEATURE_PORT = 32266`.
> Edge's own `5001`, `9001`, and `9002` ports stay unchanged.

---

### 3. Algorithm API (any reachable machine)

The API can run on the Edge node, Cloud node, or any machine reachable from Device:

```bash
python -m Src.Algorithm.Interface.api_server
```

Port used:

| Port | Purpose |
| --- | --- |
| `8000` | Decision and health HTTP API |

---

### 4. Device Node (Raspberry Pi)

**Before starting**, edit `Src/Deploy/Device/run_device.py` and set the real IP addresses:

```python
EDGE_IP  = "<Edge real IP>"
CLOUD_IP = "<Cloud real IP>"
CLOUD_STATUS_PORT = 32265
IPERF_PORT_CLOUD = 32264
ALGO_URL = "http://<API server IP>:8000/api/v1/decision"
```

Also edit `Src/Deploy/monitor/bandwidth.py` and change `IPERF_EXE` for Linux/Raspberry Pi:

```python
# Replace the Windows path:
# IPERF_EXE = "S:\\Tools\\Iperf\\iperf3.exe"
# With:
IPERF_EXE = "iperf3"   # use the system-installed iperf3 on Linux
```

Then start:

```bash
python -m Src.Deploy.Device.run_device
```

---

## Recommended Startup Order

```
1.  Cloud:   iperf3 -s -p 32264
2.  Cloud:   python -m Src.Deploy.Cloud.run_cloud
3.  Edge:    iperf3 -s -p 5001
4.  Edge:    python -m Src.Deploy.Edge.run_edge
5.  Any:     python -m Src.Algorithm.Interface.api_server
6.  Device:  python -m Src.Deploy.Device.run_device
```

Cloud and Edge services must be fully up before Device starts so that iperf3 measurements and status
queries succeed on the first inference request.

---

## Port Reference

| Port | Owner | Purpose |
| --- | --- | --- |
| `5001` | Edge | iperf3 server 闂?Device闂備焦鍓氶崑鍛櫠閻㈩泹e bandwidth measurement |
| `32264` | Cloud | iperf3 server 闂?Edge闂備焦鍓氶崑鍛櫠閻滅尦ud bandwidth measurement |
| `8000` | Algorithm API | Decision and health HTTP API |
| `9001` | Edge | Feature tensor input from Device |
| `9002` | Edge | Edge status HTTP API (`f_e_max`, `BW_e2c`) |
| `32265` | Cloud | Cloud status HTTP API |
| `32266` | Cloud | Feature tensor input from Edge |

---

## Data Flow

```text
Device                         Edge                          Cloud
  |                              |                              |
  |-- iperf3 ------------------->| :5001                        |
  |   measure BW_d2e             |                              |
  |                              |-- iperf3 ------------------->| :32264
  |                              |   measure BW_e2c             |
  |-- HTTP GET ----------------->| :9002                        |
  |   fetch edge status          |                              |
  |-- HTTP GET ------------------------------------------------>| :32265
  |   fetch cloud status         |                              |
  |-- POST (:8000) ------------->| Algorithm API                |
  |   get partition decision     |                              |
  |-- local inference stage 0..s1|                              |
  |-- TCP ---------------------->| :9001                        |
  |   send feature tensor        |-- inference stage s1+1..s2   |
  |                              |-- TCP ---------------------->| :32266
  |                              |   forward feature tensor      |-- inference s2+1..end
  |<-----------------------------|<-----------------------------|
  |   receive final result       |                              |
```

---

## Model Execution

Every node loads:

```text
MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True)
```

Weights are loaded as a `state_dict` from `Data/Weights/full_model.pth`. Runtime execution uses
`forward_partial(x, start, end)`.

| Stage | Meaning |
| --- | --- |
| `0` | stem: `conv1 闂?bn1 闂?relu 闂?maxpool` |
| `1` | `layer1` |
| `2` | `layer2`, early exit at layer `57` |
| `3` | `layer3`, early exit at layer `103` |
| `4` | `layer4 闂?avgpool 闂?flatten 闂?fc` |

---

## Deployment Notes

- The Device node only needs model inference, the algorithm API URL, Edge socket connectivity, and
  iperf/status measurement. The Algorithm API can run on a PC, Edge node, or any reachable machine
  with the DSCI training environment.
- For Raspberry Pi deployment, keep the module commands consistent and adjust:
  - IP addresses and ports in deploy scripts.
  - `IPERF_EXE` in `Src/Deploy/monitor/bandwidth.py` to `"iperf3"`.
  - PyTorch/ONNX/MNN runtime availability.
  - `Data/Weights/full_model.pth`.
  - `Data/OfflineTables/` CSV files.

---

## Validation

```bash
python -m compileall Src Scripts
curl http://127.0.0.1:8000/api/v1/health
```
