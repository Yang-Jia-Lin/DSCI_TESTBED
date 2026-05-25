# DSCI Testbed

DSCI Testbed is a three-node collaborative inference prototype for Device, Edge, and Cloud execution. Each node loads the same full `MultiEEResNet50` model and executes only the stages assigned by the algorithm decision JSON.

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

Generated experiment outputs go under `Scripts/Results/`. Device-side test CSV files go under `Src/Deploy/Device/Results/` with a timestamped name such as `test_results_YYYYMMDDHHMMSS.csv`.

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

## Start Order

Start these services from the repository root. In local testing all IPs may be `127.0.0.1`; on real devices update `EDGE_IP`, `CLOUD_IP`, and `ALGO_URL` in `Src/Deploy/Device/run_device.py`.

```powershell
iperf3 -s -p 5001
iperf3 -s -p 5002
python -m Src.Deploy.Cloud.run_cloud
python -m Src.Deploy.Edge.run_edge
python -m Src.Algorithm.Interface.api_server
python -m Src.Deploy.Device.run_device
```

Required ports:

| Port | Owner | Purpose |
| --- | --- | --- |
| `5001` | Edge side | Device to Edge iperf bandwidth |
| `5002` | Cloud side | Edge to Cloud iperf bandwidth |
| `8000` | Algorithm API | Decision and health HTTP API |
| `9001` | Edge | Feature input from Device |
| `9002` | Edge | Edge status HTTP API |
| `9003` | Cloud | Cloud status HTTP API |
| `9004` | Cloud | Feature input from Edge |

## Algorithm API

Start:

```powershell
python -m Src.Algorithm.Interface.api_server
```

Main endpoints:

```text
POST /api/v1/decision
POST /api/v1/measurements
GET  /api/v1/health
```

`/api/v1/decision` immediately returns the current usable decision. If no compatible cached DSCI result exists, it returns a default feasible decision and starts full DSCI training in the background. When training finishes, the cached decision in `Src/Algorithm/Interface/SolutionCache/` is atomically updated.

The deploy contract includes:

```text
users[].partition_s1
users[].partition_s2
users[].exit_thresholds["57"]
users[].exit_thresholds["103"]
users[].edge_compute_quota
users[].cloud_compute_quota
```

Optional `decision_mode` can request baselines such as `device_no_exit`, `device_early_exit`, `edge_no_exit`, `edge_early_exit`, `cloud_no_exit`, and `cloud_early_exit`.

## Model Execution

Every node loads:

```text
MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True)
```

Weights are loaded as a `state_dict` from `Data/Weights/full_model.pth`. Runtime execution uses `forward_partial(x, start, end)`.

| Stage | Meaning |
| --- | --- |
| `0` | stem: `conv1 -> bn1 -> relu -> maxpool` |
| `1` | `layer1` |
| `2` | `layer2`, early exit at layer `57` |
| `3` | `layer3`, early exit at layer `103` |
| `4` | `layer4 -> avgpool -> flatten -> fc` |

## Deployment Notes

The Device node only needs model inference, the algorithm API URL, Edge socket connectivity, and iperf/status measurement. The Algorithm API can run on a PC, Edge node, or any reachable machine with the DSCI training environment.

For Raspberry Pi deployment, keep the module commands consistent and adjust:

- IP addresses and ports in deploy scripts.
- PyTorch/ONNX/MNN runtime availability.
- `Data/Weights/full_model.pth`.
- `Data/OfflineTables/` CSV files.

## Validation

```powershell
python -m compileall Src Scripts
curl http://127.0.0.1:8000/api/v1/health
```
