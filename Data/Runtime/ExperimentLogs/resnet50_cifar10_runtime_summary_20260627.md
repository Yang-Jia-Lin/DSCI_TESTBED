# resnet50-cifar10-ee-v1 Runtime Results

Date: 2026-06-27

## Confirmed Preset Baselines

Only the device preset was available in the chat history. In this runtime contract,
`preset:device:no_exit` is an almost-device-only configuration: the model runs to
`b1=18` on Device and sends a tiny tail tensor to Edge for `18 -> 19`.

| Round | Mode | Samples | BW_d2e | Split | Accuracy | T_total_avg_ms | T_compute_device_avg_ms | T_compute_edge_avg_ms | T_node_edge_avg_ms |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| preset-device-001 | preset:device:no_exit | 10 | 1.0044 Mbps | b1=18, b2=19 | 0.7000 | 1948.914 | 214.867 | 0.248 | 1.849 |
| preset-device-002 | preset:device:no_exit | 10 | 0.8730 Mbps | b1=18, b2=19 | 0.7000 | 1745.153 | 207.897 | 0.324 | 2.517 |

## Missing Preset Baselines

The following complete outputs were not present in the chat history:

| Mode | Status |
|---|---|
| preset:edge:no_exit | missing |
| preset:cloud:no_exit | missing |

## Related DSCI Results After Probability Fix

These are included for comparison with the available baseline.

| Round | Source | Samples | BW_d2e | Split | Thresholds | Accuracy | T_total_avg_ms | T_compute_device_avg_ms | T_device_edge_roundtrip_avg_ms |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| dsci-fixedp-001 | default | 10 | 1.5638 Mbps | b1=6, b2=13 | layer2=1.0, layer3=1.0 | 0.7000 | 14709.897 | 86.789 | 14547.028 |
| dsci-fixedp-002 | cached_dsci:warm | 10 | 1.2711 Mbps | b1=15, b2=16 | layer2=0.6714, layer3=0.1276 | 0.9000 | 225.144 | 150.488 | not triggered / not printed |
| dsci-fixedp-100s-001 | cached_dsci:warm | 100 | 1.1739 Mbps | b1=15, b2=16 | layer2=0.6714, layer3=0.1276 | 0.8000 | 154.240 | 145.457 | not triggered / not printed |

