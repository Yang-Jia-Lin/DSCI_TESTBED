# Exp3-D System Overhead Summary

Scope: resnet50-cifar10-ee-v1; cached PPO metrics, SolutionCache, segment profiles, and runtime logs are reused.

## Phase 1: profiling overhead
- profile_segments / resnet50-cifar10-ee-v1: mean 7.936 s (min 7.936 s, max 7.936 s, runs 1).

## Phase 2: optimization overhead
- cold: mean 376.033 s (min 64.130 s, max 573.262 s, runs 4, mean objective 0.8079).
- cold_warm: mean 599.883 s (min 579.903 s, max 619.862 s, runs 2, mean objective 1.0801).
- medium: mean 98.124 s (min 98.124 s, max 98.124 s, runs 1, mean objective 0.6642).
- near: mean 19.666 s (min 15.156 s, max 23.918 s, runs 3, mean objective 0.7689).

## Phase 3: expected inference latency breakdown
- Device compute: 65.309 ms, share 100.00%.
- D2E transmission: 0.000 ms, share 0.00%.
- Edge compute: 0.000 ms, share 0.00%.
- E2C transmission: 0.000 ms, share 0.00%.
- Cloud compute: 0.000 ms, share 0.00%.
- Component shares sum to 1.000000.

## Phase 3: expected early-exit distribution
- after_layer2: 34.20%.
- after_layer3: 48.41%.
- final: 17.39%.
- early_exit_total: 82.61%.
- Early-exit total equals 82.61%; exit-point rates sum to 1.000000.

## Runtime-log sanity check
- Observed DeviceResults cover 1004 samples across 7 rounds.
- Observed exit locations: device.
- Existing logs all terminate at device, so they validate the realized device-side fast path but do not contain per-head exit_id.

## Artifacts
- PPO vs baselines: D:\Coding\Python\DSCI_testbed\Scripts\Results\Exp3_Convergency_and_Overhead\ppo_vs_baselines.csv
- Phase 3 latency breakdown: D:\Coding\Python\DSCI_testbed\Scripts\Results\Exp3_Convergency_and_Overhead\phase3_latency_breakdown.csv
- Phase 3 exit distribution: D:\Coding\Python\DSCI_testbed\Scripts\Results\Exp3_Convergency_and_Overhead\phase3_exit_distribution.csv
