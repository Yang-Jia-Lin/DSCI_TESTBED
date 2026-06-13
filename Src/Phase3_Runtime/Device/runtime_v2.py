"""Device-side helpers for executing partition-manifest decisions."""

from __future__ import annotations

import time

import torch

from Src.Phase3_Runtime.Device.comm import send_tensor
from Src.Phase3_Runtime.Shared.model_loader import load_full_model
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor


def run_partitioned_inference(input_tensor: torch.Tensor, decision: dict) -> dict:
    total_started = time.perf_counter()
    if decision.get("resource_mode") != "fixed_worker_pool":
        raise ValueError("runtime_v2 only accepts fixed_worker_pool decisions")
    manifest = load_partition_manifest(str(decision["manifest_id"]))
    if decision.get("model_hash") != manifest.model_hash:
        raise ValueError("Decision model_hash does not match partition manifest")
    user = decision["users"][0]
    b1 = int(user["partition_boundary_1"])
    b2 = int(user["partition_boundary_2"])
    manifest.validate_boundary_pair(b1, b2)
    executor = PyTorchSegmentExecutor(load_full_model(manifest), manifest)
    started = time.perf_counter()
    device_result = executor.execute_range_with_exits(
        0, b1, {"main": input_tensor}, user.get("exit_thresholds", {})
    )
    t_device = time.perf_counter() - started
    if device_result["prediction"] is not None:
        return {
            **device_result,
            "exit_location": "device",
            "T_compute_device": t_device,
            "T_total": time.perf_counter() - total_started,
        }
    device_output = device_result["tensors"]
    payload = {
        "manifest_id": manifest.manifest_id,
        "model_hash": manifest.model_hash,
        "boundary_id": b1,
        "tensors": device_output,
        "meta": {
            "partition_boundary_2": b2,
            "exit_thresholds": user.get("exit_thresholds", {}),
        },
    }
    response = send_tensor(
        payload, TESTBED_CFG.edge_host, TESTBED_CFG.edge_feature_port
    )
    response["T_compute_device"] = t_device
    response["T_total"] = time.perf_counter() - total_started
    return response
