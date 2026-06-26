"""Device-side helpers for executing partition-manifest decisions."""

from __future__ import annotations

import time

import torch

from Src.Phase3_Runtime.Device.comm import send_tensor
from Src.Phase3_Runtime.Shared.model_loader import load_full_model
from Src.Phase3_Runtime.Shared.request_identity import request_identity
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor


def _select_user_decision(decision: dict, user_id: int) -> dict:
    if "user" in decision:
        user = decision["user"]
        if int(user.get("user_id", -1)) != int(user_id):
            raise ValueError("Decision user_id does not match this Device")
        return user
    matches = [
        user
        for user in decision.get("users", [])
        if int(user.get("user_id", -1)) == int(user_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Decision must contain exactly one entry for user_id {user_id}")
    return matches[0]


def run_partitioned_inference(
    input_tensor: torch.Tensor,
    decision: dict,
    *,
    user_id: int,
    request_id: str,
) -> dict:
    total_started = time.perf_counter()
    if decision.get("resource_mode") != "fixed_worker_pool":
        raise ValueError("runtime_v2 only accepts fixed_worker_pool decisions")
    if "bundle_id" not in decision:
        raise ValueError("Legacy decision without bundle_id is not supported")
    identity = request_identity(
        {
            "round_id": decision.get("round_id"),
            "user_id": int(user_id),
            "request_id": str(request_id),
            "decision_id": decision.get("decision_id"),
            "decision_version": decision.get("decision_version"),
        }
    )
    manifest = load_partition_manifest(str(decision["bundle_id"]))
    if decision.get("model_hash") != manifest.model_hash:
        raise ValueError("Decision model_hash does not match partition manifest")
    user = _select_user_decision(decision, user_id)
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
            **identity,
            "exit_location": "device",
            "T_compute_device": t_device,
            "T_total": time.perf_counter() - total_started,
        }
    device_output = device_result["tensors"]
    payload = {
        **identity,
        "bundle_id": manifest.bundle_id,
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
    if request_identity(response) != identity:
        raise ValueError("Response identity does not match request identity")
    return response
