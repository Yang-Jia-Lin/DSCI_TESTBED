"""Cross-node request trace construction and mutually exclusive timings."""

from __future__ import annotations

import copy


def node_trace(node: str, result: dict) -> dict:
    breakdown = result.get("compute_breakdown") or {}
    return {
        "node": str(node),
        "executed_segments": [int(value) for value in result.get("executed_segments", [])],
        "queue_s": float(result.get("T_queue_s", 0.0)),
        "segment_compute_s": float(breakdown.get("segment_compute_s", 0.0)),
        "exit_head_compute_s": float(breakdown.get("exit_head_compute_s", 0.0)),
        "exit_check_s": float(breakdown.get("exit_check_s", 0.0)),
        "worker_elapsed_s": float(result.get("T_worker_elapsed_s", result.get("T_compute_s", 0.0))),
    }


def append_node_trace(existing: list[dict] | None, node: str, result: dict) -> list[dict]:
    trace = copy.deepcopy(existing or [])
    trace.append(node_trace(node, result))
    return trace


def finalize_request_trace(result: dict) -> dict:
    nodes = copy.deepcopy(result.get("node_trace") or [])
    by_node = {str(item["node"]): item for item in nodes}
    edge = by_node.get("edge", {})
    cloud = by_node.get("cloud", {})

    edge_cloud_roundtrip = float(result.get("T_edge_cloud_roundtrip", 0.0))
    edge_node = float(result.get("T_node_edge", 0.0))
    cloud_node = float(result.get("T_node_cloud", 0.0))
    device_edge_roundtrip = float(result.get("T_device_edge_roundtrip", 0.0))
    d2e_transport = max(
        0.0, device_edge_roundtrip - edge_node - edge_cloud_roundtrip
    )
    e2c_transport = max(0.0, edge_cloud_roundtrip - cloud_node)

    fields = {
        "device_compute": float(result.get("T_compute_device", 0.0)),
        "d2e_transport": d2e_transport,
        "edge_queue": float(edge.get("queue_s", 0.0)),
        "edge_segment_compute": float(edge.get("segment_compute_s", 0.0)),
        "edge_exit_head_compute": float(edge.get("exit_head_compute_s", 0.0)),
        "edge_exit_check": float(edge.get("exit_check_s", 0.0)),
        "e2c_transport": e2c_transport,
        "cloud_queue": float(cloud.get("queue_s", 0.0)),
        "cloud_segment_compute": float(cloud.get("segment_compute_s", 0.0)),
        "cloud_exit_head_compute": float(cloud.get("exit_head_compute_s", 0.0)),
        "cloud_exit_check": float(cloud.get("exit_check_s", 0.0)),
        "total_latency": float(result.get("T_total", 0.0)),
    }
    known = sum(value for key, value in fields.items() if key != "total_latency")
    fields["unattributed_overhead"] = fields["total_latency"] - known
    fields.update(
        {
            "request_id": result.get("request_id"),
            "user_id": result.get("user_id"),
            "prediction": result.get("prediction"),
            "exit_id": result.get("exit_id"),
            "exit_boundary_id": result.get("exit_boundary_id"),
            "exit_location": result.get("exit_location"),
            "confidence": result.get("confidence"),
            "executed_segments_by_node": {
                str(item["node"]): list(item.get("executed_segments", []))
                for item in nodes
            },
            "nodes": nodes,
        }
    )
    return fields
