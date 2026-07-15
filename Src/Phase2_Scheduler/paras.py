"""Validated bundle-scoped input for one Phase 2 scheduling problem."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from Src.Phase2_Scheduler.algo_config import DEFAULT as ALGO_CFG
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import ModelBundleSpec, require_bundle_id
from Src.Shared.Config.paths import BundleArtifactPaths, bundle_paths
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Profiles.segment_profile import load_segment_profile


@dataclass
class Paras:
    n: int
    m: int
    E: list[int]
    exit_ids: list[str]
    bundle: ModelBundleSpec
    partition_manifest: Any
    D: list[int]
    C: list[float]
    C_theoretical: list[float] | None = None
    C_u: np.ndarray | None = None
    C_e: np.ndarray | None = None
    C_c: np.ndarray | None = None
    F_u: np.ndarray | None = None
    H_u: np.ndarray | None = None
    B_u: np.ndarray | None = None
    f_e_max: float = 1.0
    f_c_max: float = 1.0
    b_e: float = TESTBED_CFG.default_bw_d2e
    b_c: float = TESTBED_CFG.default_bw_e2c
    G: float = 1.0
    delta: float = 8e-11
    alpha: float = ALGO_CFG.alpha
    beta: float = ALGO_CFG.beta
    resource_mode: str = "fixed_worker_pool"
    segment_latency_u: np.ndarray | None = None
    segment_latency_e: np.ndarray | None = None
    segment_latency_c: np.ndarray | None = None
    exit_head_latency_u: list[dict[str, float]] | None = None
    exit_head_latency_e: dict[str, float] | None = None
    exit_head_latency_c: dict[str, float] | None = None
    edge_worker_count: int = 1
    cloud_worker_count: int = 1
    shared_resource_model: bool = False
    d2e_link_ids: list[str] | None = None
    d2e_link_capacities_mbps: dict[str, float] | None = None
    e2c_link_id: str = "edge-cloud"
    e2c_link_capacity_mbps: float | None = None
    protocol_overhead_d2e_s: float = 0.0
    protocol_overhead_e2c_s: float = 0.0
    tensor_transport_dtype: str = "float32"
    transport_byte_scale: float = 1.0
    rates: np.ndarray | None = field(init=False, default=None)
    accs: np.ndarray | None = field(init=False, default=None)

    @property
    def bundle_id(self) -> str:
        return self.bundle.bundle_id

    @property
    def bundle_paths(self) -> BundleArtifactPaths:
        return bundle_paths(self.bundle_id)

    @property
    def manifest_id(self) -> str:
        return self.partition_manifest.manifest_id

    @property
    def partition_boundary_ids(self) -> list[int]:
        return list(self.partition_manifest.boundary_ids)

    @property
    def boundary_bytes(self) -> list[int]:
        return list(self.partition_manifest.boundary_bytes)

    def __post_init__(self):
        self.C_theoretical = list(self.C if self.C_theoretical is None else self.C_theoretical)
        self.C = list(self.C_theoretical)
        self.F_u = np.ones(self.n) if self.F_u is None else np.asarray(self.F_u, dtype=float)
        self.C_u = np.tile(self.C, (self.n, 1)) if self.C_u is None else np.asarray(self.C_u, dtype=float)
        self.C_e = np.asarray(self.C if self.C_e is None else self.C_e, dtype=float)
        self.C_c = np.asarray(self.C if self.C_c is None else self.C_c, dtype=float)
        if self.H_u is not None:
            self.H_u = np.asarray(self.H_u, dtype=float)
        self.B_u = (
            np.full(self.n, float(self.b_e), dtype=float)
            if self.B_u is None
            else np.asarray(self.B_u, dtype=float)
        )
        if self.d2e_link_ids is None:
            self.d2e_link_ids = [f"user-{index}" for index in range(self.n)]
        if len(self.d2e_link_ids) != self.n:
            raise ValueError("d2e_link_ids must contain one entry per user")
        if self.d2e_link_capacities_mbps is None:
            self.d2e_link_capacities_mbps = {
                str(self.d2e_link_ids[index]): float(self.B_u[index])
                for index in range(self.n)
            }
        if self.e2c_link_capacity_mbps is None:
            self.e2c_link_capacity_mbps = float(self.b_c)
        if self.C_u.shape != (self.n, self.m) or self.C_e.shape != (self.m,) or self.C_c.shape != (self.m,):
            raise ValueError("Bundle computation arrays do not match manifest boundary dimension")
        from Src.Phase2_Scheduler.Utils.parsing_data import parsing_rate_and_acc
        self.rates, self.accs = parsing_rate_and_acc(self)

    @staticmethod
    def _normalise_transport_dtype(value: Any) -> str:
        value = str(value or "float32").strip().lower()
        if value in {"", "none", "fp32", "float32"}:
            return "float32"
        if value in {"fp16", "float16"}:
            return "float16"
        raise ValueError(
            "tensor_transport_dtype must be one of: float32, fp32, none, "
            "float16, fp16"
        )

    @classmethod
    def _transport_dtype_from_state(cls, state: dict) -> str:
        if state.get("tensor_transport_dtype") is not None:
            return cls._normalise_transport_dtype(state.get("tensor_transport_dtype"))
        scheduler_override = os.environ.get("DSCI_SCHEDULER_TRANSPORT_DTYPE")
        if scheduler_override is not None:
            return cls._normalise_transport_dtype(scheduler_override)
        owner_values = [
            owner.get("tensor_transport_dtype")
            for owner in [
                *(state.get("users") or []),
                state.get("edge") or {},
                state.get("cloud") or {},
            ]
            if owner.get("tensor_transport_dtype") is not None
        ]
        if owner_values:
            normalised = {cls._normalise_transport_dtype(value) for value in owner_values}
            if len(normalised) != 1:
                raise ValueError(
                f"Inconsistent tensor_transport_dtype across nodes: {sorted(normalised)}"
                )
            return normalised.pop()
        return cls._normalise_transport_dtype(
            os.environ.get("DSCI_TENSOR_TRANSPORT_DTYPE") or "float32"
        )

    @classmethod
    def _transport_byte_scale(cls, dtype: str) -> float:
        dtype = cls._normalise_transport_dtype(dtype)
        return 0.5 if dtype == "float16" else 1.0

    @classmethod
    def from_state(cls, state: dict, algo_cfg=None):
        algo_cfg = algo_cfg or ALGO_CFG
        bundle = require_bundle_id(state)
        manifest = load_partition_manifest(bundle.bundle_id)
        owners = [*state["users"], state["edge"], state["cloud"]]
        for owner in owners:
            if owner.get("bundle_id") != bundle.bundle_id:
                raise ValueError("Every node must report the selected bundle_id")
            if owner.get("manifest_id") != manifest.manifest_id:
                raise ValueError("Every node must report the selected manifest_id")
            if owner.get("model_hash") != manifest.model_hash:
                raise ValueError("Every node model_hash must match the manifest")
        users, edge, cloud = state["users"], state["edge"], state["cloud"]
        resource_mode = str(state.get("resource_mode", "fixed_worker_pool"))
        if resource_mode != "fixed_worker_pool":
            raise ValueError("Only fixed_worker_pool segment profiles are supported")
        m = len(manifest.boundaries)
        exits = list(manifest.exit_boundary_ids)
        exit_ids = list(manifest.exit_ids)
        transport_dtype = cls._transport_dtype_from_state(state)
        transport_byte_scale = cls._transport_byte_scale(transport_dtype)
        curves = pd.read_csv(bundle_paths(bundle.bundle_id).offline_table_path)
        curves.columns = [str(column).strip() for column in curves.columns]
        if not {"threshold", "final_accuracy"}.issubset(curves.columns):
            raise ValueError("Legacy offline tables are not supported")

        if resource_mode == "fixed_worker_pool":
            def execution_profile(owner):
                profile = load_segment_profile(
                    str(owner["execution_profile_id"]),
                    manifest=manifest,
                    expected_backend=str(owner["backend"]),
                )
                if profile.worker_count != int(owner["worker_count"]):
                    raise ValueError("worker_count does not match execution profile")
                return profile
            user_profiles = [execution_profile(user) for user in users]
            edge_profile, cloud_profile = execution_profile(edge), execution_profile(cloud)
            d2e_overheads = [
                float(user.get("protocol_overhead_s", 0.0))
                for user in users
            ]
            d2e_link_ids = [
                str(user.get("d2e_link_id", f"user-{index}"))
                for index, user in enumerate(users)
            ]
            shared_d2e_capacity = edge.get("d2e_capacity_mbps")
            d2e_capacities: dict[str, float] = {}
            for index, user in enumerate(users):
                link_id = d2e_link_ids[index]
                capacity = user.get("d2e_capacity_mbps", shared_d2e_capacity)
                if capacity is None:
                    capacity = user["BW_d2e"]
                capacity = float(capacity)
                previous = d2e_capacities.get(link_id)
                if previous is not None and not np.isclose(previous, capacity):
                    raise ValueError(
                        f"inconsistent capacity for shared D2E link {link_id!r}"
                    )
                d2e_capacities[link_id] = capacity
            zero_work = np.zeros(m)
            return cls(
                n=len(users), m=m, E=exits, exit_ids=exit_ids, bundle=bundle,
                partition_manifest=manifest, D=list(manifest.boundary_bytes),
                C=list(zero_work), C_u=np.tile(zero_work, (len(users), 1)),
                C_e=zero_work, C_c=zero_work, F_u=np.ones(len(users)),
                B_u=np.array([float(user["BW_d2e"]) for user in users]),
                b_c=float(cloud["BW_e2c"]), alpha=algo_cfg.alpha, beta=algo_cfg.beta,
                resource_mode=resource_mode,
                segment_latency_u=np.stack([p.latencies for p in user_profiles]),
                segment_latency_e=edge_profile.latencies,
                segment_latency_c=cloud_profile.latencies,
                exit_head_latency_u=[p.exit_head_latencies for p in user_profiles],
                exit_head_latency_e=edge_profile.exit_head_latencies,
                exit_head_latency_c=cloud_profile.exit_head_latencies,
                edge_worker_count=edge_profile.worker_count,
                cloud_worker_count=cloud_profile.worker_count,
                shared_resource_model=bool(state.get("shared_resource_model", False)),
                d2e_link_ids=d2e_link_ids,
                d2e_link_capacities_mbps=d2e_capacities,
                e2c_link_id=str(cloud.get("e2c_link_id", "edge-cloud")),
                e2c_link_capacity_mbps=float(
                    cloud.get("e2c_capacity_mbps", cloud["BW_e2c"])
                ),
                protocol_overhead_d2e_s=max(d2e_overheads, default=0.0),
                protocol_overhead_e2c_s=float(edge.get("protocol_overhead_s", 0.0)),
                tensor_transport_dtype=transport_dtype,
                transport_byte_scale=transport_byte_scale,
            )
