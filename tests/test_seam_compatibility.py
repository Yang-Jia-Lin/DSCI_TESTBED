from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Scripts.EvaluationCommon import artifacts
from Scripts.EvaluationCommon.config import EXPECTED_BUNDLES, EXP1_RESULT_DIR
from Src.Phase2_Scheduler.Optimizer.DSCI.networks import ActorCritic
from Src.Phase2_Scheduler.Service.decision_codec import encode
from Src.Phase2_Scheduler.Utils.parsing_data import parsing_rate_and_acc
from Src.Shared.Config.model_config import DEFAULT_BUNDLE_ID
from Src.Shared.Config.paths import RESULT_TESTBED_PATH, bundle_paths
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.split_actions import encode_split_row


SEAM_BUNDLES = (
    "resnet50-cifar10",
    "resnet50-neucls64",
    "resnet50-imagenet100",
    "vit-base-cifar10",
    "vit-base-neucls64",
    "vit-base-imagenet100",
)


@pytest.mark.parametrize("bundle_id", SEAM_BUNDLES)
def test_seam_bundle_has_three_manifest_exits_and_curve_columns(bundle_id):
    manifest = load_partition_manifest(bundle_id)
    if bundle_id.startswith("resnet50-"):
        expected_ids = ("after_layer1", "after_layer2", "after_layer3")
        expected_boundaries = (4, 8, 14)
    else:
        expected_ids = ("after_block3", "after_block6", "after_block9")
        expected_boundaries = (4, 7, 10)

    assert manifest.exit_ids == expected_ids
    assert manifest.exit_boundary_ids == expected_boundaries

    with bundle_paths(bundle_id).offline_table_path.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        columns = {column.strip() for column in next(csv.reader(handle))}
    for exit_id in expected_ids:
        assert f"{exit_id}_rate" in columns
        assert f"{exit_id}_accuracy" in columns

    paras = SimpleNamespace(
        bundle_paths=bundle_paths(bundle_id),
        exit_ids=list(manifest.exit_ids),
        E=list(manifest.exit_boundary_ids),
        m=len(manifest.boundaries),
    )
    rates, accuracies = parsing_rate_and_acc(paras)
    assert rates.shape == accuracies.shape
    assert rates.shape[1] == len(manifest.boundaries)


def test_seam_defaults_and_result_paths_use_current_experiment():
    assert DEFAULT_BUNDLE_ID == "resnet50-cifar10"
    assert EXPECTED_BUNDLES == SEAM_BUNDLES
    assert EXP1_RESULT_DIR.name == "Exp1_SEAM"
    assert RESULT_TESTBED_PATH.name == "Exp1_SEAM"


@pytest.mark.parametrize("bundle_id", SEAM_BUNDLES)
def test_actor_and_decision_codec_emit_all_three_thresholds(bundle_id):
    manifest = load_partition_manifest(bundle_id)
    exits = list(manifest.exit_boundary_ids)
    exit_ids = list(manifest.exit_ids)
    m = len(manifest.boundaries)
    network = ActorCritic(
        state_dim=8,
        num_layers=m,
        action_dim_Y=len(exits),
        partition_boundary_ids=list(manifest.boundary_ids),
    )
    _, alpha, beta, _ = network(torch.zeros((1, 8)))
    assert alpha.shape == beta.shape == (1, 3)

    paras = SimpleNamespace(
        n=1,
        m=m,
        E=exits,
        exit_ids=exit_ids,
        resource_mode="fixed_worker_pool",
        partition_manifest=manifest,
        bundle_id=bundle_id,
        manifest_id=manifest.manifest_id,
        f_e_max=1.0,
        f_c_max=1.0,
    )
    X = encode_split_row(
        0, manifest.final_boundary_id, m, dtype=np.float32
    )[None, :]
    Y = np.ones((1, m), dtype=np.float32)
    for index, boundary in enumerate(exits):
        Y[0, boundary] = 0.2 + 0.1 * index

    decision = encode(
        X,
        Y,
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        paras,
    )
    thresholds = decision["users"][0]["exit_thresholds"]
    assert tuple(thresholds) == tuple(exit_ids)
    assert len(thresholds) == 3


def _write_profile(root, profile_id: str, bundle_id: str) -> None:
    directory = root / profile_id
    directory.mkdir()
    (directory / "metadata.json").write_text(
        json.dumps({"profile_id": profile_id, "bundle_id": bundle_id}),
        encoding="utf-8",
    )


def test_readiness_requires_nx_nano_pi5_edge_and_cloud(monkeypatch, tmp_path):
    bundle_id = "resnet50-cifar10"
    monkeypatch.setattr(artifacts, "SEGMENT_PROFILE_DIR", tmp_path)
    for profile_id in (
        "device-nx-pytorch-resnet50-cifar10",
        "device-nano-pytorch-resnet50-cifar10",
        "edge-windows-pytorch-resnet50-cifar10",
        "cloud-v100-pytorch-resnet50-cifar10",
    ):
        _write_profile(tmp_path, profile_id, bundle_id)

    pending = artifacts.check_bundle_readiness(bundle_id)
    assert pending.status == "pending"
    assert pending.device_pi5_profiles == 0
    assert "missing Pi5 device profile" in pending.notes

    _write_profile(
        tmp_path, "device-pi5-pytorch-resnet50-cifar10", bundle_id
    )
    ready = artifacts.check_bundle_readiness(bundle_id)
    assert ready.status == "ready"
    assert ready.device_profiles == 3
    assert ready.device_nx_profiles == 1
    assert ready.device_nano_profiles == 1
    assert ready.device_pi5_profiles == 1
    assert ready.edge_profiles == 1
    assert ready.cloud_profiles == 1
    assert ready.notes == "ok"
