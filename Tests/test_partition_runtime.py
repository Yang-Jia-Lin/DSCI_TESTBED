from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from Src.Phase2_Scheduler.Objective.compute_latency import compute_5_latency
from Src.Phase2_Scheduler.Optimizer.BF.alg_BF import optimize_BF
from Src.Phase2_Scheduler.Optimizer.DSCI.agent import _init_feasible_XY
from Src.Phase2_Scheduler.Optimizer.GA.alg_GA import optimize_GA
from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig
from Src.Phase2_Scheduler.Service.decision_codec import encode
from Src.Phase2_Scheduler.paras import Paras
from Src.Phase3_Runtime.Shared.fixed_worker_pool import FixedWorkerPool, WorkerPoolConfig
from Src.Phase3_Runtime.Shared.segment_worker import (
    execute_pytorch_range,
    init_pytorch_worker,
)
from Src.Shared.Config.paths import RESNET50_PATHS
from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor
from Src.Shared.Profiles.segment_profile import write_segment_profile


def _timed_sleep(delay):
    started = time.perf_counter()
    time.sleep(delay)
    return {"T_compute_s": time.perf_counter() - started}


class PartitionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_partition_manifest()

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dsci-segment-profile-"))
        self.previous_root = os.environ.get("DSCI_SEGMENT_PROFILE_ROOT")
        os.environ["DSCI_SEGMENT_PROFILE_ROOT"] = str(self.temp_dir)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop("DSCI_SEGMENT_PROFILE_ROOT", None)
        else:
            os.environ["DSCI_SEGMENT_PROFILE_ROOT"] = self.previous_root
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_profile(self, profile_id, scale, worker_count=1):
        samples = [
            [scale * (segment_id + 1), scale * (segment_id + 1) * 1.1]
            for segment_id in self.manifest.segment_ids
        ]
        return write_segment_profile(
            profile_id=profile_id,
            manifest=self.manifest,
            backend="pytorch",
            worker_count=worker_count,
            threads_per_worker=1,
            samples_s=samples,
            total_model_latency_s=scale * 100,
            exit_head_samples_s={57: [scale * 2], 103: [scale * 3]},
            profile_root=self.temp_dir,
        )

    def _fixed_state(self):
        self._write_profile("device", 0.001)
        self._write_profile("edge", 0.002, worker_count=2)
        self._write_profile("cloud", 0.003, worker_count=2)
        common = {
            "resource_mode": "fixed_worker_pool",
            "manifest_id": self.manifest.manifest_id,
            "model_hash": self.manifest.model_hash,
            "backend": "pytorch",
            "threads_per_worker": 1,
        }
        return {
            "resource_mode": "fixed_worker_pool",
            "users": [
                {
                    **common,
                    "execution_profile_id": "device",
                    "worker_count": 1,
                    "BW_d2e": 20,
                }
            ],
            "edge": {
                **common,
                "execution_profile_id": "edge",
                "worker_count": 2,
                "protocol_overhead_s": 0.002,
            },
            "cloud": {
                **common,
                "execution_profile_id": "cloud",
                "worker_count": 2,
                "BW_e2c": 100,
            },
        }

    def test_manifest_and_segment_chain_match_full_model(self):
        self.assertEqual(len(self.manifest.boundaries), 20)
        self.assertEqual(len(self.manifest.segments), 19)
        self.assertTrue(
            all(len(boundary["fx_live_values"]) == 1 for boundary in self.manifest.boundaries)
        )
        model = MultiEEResNet50(
            Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
        ).eval()
        model.load_state_dict(
            torch.load(
                RESNET50_PATHS.resolve_weight_path(),
                map_location="cpu",
                weights_only=True,
            )
        )
        executor = PyTorchSegmentExecutor(model, self.manifest)
        sample = torch.randn(1, 3, 64, 64)
        empty = executor.execute_range(0, 0, {"main": sample})
        self.assertIs(empty["main"], sample)
        with torch.no_grad():
            expected = model(sample, stage="final")
        bundle = executor.execute_range(0, 7, {"main": sample})
        bundle = executor.execute_range(7, 13, bundle)
        bundle = executor.execute_range(13, self.manifest.final_boundary_id, bundle)
        torch.testing.assert_close(bundle["logits"], expected)

    def test_multiple_legal_partition_pairs_match_full_model_and_exit_heads(self):
        model = MultiEEResNet50(
            Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
        ).eval()
        model.load_state_dict(
            torch.load(
                RESNET50_PATHS.resolve_weight_path(),
                map_location="cpu",
                weights_only=True,
            )
        )
        executor = PyTorchSegmentExecutor(model, self.manifest)
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected_final = model(sample, stage="final")
            expected_exit_57 = model(sample, stage="x2_fc")
            expected_exit_103 = model(sample, stage="x3_fc")

        for first, second in ((0, 1), (1, 8), (4, 14), (8, 14), (14, 18), (18, 19)):
            bundle = executor.execute_range(0, first, {"main": sample})
            bundle = executor.execute_range(first, second, bundle)
            bundle = executor.execute_range(second, self.manifest.final_boundary_id, bundle)
            torch.testing.assert_close(bundle["logits"], expected_final)

        bundle_8 = executor.execute_range(0, 8, {"main": sample})
        bundle_14 = executor.execute_range(8, 14, bundle_8)
        torch.testing.assert_close(executor.exit_logits(8, bundle_8), expected_exit_57)
        torch.testing.assert_close(executor.exit_logits(14, bundle_14), expected_exit_103)

    def test_manifest_serialized_bytes_match_runtime_payload(self):
        for boundary in self.manifest.boundaries:
            tensors = {
                item["name"]: torch.zeros(tuple(item["shape"]), dtype=torch.float32)
                for item in boundary["live_tensors"]
            }
            payload = {
                "manifest_id": self.manifest.manifest_id,
                "boundary_id": int(boundary["boundary_id"]),
                "tensors": tensors,
            }
            self.assertEqual(
                len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)),
                int(boundary["serialized_num_bytes"]),
            )

    def test_fixed_profiles_latency_and_decision_v2(self):
        state = self._fixed_state()
        paras = Paras.from_state(state)
        x = np.zeros((1, paras.m))
        x[0, 8] = 1
        x[0, 14] = 1
        p = np.zeros((1, paras.m))
        p[0, 127] = 1
        parts = compute_5_latency(x, p, np.zeros((1, 1)), np.zeros((1, 1)), paras)
        self.assertAlmostEqual(
            float(parts[0][0]),
            float(paras.segment_latency_u[0, :8].sum())
            + float(paras.exit_head_latency_u[0][57]),
        )
        self.assertGreater(float(parts[1][0]), 0)
        self.assertAlmostEqual(
            float(parts[2][0]),
            float(paras.segment_latency_e[8:14].sum())
            + float(paras.exit_head_latency_e[103]),
        )
        decision = encode(x, np.ones_like(x), np.zeros((1, 1)), np.zeros((1, 1)), paras)
        user = decision["users"][0]
        self.assertEqual(decision["resource_mode"], "fixed_worker_pool")
        self.assertEqual(decision["model_hash"], self.manifest.model_hash)
        self.assertEqual(user["partition_boundary_1"], 8)
        self.assertNotIn("edge_compute_quota", user)
        self.assertNotIn("device_layers", user)
        initial_x, _ = _init_feasible_XY(paras)
        self.assertTrue(
            set(np.flatnonzero(initial_x[0])).issubset(set(self.manifest.boundary_ids))
        )

        service = AlgoService(
            AlgoServiceConfig(
                auto_train=False,
                fixed_split=(8, 14),
                latest_solution_path=self.temp_dir / "latest.npz",
                latest_meta_path=self.temp_dir / "latest.json",
            )
        )
        service_decision = service.make_decision(state)
        self.assertEqual(service_decision["users"][0]["partition_boundary_2"], 14)

    def test_profile_backend_mismatch_is_rejected(self):
        state = self._fixed_state()
        state["edge"]["backend"] = "mnn"
        with self.assertRaises(ValueError):
            Paras.from_state(state)

    def test_profile_model_hash_mismatch_is_rejected(self):
        state = self._fixed_state()
        metadata_path = self.temp_dir / "edge" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_hash"] = "wrong-model-hash"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            Paras.from_state(state)

    def test_reported_model_hash_mismatch_is_rejected(self):
        state = self._fixed_state()
        state["edge"]["model_hash"] = "wrong-model-hash"
        with self.assertRaises(ValueError):
            Paras.from_state(state)

    def test_bf_fixed_worker_uses_manifest_boundaries_and_no_allocations(self):
        paras = Paras.from_state(self._fixed_state())
        _value, solution, _history = optimize_BF(
            paras,
            max_iter=1,
            restarts=1,
            threshold_step=1.0,
            verbose=False,
            F_opt_iters=1,
        )
        self.assertIsNotNone(solution)
        x, _y, f_e, f_c = solution
        self.assertTrue(
            set(np.flatnonzero(x[0])).issubset(set(self.manifest.boundary_ids))
        )
        np.testing.assert_array_equal(f_e, np.zeros_like(f_e))
        np.testing.assert_array_equal(f_c, np.zeros_like(f_c))

    def test_ga_fixed_worker_uses_manifest_boundaries_and_no_allocations(self):
        paras = Paras.from_state(self._fixed_state())
        _value, solution, _history = optimize_GA(
            paras, population_size=4, generations=1, mutation_rate=0.2
        )
        x, _y, f_e, f_c = solution
        self.assertTrue(
            set(np.flatnonzero(x[0])).issubset(set(self.manifest.boundary_ids))
        )
        np.testing.assert_array_equal(f_e, np.zeros_like(f_e))
        np.testing.assert_array_equal(f_c, np.zeros_like(f_c))

    def test_fixed_worker_pool_accepts_multiple_requests_without_queue_model(self):
        pool = FixedWorkerPool(WorkerPoolConfig(2, 1, 2), _timed_sleep)
        try:
            first = pool.submit(0.05)
            second = pool.submit(0.05)
            first_result = first.result(timeout=30)
            second_result = second.result(timeout=30)
            self.assertNotIn("T_queue_s", first_result)
            self.assertNotIn("T_queue_s", second_result)
            self.assertGreater(first_result["T_compute_s"], 0.0)
            self.assertGreater(second_result["T_compute_s"], 0.0)
        finally:
            pool.shutdown()

    def test_fixed_worker_pool_executes_multiple_real_segment_requests(self):
        pool = FixedWorkerPool(
            WorkerPoolConfig(2, 1, 2),
            execute_pytorch_range,
            initializer=init_pytorch_worker,
            initargs=(self.manifest.manifest_id,),
        )
        try:
            thresholds = {"57": 0.0, "103": 0.0}
            first = pool.submit(0, 14, {"main": torch.randn(1, 3, 64, 64)}, thresholds)
            second = pool.submit(0, 14, {"main": torch.randn(1, 3, 64, 64)}, thresholds)
            for result in (first.result(timeout=60), second.result(timeout=60)):
                self.assertEqual(result["executed_segments"], list(range(8)))
                self.assertEqual(tuple(result["tensors"]["main"].shape), (1, 512, 8, 8))
                self.assertIsNotNone(result["prediction"])
                self.assertEqual(result["exit_logical_layer"], 57)
                self.assertGreater(result["T_compute_s"], 0.0)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
