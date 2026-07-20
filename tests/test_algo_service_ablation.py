from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Src.Phase2_Scheduler.Service.algo_service import (
    AlgoService,
    AlgoServiceConfig,
)
from Scripts.EvaluationCommon.solutions import state_from_solution_meta
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.split_actions import decode_split_row, encode_split_row


class _FakeAgent:
    trained_initial_solutions: list[object] = []

    def __init__(self, paras, params):
        self.paras = paras
        self.params = params

    def train(self, *, initial_solution=None):
        self.trained_initial_solutions.append(initial_solution)
        X = np.tile(
            encode_split_row(1, 2, self.paras.m, dtype=np.float32),
            (self.paras.n, 1),
        )
        Y = np.full((self.paras.n, self.paras.m), 1.0, dtype=np.float32)
        exit_boundaries = list(getattr(self.paras, "E", (1, 2)))
        for index, boundary in enumerate(exit_boundaries):
            Y[:, int(boundary)] = 0.2 + 0.1 * index
        F_e = np.zeros((self.paras.n, 1), dtype=np.float32)
        F_c = np.zeros((self.paras.n, 1), dtype=np.float32)
        return 1.0, (X, Y, F_e, F_c), []


def _paras():
    return SimpleNamespace(
        n=1,
        m=4,
        alpha=1.0,
        beta=5.0,
        partition_manifest=SimpleNamespace(final_boundary_id=3),
    )


def _resnet_imagenet100_state():
    bundle_id = "resnet50-imagenet100"
    manifest = load_partition_manifest(bundle_id)
    meta = {
        "state_signature": {
            "model": {"bundle_id": bundle_id},
            "manifest_id": manifest.manifest_id,
            "model_hash": manifest.model_hash,
            "resource_mode": "fixed_worker_pool",
            "users": [
                {
                    "user_id": 0,
                    "BW_d2e": 10.0,
                    "execution_profile_id": f"{bundle_id}-device-pi5",
                }
            ],
            "edge": {
                "execution_profile_id": f"{bundle_id}-edge-jialin-desktop",
            },
            "cloud": {
                "execution_profile_id": f"{bundle_id}-cloud-v100",
                "BW_e2c": 50.0,
            },
        }
    }
    state = state_from_solution_meta(meta)
    state["round_id"] = "ablation-integration-test"
    return state


class AlgoServiceAblationTests(unittest.TestCase):
    def test_split_only_keeps_x_and_disables_all_exits(self):
        paras = _paras()
        X = np.tile(encode_split_row(1, 2, 4), (1, 1))
        Y = np.array([[1.0, 0.3, 0.6, 1.0]], dtype=np.float32)
        X_out, Y_out = AlgoService._ablation_arrays("split-only", X, Y, paras)
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(Y_out, np.ones_like(Y))

    def test_ee_only_forces_pure_device_and_keeps_y(self):
        paras = _paras()
        X = np.tile(encode_split_row(1, 2, 4), (1, 1))
        Y = np.array([[1.0, 0.3, 0.6, 1.0]], dtype=np.float32)
        X_out, Y_out = AlgoService._ablation_arrays("ee-only", X, Y, paras)
        self.assertEqual(decode_split_row(X_out[0]), (3, 3))
        np.testing.assert_array_equal(Y_out, Y)

    def test_ablation_mode_skips_disk_cache_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solution = root / "latest_solution.npz"
            meta = root / "latest_solution_meta.json"
            solution.write_bytes(b"not an npz")
            meta.write_text("not json", encoding="utf-8")
            service = AlgoService(
                AlgoServiceConfig(
                    ablation_mode="split-only",
                    latest_solution_path=solution,
                    latest_meta_path=meta,
                    training_events_path=root / "events.jsonl",
                )
            )
            health = service.health()
            self.assertFalse(health["has_cached_solution"])
            self.assertEqual(health["cache_entries"], 0)
            self.assertFalse(health["policy_cache_enabled"])

    def test_fresh_solve_does_not_warm_start_or_store_cache(self):
        _FakeAgent.trained_initial_solutions.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AlgoService(
                AlgoServiceConfig(
                    ablation_mode="split-only",
                    latest_solution_path=root / "latest_solution.npz",
                    latest_meta_path=root / "latest_solution_meta.json",
                    training_events_path=root / "events.jsonl",
                )
            )
            paras = _paras()

            def annotate(solution, _paras):
                solution.objective = float(np.sum(solution.X) - np.sum(solution.Y))
                solution.objective_alpha = 1.0
                solution.objective_beta = 5.0
                solution.expected_accuracy = 0.8
                solution.expected_latency = 0.1
                return solution

            zeros = np.zeros((1, 1), dtype=np.float32)
            with (
                patch(
                    "Src.Phase2_Scheduler.Service.algo_service.PPOAgent",
                    _FakeAgent,
                ),
                patch.object(service, "_annotate_solution", side_effect=annotate),
                patch.object(
                    service,
                    "_allocate_resources_for_xy",
                    return_value=(zeros, zeros),
                ),
            ):
                result, source_objective, _duration = service._solve_fresh_ablation(
                    {"round_id": "test-round"}, {"model": {"bundle_id": "test"}}, paras
                )

            self.assertEqual(_FakeAgent.trained_initial_solutions, [None])
            np.testing.assert_array_equal(result.Y, np.ones_like(result.Y))
            self.assertIsInstance(source_objective, float)
            self.assertIsNone(service._cached_solution)
            self.assertEqual(service._cache_entries, [])
            self.assertFalse((root / "latest_solution.npz").exists())
            event = json.loads((root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "fresh_ablation_solve_complete")
            self.assertFalse(event["cache_used"])

    def test_make_decision_returns_deployable_transformed_solution(self):
        state = _resnet_imagenet100_state()
        for mode in ("split-only", "ee-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                service = AlgoService(
                    AlgoServiceConfig(
                        ablation_mode=mode,
                        latest_solution_path=root / "latest_solution.npz",
                        latest_meta_path=root / "latest_solution_meta.json",
                        training_events_path=root / "events.jsonl",
                    )
                )
                with patch(
                    "Src.Phase2_Scheduler.Service.algo_service.PPOAgent",
                    _FakeAgent,
                ):
                    decision = service.make_decision(state)

                self.assertEqual(
                    decision["decision_source"],
                    f"synchronous:ppo:fresh:ablation:{mode}",
                )
                self.assertFalse(decision["cache_used"])
                user = decision["users"][0]
                if mode == "split-only":
                    self.assertEqual(
                        (user["partition_boundary_1"], user["partition_boundary_2"]),
                        (1, 2),
                    )
                    self.assertTrue(
                        all(value == 1.0 for value in user["exit_thresholds"].values())
                    )
                else:
                    self.assertEqual(
                        (user["partition_boundary_1"], user["partition_boundary_2"]),
                        (19, 19),
                    )
                    np.testing.assert_allclose(
                        list(user["exit_thresholds"].values()),
                        [0.2, 0.3, 0.4],
                    )
                self.assertFalse((root / "latest_solution.npz").exists())

    def test_ablation_rejects_fixed_controls_and_force_retrain(self):
        with self.assertRaises(ValueError):
            AlgoService(
                AlgoServiceConfig(
                    ablation_mode="split-only",
                    fixed_split=(1, 2),
                )
            )
        with self.assertRaises(ValueError):
            AlgoService(
                AlgoServiceConfig(ablation_mode="ee-only", force_retrain=True)
            )


if __name__ == "__main__":
    unittest.main()
