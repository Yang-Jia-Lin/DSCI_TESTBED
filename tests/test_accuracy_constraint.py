from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig
from Src.Phase3_Runtime.Device.run_device import (
    _decision_objective_weights,
    _measurement_record,
)


def _service(tmp_path: Path, **overrides) -> AlgoService:
    config = AlgoServiceConfig(
        latest_solution_path=tmp_path / "latest_solution.npz",
        latest_meta_path=tmp_path / "latest_solution_meta.json",
        training_events_path=tmp_path / "training_events.jsonl",
        **overrides,
    )
    return AlgoService(config=config)


class _FakeAgent:
    trained_alphas: list[float] = []

    def __init__(self, paras, _params):
        self.paras = paras
        self.best_policy_state_dict = {"alpha": float(paras.alpha)}

    def load_policy_state_dict(self, _state):
        return None

    def train(self, initial_solution=None):
        del initial_solution
        alpha = float(self.paras.alpha)
        self.trained_alphas.append(alpha)
        array = np.array([[alpha]], dtype=np.float32)
        resources = np.ones((1, 1), dtype=np.float32)
        return 0.0, (array, array, resources, resources), []


class AccuracyConstraintTests(unittest.TestCase):
    def test_constraint_config_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "target_accuracy"):
                _service(root, target_accuracy=0.0)
            with self.assertRaisesRegex(ValueError, "requires auto_train"):
                _service(root, target_accuracy=0.9, auto_train=False)

    def test_objective_config_separates_cache_keys(self):
        weighted = AlgoService._compat_key(
            {"objective": {"mode": "weighted", "alpha": 1.0, "beta": 1.0}}
        )
        other_weight = AlgoService._compat_key(
            {"objective": {"mode": "weighted", "alpha": 2.0, "beta": 1.0}}
        )
        constrained = AlgoService._compat_key(
            {
                "objective": {
                    "mode": "accuracy_constraint",
                    "target_accuracy": 0.9,
                    "accuracy_tolerance": 0.005,
                    "beta": 1.0,
                }
            }
        )
        self.assertNotEqual(weighted, other_weight)
        self.assertNotEqual(weighted, constrained)

    def _run_search(self, accuracy_fn, latency_fn):
        with TemporaryDirectory() as directory:
            service = _service(
                Path(directory),
                target_accuracy=0.9,
                accuracy_tolerance=0.0,
                constraint_search_runs=5,
            )
            _FakeAgent.trained_alphas = []

            def fake_to_paras(_state, algo_cfg=None):
                return SimpleNamespace(
                    n=1,
                    alpha=float(algo_cfg.alpha),
                    beta=float(algo_cfg.beta),
                )

            def annotate(solution, _paras, *, alpha=None, beta=None):
                solution.objective_alpha = float(alpha)
                solution.objective_beta = float(beta)
                solution.expected_accuracy = float(accuracy_fn(float(alpha)))
                solution.expected_latency = float(latency_fn(float(alpha)))
                solution.constraint_satisfied = solution.expected_accuracy >= 0.9
                solution.objective = (
                    float(alpha) * solution.expected_accuracy
                    - float(beta) * solution.expected_latency
                )
                return solution

            with (
                patch(
                    "Src.Phase2_Scheduler.Service.algo_service.to_paras",
                    side_effect=fake_to_paras,
                ),
                patch(
                    "Src.Phase2_Scheduler.Service.algo_service.PPOAgent",
                    _FakeAgent,
                ),
                patch.object(service, "_annotate_solution", side_effect=annotate),
                patch.object(service, "_append_training_event"),
            ):
                selected, _policy, _mode, _source = service._train_constraint_search(
                    {}, {"objective": service._objective_signature(SimpleNamespace(alpha=1.0, beta=1.0))}, None, "round"
                )
            return selected, list(_FakeAgent.trained_alphas)

    def test_search_changes_alpha_and_selects_lowest_latency_feasible(self):
        selected, alphas = self._run_search(
            lambda alpha: 0.92 if alpha >= 0.5 else 0.85,
            lambda alpha: alpha,
        )
        self.assertEqual(alphas[:3], [1.0, 0.5, 0.25])
        self.assertEqual(len(alphas), 5)
        self.assertAlmostEqual(selected.objective_alpha, 0.5)
        self.assertTrue(selected.constraint_satisfied)

    def test_unreachable_target_returns_highest_accuracy_candidate(self):
        selected, alphas = self._run_search(
            lambda alpha: min(0.89, 0.70 + 0.01 * alpha),
            lambda alpha: 1.0 / alpha,
        )
        self.assertEqual(alphas, [1.0, 2.0, 4.0, 8.0, 16.0])
        self.assertAlmostEqual(selected.objective_alpha, 16.0)
        self.assertFalse(selected.constraint_satisfied)

    def test_device_uses_decision_weights(self):
        alpha, beta = _decision_objective_weights(
            {"objective_alpha": 2.0, "objective_beta": 3.0}
        )
        record = _measurement_record(
            {
                "request_id": "request",
                "prediction": 1,
                "T_total": 0.25,
                "request_trace": {},
            },
            user_id=0,
            sample_index=0,
            label=1,
            is_correct=True,
            objective_alpha=alpha,
            objective_beta=beta,
        )
        self.assertAlmostEqual(record["observed_utility"], 1.25)
        self.assertEqual(record["objective_alpha"], 2.0)
        self.assertEqual(record["objective_beta"], 3.0)

    def test_scheduler_reward_uses_decision_weights(self):
        with TemporaryDirectory() as directory:
            service = _service(Path(directory))
            result = service.report_measurements(
                {
                    "decision_id": "round",
                    "objective_alpha": 2.0,
                    "objective_beta": 3.0,
                    "measurements": [
                        {"user_id": 0, "is_correct": 1.0, "T_total": 0.25}
                    ],
                }
            )
        self.assertAlmostEqual(result["utility_sum"], 1.25)


if __name__ == "__main__":
    unittest.main()
