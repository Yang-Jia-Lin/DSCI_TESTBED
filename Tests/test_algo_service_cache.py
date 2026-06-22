import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from Src.Phase2_Scheduler.Service import algo_service as service_mod
from Src.Phase2_Scheduler.Service.algo_service import (
    AlgoService,
    AlgoServiceConfig,
    CachedSolution,
    CacheMatch,
)


class FakeManifest:
    model_hash = "hash"


class FakeParas:
    def __init__(self, *, bandwidths=(10.0, 20.0), resource_mode="fixed_worker_pool"):
        self.n = len(bandwidths)
        self.m = 4
        self.E = [1, 2]
        self.exit_ids = ["e1", "e2"]
        self.bundle_id = "bundle"
        self.manifest_id = "manifest"
        self.partition_manifest = FakeManifest()
        self.partition_boundary_ids = [0, 1, 2, 3]
        self.resource_mode = resource_mode
        self.F_u = np.ones(self.n, dtype=float) * 2e9
        self.B_u = np.asarray(bandwidths, dtype=float)
        self.f_e_max = 20e9
        self.f_c_max = 50e9
        self.b_c = 120.0


def fake_state(*, bandwidths=(10.0, 20.0), resource_mode="fixed_worker_pool"):
    users = [
        {
            "user_id": i,
            "BW_d2e": bw,
            "f_u": 2e9,
            "execution_profile_id": f"user-{i}",
        }
        for i, bw in enumerate(bandwidths)
    ]
    return {
        "round_id": "round",
        "resource_mode": resource_mode,
        "users": users,
        "edge": {"execution_profile_id": "edge", "worker_count": 1},
        "cloud": {
            "execution_profile_id": "cloud",
            "worker_count": 1,
            "BW_e2c": 120.0,
        },
    }


def fake_to_paras(state):
    return FakeParas(
        bandwidths=tuple(user["BW_d2e"] for user in state["users"]),
        resource_mode=state.get("resource_mode", "fixed_worker_pool"),
    )


def fake_encode(X, Y, F_e, F_c, paras, **kwargs):
    return {
        "decision_id": kwargs["decision_id"],
        "users": [{"user_id": uid} for uid in kwargs["user_ids"]],
    }


class AlgoServiceCacheTests(unittest.TestCase):
    def make_service(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        return AlgoService(
            config=AlgoServiceConfig(
                auto_train=True,
                latest_solution_path=root / "latest_solution.npz",
                latest_meta_path=root / "latest_solution_meta.json",
            )
        )

    def cached_solution(self, svc, state):
        paras = fake_to_paras(state)
        signature = svc._state_signature(state, paras)
        return CachedSolution(
            X=np.zeros((paras.n, paras.m), dtype=np.float32),
            Y=np.ones((paras.n, paras.m), dtype=np.float32),
            F_e=np.zeros((paras.n, 1), dtype=np.float32),
            F_c=np.zeros((paras.n, 1), dtype=np.float32),
            objective=1.0,
            state_signature=signature,
            compat_key=svc._compat_key(signature),
            state_vector=svc._state_vector(signature),
            policy_path="policy.pt",
        )

    def test_state_distance_exact_and_small_perturbation(self):
        svc = self.make_service()
        base = fake_state(bandwidths=(10.0, 20.0))
        small = fake_state(bandwidths=(10.01, 20.0))
        paras = fake_to_paras(base)

        sig_a = svc._state_signature(base, paras)
        sig_b = svc._state_signature(small, fake_to_paras(small))

        self.assertEqual(
            svc._state_distance(svc._state_vector(sig_a), svc._state_vector(sig_a)),
            0.0,
        )
        distance = svc._state_distance(svc._state_vector(sig_a), svc._state_vector(sig_b))
        self.assertLessEqual(distance, service_mod.DIRECT_REUSE_DISTANCE)

    def test_incompatible_profile_does_not_match(self):
        svc = self.make_service()
        state = fake_state()
        paras = fake_to_paras(state)
        svc._remember_cache_entry(self.cached_solution(svc, state))

        changed = fake_state()
        changed["edge"]["execution_profile_id"] = "different-edge"
        signature = svc._state_signature(changed, fake_to_paras(changed))

        self.assertIsNone(svc._best_cache_match(signature, paras))

    def test_tiny_perturbation_reuses_cache_without_training(self):
        svc = self.make_service()
        base = fake_state(bandwidths=(10.0, 20.0))
        svc._cached_solution = self.cached_solution(svc, base)
        svc._remember_cache_entry(svc._cached_solution)
        changed = fake_state(bandwidths=(10.01, 20.0))

        with (
            mock.patch.object(service_mod, "to_paras", side_effect=fake_to_paras),
            mock.patch.object(service_mod, "encode", side_effect=fake_encode),
            mock.patch.object(service_mod, "objective", return_value=1.0),
            mock.patch.object(svc, "_start_training_locked") as start_training,
        ):
            decision = svc.make_decision(changed)

        self.assertIn("cached_dsci:reuse", decision["decision_source"])
        start_training.assert_not_called()

    def test_adaptive_training_params_for_near_and_medium(self):
        svc = self.make_service()
        match = CacheMatch(
            solution=self.cached_solution(svc, fake_state()),
            distance=0.01,
            training_mode="near",
            policy_path="policy.pt",
        )
        params = svc._training_params(match)
        self.assertEqual(params["max_epochs"], 30)
        self.assertEqual(params["min_epochs"], 8)
        self.assertEqual(params["target_steps"], 400)

        match.training_mode = "medium"
        params = svc._training_params(match)
        self.assertEqual(params["max_epochs"], 80)
        self.assertEqual(params["min_epochs"], 20)
        self.assertEqual(params["outer_ema"], 0.5)

    def test_background_training_loads_near_policy(self):
        svc = self.make_service()
        state = fake_state(bandwidths=(10.1, 20.0))
        signature = svc._state_signature(state, fake_to_paras(state))
        source = self.cached_solution(svc, fake_state())
        match = CacheMatch(
            solution=source,
            distance=0.01,
            training_mode="near",
            policy_path="warm-policy.pt",
        )

        class FakeAgent:
            loaded_paths = []
            train_initial_solution = None
            params = None

            def __init__(self, paras, params):
                self.paras = paras
                self.hparams = params
                FakeAgent.params = params
                self.best_policy_state_dict = {"w": np.array([1.0])}

            def load_checkpoint(self, path):
                FakeAgent.loaded_paths.append(path)

            def train(self, initial_solution=None):
                FakeAgent.train_initial_solution = initial_solution
                return 2.0, (
                    source.X.copy(),
                    source.Y.copy(),
                    source.F_e.copy(),
                    source.F_c.copy(),
                ), [2.0]

        with (
            mock.patch.object(service_mod, "to_paras", side_effect=fake_to_paras),
            mock.patch.object(service_mod, "PPOAgent", FakeAgent),
            mock.patch.object(service_mod.torch, "save"),
        ):
            svc._train_background(state, signature, match)

        self.assertEqual(FakeAgent.loaded_paths, ["warm-policy.pt"])
        self.assertIsNotNone(FakeAgent.train_initial_solution)
        self.assertEqual(FakeAgent.params["max_epochs"], 30)
        self.assertEqual(svc.health()["training_status"], "idle")


if __name__ == "__main__":
    unittest.main()
