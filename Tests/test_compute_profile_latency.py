from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from typing import cast
import pandas as pd

from Src.Phase2_Scheduler.Objective.compute_latency import (
    compute_5_latency,
    compute_total_latency,
    compute_user_latency,
)
from Src.Phase2_Scheduler.Optimizer.DSCI.agent import compute_iota_kappa
from Src.Phase2_Scheduler.paras import Paras
from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig
from Src.Shared.Config.model_config import RESNET50
from Src.Shared.Config.paths import RESNET50_PATHS
from Src.Shared.Profiles.compute_profile import (
    ComputeProfileError,
    load_compute_profile,
    write_compute_profile,
)


class ComputeProfileLatencyTests(unittest.TestCase):
    def setUp(self):
        self.profile_root = Path(__file__).resolve().parent / ".profile_test_root"
        shutil.rmtree(self.profile_root, ignore_errors=True)
        self.profile_root.mkdir(parents=True)
        self.previous_root = os.environ.get("DSCI_COMPUTE_PROFILE_ROOT")
        os.environ["DSCI_COMPUTE_PROFILE_ROOT"] = str(self.profile_root)
        stats = pd.read_csv(
            RESNET50_PATHS.resolve_layer_stats_csv(), skipinitialspace=True
        )
        stats.columns = [str(column).strip() for column in stats.columns]
        self.names = stats["layer"].astype(str).tolist()
        self.flops = stats["approx_flops"].to_numpy(dtype=np.float64)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop("DSCI_COMPUTE_PROFILE_ROOT", None)
        else:
            os.environ["DSCI_COMPUTE_PROFILE_ROOT"] = self.previous_root
        shutil.rmtree(self.profile_root, ignore_errors=True)

    def make_profile(self, profile_id, scale, backend="pytorch", reverse=False):
        raw = np.arange(1, len(self.flops) + 1, dtype=np.float64) * scale
        if reverse:
            raw = raw[::-1].copy()
        return write_compute_profile(
            profile_id=profile_id,
            layer_names=self.names,
            theoretical_flops=self.flops,
            raw_latencies_s=raw,
            total_latency_s=float(raw.sum() * 0.8),
            model_name=RESNET50.name,
            backend=backend,
        )

    def test_profile_conservation(self):
        profile = self.make_profile("device-a", 1e-6)
        loaded = load_compute_profile("device-a", expected_layers=self.names)
        self.assertAlmostEqual(
            float(loaded.layers["calibrated_latency_s"].sum()),
            loaded.total_latency_s,
        )
        self.assertAlmostEqual(
            float(loaded.equivalent_flops.sum()),
            float(self.flops.sum()),
            places=4,
        )
        self.assertAlmostEqual(
            float(loaded.equivalent_flops.sum() / loaded.theta),
            profile.total_latency_s,
        )

    def test_from_state_loads_heterogeneous_profiles(self):
        user_a = self.make_profile("user-a", 1e-6)
        user_b = self.make_profile("user-b", 2e-6, reverse=True)
        edge = self.make_profile("edge", 3e-6)
        cloud = self.make_profile("cloud", 4e-6)
        state = {
            "users": [
                {"compute_profile_id": "user-a", "f_u": user_a.theta, "BW_d2e": 10},
                {"compute_profile_id": "user-b", "f_u": user_b.theta, "BW_d2e": 20},
            ],
            "edge": {"compute_profile_id": "edge", "f_e_max": edge.theta},
            "cloud": {
                "compute_profile_id": "cloud",
                "f_c_max": cloud.theta,
                "BW_e2c": 100,
            },
        }
        paras = Paras.from_state(state)
        self.assertIsNotNone(paras.C_u)
        assert paras.C_u is not None
        self.assertEqual(paras.C_u.shape, (2, 128))
        self.assertFalse(np.allclose(paras.C_u[0], paras.C_u[1]))
        self.assertIsNotNone(paras.C_e)
        assert paras.C_e is not None
        np.testing.assert_allclose(cast(np.ndarray, paras.C_e), edge.equivalent_flops)
        self.assertIsNotNone(paras.C_c)
        np.testing.assert_allclose(cast(np.ndarray, paras.C_c), cloud.equivalent_flops)

        service = AlgoService(
            AlgoServiceConfig(
                auto_train=False,
                fixed_split=(10, 20),
                fixed_threshold=0.7,
                latest_solution_path=self.profile_root / "latest_solution.npz",
                latest_meta_path=self.profile_root / "latest_solution_meta.json",
            )
        )
        decision = service.make_decision(state)
        self.assertEqual(decision["users"][0]["partition_s1"], 10)
        self.assertEqual(decision["users"][0]["partition_s2"], 20)
        self.assertTrue(
            all(
                threshold == 0.7
                for threshold in decision["users"][0]["exit_thresholds"].values()
            )
        )

    def test_missing_profile_fails(self):
        with self.assertRaises((ComputeProfileError, KeyError)):
            Paras.from_state(
                {
                    "users": [{"compute_profile_id": "missing", "f_u": 1, "BW_d2e": 1}],
                    "edge": {"compute_profile_id": "missing", "f_e_max": 1},
                    "cloud": {
                        "compute_profile_id": "missing",
                        "f_c_max": 1,
                        "BW_e2c": 1,
                    },
                }
            )

    def test_latency_helpers_use_device_specific_work(self):
        n, m = 1, 128
        c_u = np.ones((n, m))
        c_e = np.ones(m) * 2
        c_c = np.ones(m) * 3
        paras = Paras(
            n=n,
            F_u=np.array([10.0]),
            H_u=np.array([1.0]),
            C_u=c_u,
            C_e=c_e,
            C_c=c_c,
            f_e_max=20.0,
            f_c_max=30.0,
        )
        x = np.zeros((n, m))
        x[0, 10] = 1
        x[0, 20] = 1
        p = np.zeros((n, m))
        p[0, 127] = 1
        f_e = np.array([[20.0]])
        f_c = np.array([[30.0]])
        total = compute_total_latency(x, p, f_e, f_c, paras)[0]
        parts = compute_5_latency(x, p, f_e, f_c, paras)
        single = compute_user_latency(0, 10, 20, p[0], 20.0, 30.0, paras)
        self.assertAlmostEqual(total, sum(float(part[0]) for part in parts))
        self.assertAlmostEqual(total, single)
        self.assertAlmostEqual(float(parts[0][0]), 10 / 10)
        self.assertAlmostEqual(float(parts[2][0]), 20 / 20)
        self.assertAlmostEqual(float(parts[4][0]), 108 * 3 / 30)

    def test_iota_kappa_match_expected_segments(self):
        x = np.zeros((1, 6))
        x[0, 2] = 1
        x[0, 4] = 1
        p = np.zeros((1, 6))
        p[0, 5] = 1
        iota, kappa = compute_iota_kappa(
            x,
            np.array([1, 1, 2, 2, 2, 2]),
            np.array([1, 1, 3, 3, 3, 3]),
            p,
        )
        self.assertEqual(float(iota[0]), 4.0)
        self.assertEqual(float(kappa[0]), 6.0)


if __name__ == "__main__":
    unittest.main()
