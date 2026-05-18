"""
Smoke test for deploy-facing JSON interface (no HTTP).

Run from repo root:
    python Scripts/Exp1_Testbed/test_interface_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Src.Algorithm.Interface.algo_service import AlgoService, AlgoServiceConfig


def main() -> None:
    state = {
        "round_id": "round_0001",
        "model_name": "Resnet50",
        "users": [
            {"user_id": 0, "BW_d2e": 18.5, "f_u": 2.0},
            {"user_id": 1, "BW_d2e": 12.0, "f_u": 1.8},
        ],
        "edge": {"f_e_max": 20.0, "cpu_util": 0.6},
        "cloud": {"BW_e2c": 120.0, "f_c_max": 50.0, "cpu_util": 0.4},
    }

    svc = AlgoService(config=AlgoServiceConfig(enable_training=False))
    decision = svc.make_decision(state)
    print("=== Decision JSON (excerpt) ===")
    print(json.dumps({k: decision[k] for k in decision if k != "users"}, indent=2))
    print("user[0]:", json.dumps(decision["users"][0], indent=2)[:500], "...")

    measurements = {
        "decision_id": "round_0001",
        "measurements": [
            {
                "user_id": 0,
                "T_device": 1.0,
                "T_trans_d2e": 0.5,
                "T_edge": 2.0,
                "T_trans_e2c": 0.1,
                "T_cloud": 1.0,
                "T_total": 4.6,
                "exit_layer": 103,
                "exit_location": "edge",
                "exit_confidence": 0.9,
                "prediction": 1,
                "ground_truth": 1,
                "is_correct": True,
            },
            {
                "user_id": 1,
                "T_device": 0.8,
                "T_trans_d2e": 0,
                "T_edge": 0,
                "T_trans_e2c": 0,
                "T_cloud": 0,
                "T_total": 0.8,
                "exit_layer": 57,
                "exit_location": "device",
                "exit_confidence": 0.95,
                "prediction": 2,
                "ground_truth": 2,
                "is_correct": True,
            },
        ],
    }
    resp = svc.report_measurements(measurements)
    print("\n=== Measurements response ===")
    print(json.dumps(resp, indent=2))
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
