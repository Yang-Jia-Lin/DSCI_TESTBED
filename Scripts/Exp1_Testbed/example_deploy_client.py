"""
Minimal HTTP client example for the deploy module (reference only).

Usage (algo server must be running):
    python Scripts/Exp1_Testbed/example_deploy_client.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    state = {
        "round_id": "round_demo_001",
        "model_name": "Resnet50",
        "users": [
            {"user_id": 0, "BW_d2e": 20.0, "f_u": 2.0},
            {"user_id": 1, "BW_d2e": 15.0, "f_u": 2.0},
        ],
        "edge": {"f_e_max": 20.0},
        "cloud": {"BW_e2c": 100.0, "f_c_max": 50.0},
    }

    try:
        health = urllib.request.urlopen(f"{base}/api/v1/health", timeout=10)
        print("health:", health.read().decode())
    except urllib.error.URLError as exc:
        print(f"Cannot reach server at {base}: {exc}")
        sys.exit(1)

    decision = _post(f"{base}/api/v1/decision", state)
    print("decision_id:", decision.get("decision_id"))
    print("user0 layers:", decision["users"][0]["device_layers"])

    # Deploy would run real inference here; we send dummy measurements.
    measurements = {
        "decision_id": decision["decision_id"],
        "measurements": [
            {"user_id": i, "T_total": 1.0 + i, "is_correct": True}
            for i in range(decision["num_users"])
        ],
    }
    ack = _post(f"{base}/api/v1/measurements", measurements)
    print("measurements ack:", json.dumps(ack, indent=2))


if __name__ == "__main__":
    main()
