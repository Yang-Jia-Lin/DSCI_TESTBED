import unittest

from Src.Phase2_Scheduler.Service.api_server import create_app
from Src.Phase2_Scheduler.Service.reward_adapter import compute_user_reward
from Src.Phase2_Scheduler.Service.round_coordinator import (
    RoundConflictError,
    RoundCoordinator,
)


class FakeService:
    def __init__(self):
        self.decision_calls = []
        self.measurement_calls = []

    def make_decision(self, state):
        self.decision_calls.append(state)
        return {
            "round_id": state["round_id"],
            "decision_id": state["round_id"],
            "decision_version": 1,
            "bundle_id": state["bundle_id"],
            "manifest_id": "manifest",
            "model_hash": "hash",
            "resource_mode": "fixed_worker_pool",
            "users": [
                {
                    "user_id": user["user_id"],
                    "partition_boundary_1": 1,
                    "partition_boundary_2": 2,
                    "exit_thresholds": {},
                }
                for user in state["users"]
            ],
        }

    def report_measurements(self, payload):
        self.measurement_calls.append(payload)
        return {"status": "ok"}

    def health(self):
        return {"status": "ok"}


def node_state(role):
    state = {
        "bundle_id": "bundle",
        "manifest_id": "manifest",
        "model_hash": "hash",
        "execution_profile_id": role,
        "backend": "pytorch",
        "worker_count": 1,
        "threads_per_worker": 1,
    }
    if role == "cloud":
        state["BW_e2c"] = 10.0
    return state


def registration(user_id, bandwidth=10.0):
    return {
        "user_id": user_id,
        "bundle_id": "bundle",
        "resource_mode": "fixed_worker_pool",
        "device": {
            **node_state(f"device-{user_id}"),
            "BW_d2e": bandwidth,
        },
    }


class RoundCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.service = FakeService()
        self.coordinator = RoundCoordinator(
            self.service,
            expected_users=2,
            node_state_provider=lambda: (node_state("edge"), node_state("cloud")),
            clock=lambda: self.now,
        )

    def test_waits_for_fixed_barrier_and_optimizes_once(self):
        first = self.coordinator.register("round-a", registration(9))
        self.assertEqual(first["status"], "waiting")
        self.assertEqual(len(self.service.decision_calls), 0)

        second = self.coordinator.register("round-a", registration(3))
        self.assertEqual(second["status"], "ready")
        self.assertEqual(len(self.service.decision_calls), 1)
        users = self.service.decision_calls[0]["users"]
        self.assertEqual([user["user_id"] for user in users], [3, 9])

        decision = self.coordinator.decision_for_user("round-a", 9)
        self.assertEqual(decision["user"]["user_id"], 9)

    def test_registration_is_idempotent_but_conflicting_state_fails(self):
        self.coordinator.register("round-a", registration(1))
        self.coordinator.register("round-a", registration(1))
        with self.assertRaises(RoundConflictError):
            self.coordinator.register("round-a", registration(1, bandwidth=5.0))

    def test_heartbeat_timeout_fails_waiting_round(self):
        self.coordinator.register("round-a", registration(1))
        self.now += 16.0
        status = self.coordinator.status("round-a")
        self.assertEqual(status["status"], "failed")
        self.assertIn("Heartbeat timeout", status["error"])

    def test_completes_after_all_users_submit_measurements(self):
        self.coordinator.register("round-a", registration(1))
        self.coordinator.register("round-a", registration(2))
        base = {"decision_id": "round-a", "decision_version": 1}
        first = self.coordinator.submit_measurements(
            "round-a",
            1,
            {
                **base,
                "measurements": [
                    {"request_id": "a", "T_total": 1.0, "is_correct": 0.25},
                    {"request_id": "b", "T_total": 3.0, "is_correct": 0.75},
                ],
            },
        )
        self.assertEqual(first["status"], "ready")
        completed = self.coordinator.submit_measurements(
            "round-a",
            2,
            {
                **base,
                "measurements": [
                    {"request_id": "c", "T_total": 2.0, "is_correct": True}
                ],
            },
        )
        self.assertEqual(completed["status"], "completed")
        records = self.service.measurement_calls[0]["measurements"]
        self.assertEqual(records[0]["T_total"], 2.0)
        self.assertEqual(records[0]["is_correct"], 0.5)

    def test_node_state_provider_runs_outside_coordinator_lock(self):
        service = FakeService()
        coordinator = None

        def provider():
            is_owned = getattr(coordinator._lock, "_is_owned", lambda: False)
            self.assertFalse(is_owned())
            return node_state("edge"), node_state("cloud")

        coordinator = RoundCoordinator(
            service,
            expected_users=2,
            node_state_provider=provider,
            clock=lambda: self.now,
        )
        coordinator.register("round-a", registration(1))
        status = coordinator.register("round-a", registration(2))

        self.assertEqual(status["status"], "ready")
        self.assertEqual(len(service.decision_calls), 1)

    def test_reward_accepts_per_user_average_accuracy(self):
        self.assertEqual(
            compute_user_reward(0.5, 2.0, alpha=4.0, beta=0.5),
            1.0,
        )

    def test_rejects_request_id_reused_by_another_user(self):
        self.coordinator.register("round-a", registration(1))
        self.coordinator.register("round-a", registration(2))
        base = {"decision_id": "round-a", "decision_version": 1}
        self.coordinator.submit_measurements(
            "round-a",
            1,
            {
                **base,
                "measurements": [
                    {"request_id": "same", "T_total": 1.0, "is_correct": True}
                ],
            },
        )
        with self.assertRaises(ValueError):
            self.coordinator.submit_measurements(
                "round-a",
                2,
                {
                    **base,
                    "measurements": [
                        {"request_id": "same", "T_total": 1.0, "is_correct": True}
                    ],
                },
            )

    def test_completed_round_id_cannot_be_reused(self):
        self.coordinator.register("round-a", registration(1))
        self.coordinator.register("round-a", registration(2))
        base = {"decision_id": "round-a", "decision_version": 1}
        for user_id in (1, 2):
            self.coordinator.submit_measurements(
                "round-a",
                user_id,
                {
                    **base,
                    "measurements": [
                        {
                            "request_id": f"request-{user_id}",
                            "T_total": 1.0,
                            "is_correct": True,
                        }
                    ],
                },
            )
        with self.assertRaises(RoundConflictError):
            self.coordinator.register("round-a", registration(1))


class RoundApiTests(unittest.TestCase):
    def test_decision_endpoint_returns_202_until_barrier_closes(self):
        service = FakeService()
        coordinator = RoundCoordinator(
            service,
            expected_users=2,
            node_state_provider=lambda: (node_state("edge"), node_state("cloud")),
        )
        client = create_app(service, coordinator).test_client()
        response = client.post(
            "/api/v2/rounds/r/devices/register", json=registration(1)
        )
        self.assertEqual(response.status_code, 200)
        response = client.get("/api/v2/rounds/r/decisions/1")
        self.assertEqual(response.status_code, 202)

        client.post("/api/v2/rounds/r/devices/register", json=registration(2))
        response = client.get("/api/v2/rounds/r/decisions/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["user"]["user_id"], 1)


if __name__ == "__main__":
    unittest.main()
