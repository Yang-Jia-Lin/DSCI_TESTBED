from __future__ import annotations

import socket
import threading
import unittest

from Src.Phase2_Scheduler.Service.round_coordinator import RoundCoordinator
from Src.Phase3_Runtime.Device.comm import send_tensor
from Src.Phase3_Runtime.Shared.dynamic_bandwidth import BandwidthEstimator
from Src.Phase3_Runtime.Shared.socket_server import _handle_connection


class FakeAlgoService:
    def __init__(self):
        self.states = []
        self.reported = []

    def make_decision(self, state):
        self.states.append(state)
        return {
            "decision_id": state["round_id"],
            "round_id": state["round_id"],
            "decision_version": 1,
            "bundle_id": state["bundle_id"],
            "manifest_id": state["edge"]["manifest_id"],
            "model_hash": state["edge"]["model_hash"],
            "resource_mode": "fixed_worker_pool",
            "decision_source": "fake",
            "objective": 0.0,
            "users": [
                {
                    "user_id": user["user_id"],
                    "partition_boundary_1": 0,
                    "partition_boundary_2": 1,
                    "exit_thresholds": {},
                }
                for user in state["users"]
            ],
        }

    def report_measurements(self, payload):
        self.reported.append(payload)
        return {"status": "ok"}


def owner(role: str) -> dict:
    return {
        "bundle_id": "bundle",
        "manifest_id": "manifest",
        "model_hash": "hash",
        "execution_profile_id": f"profile-{role}",
        "backend": "pytorch",
        "worker_count": 1,
        "threads_per_worker": 1,
    }


def registration(user_id: int, *, dynamic: bool = True) -> dict:
    return {
        "user_id": user_id,
        "bundle_id": "bundle",
        "resource_mode": "fixed_worker_pool",
        "dynamic_bandwidth": dynamic,
        "device": {**owner(f"device-{user_id}"), "BW_d2e": 10.0},
    }


class BandwidthEstimatorTests(unittest.TestCase):
    def test_passive_samples_replace_calibration_after_three_large_payloads(self):
        clock = [100.0]
        estimator = BandwidthEstimator(
            link="d2e",
            initial_mbps=10.0,
            alpha=0.5,
            stale_after_s=30.0,
            clock=lambda: clock[0],
        )
        estimator.observe(20.0, source="iperf")
        self.assertEqual(estimator.effective_mbps(), 20.0)
        self.assertIsNone(
            estimator.observe(100.0, source="passive", payload_bytes=1024)
        )
        estimator.observe(40.0, source="passive", payload_bytes=300_000)
        estimator.observe(60.0, source="passive", payload_bytes=300_000)
        self.assertEqual(estimator.effective_mbps(), 20.0)
        sample = estimator.observe(
            80.0, source="passive", payload_bytes=300_000
        )
        self.assertAlmostEqual(sample.filtered_bw_mbps, 65.0)
        clock[0] = 131.0
        self.assertEqual(estimator.effective_mbps(), 10.0)
        self.assertTrue(estimator.needs_calibration())


class TransportAckTests(unittest.TestCase):
    def test_optional_upload_ack_returns_goodput(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()

        def serve_once():
            conn, addr = listener.accept()
            _handle_connection(conn, lambda payload: {"echo": payload["value"]}, addr)
            listener.close()

        thread = threading.Thread(target=serve_once)
        thread.start()
        response, metrics = send_tensor(
            {"value": "x" * 300_000}, host, port, measure_upload=True
        )
        thread.join(timeout=2.0)
        self.assertEqual(response["echo"], "x" * 300_000)
        self.assertGreater(metrics["payload_bytes"], 256 * 1024)
        self.assertGreater(metrics["bw_mbps"], 0.0)


class DynamicCoordinatorTests(unittest.TestCase):
    def make_coordinator(self, users=2):
        service = FakeAlgoService()
        edge = {**owner("edge"), "BW_e2c": 50.0}
        cloud = {**owner("cloud"), "BW_e2c": 50.0}
        coordinator = RoundCoordinator(
            service,
            expected_users=users,
            node_state_provider=lambda: (edge, cloud),
            dynamic_bandwidth=True,
            bandwidth_change_threshold=0.2,
            bandwidth_debounce_s=3600.0,
            bandwidth_min_reschedule_interval_s=0.0,
        )
        return coordinator, service

    @staticmethod
    def finish_calibration(coordinator, round_id: str, users: int):
        for user_id in range(users):
            lease = coordinator.acquire_bandwidth_lease(round_id, user_id)
            assert lease["status"] == "granted"
            coordinator.report_device_bandwidth(
                round_id,
                user_id,
                {
                    "link": "d2e",
                    "source": "iperf",
                    "bw_mbps": 10.0 + user_id,
                    "filtered_bw_mbps": 10.0 + user_id,
                    "lease_token": lease["lease_token"],
                },
            )

    def test_leases_are_serial_and_initial_calibration_builds_v1(self):
        coordinator, service = self.make_coordinator(users=10)
        for user_id in range(10):
            coordinator.register("round", registration(user_id))
        first = coordinator.acquire_bandwidth_lease("round", 0)
        blocked = coordinator.acquire_bandwidth_lease("round", 1)
        self.assertEqual(first["status"], "granted")
        self.assertEqual(blocked["status"], "waiting")
        coordinator.report_device_bandwidth(
            "round",
            0,
            {
                "source": "iperf",
                "link": "d2e",
                "bw_mbps": 10.0,
                "lease_token": first["lease_token"],
            },
        )
        for user_id in range(1, 10):
            lease = coordinator.acquire_bandwidth_lease("round", user_id)
            coordinator.report_device_bandwidth(
                "round",
                user_id,
                {
                    "source": "iperf",
                    "link": "d2e",
                    "bw_mbps": 10.0 + user_id,
                    "lease_token": lease["lease_token"],
                },
            )
        self.assertEqual(coordinator.status("round")["decision_version"], 1)
        self.assertEqual(len(service.states), 1)

    def test_pending_decision_activates_atomically_at_barrier(self):
        coordinator, service = self.make_coordinator(users=2)
        coordinator.register("round", registration(0))
        coordinator.register("round", registration(1))
        self.finish_calibration(coordinator, "round", 2)
        coordinator.report_device_bandwidth(
            "round",
            0,
            {
                "source": "passive",
                "link": "d2e",
                "bw_mbps": 30.0,
                "filtered_bw_mbps": 30.0,
                "payload_bytes": 300_000,
                "decision_version": 1,
            },
        )
        coordinator._run_reoptimization("round")
        self.assertEqual(coordinator.status("round")["pending_decision_version"], 2)
        waiting = coordinator.ready_request("round", 0, 0)
        released = coordinator.ready_request("round", 1, 0)
        user_zero = coordinator.ready_request("round", 0, 0)
        self.assertIsNone(waiting["release_at"])
        self.assertEqual(released["decision_version"], 2)
        self.assertEqual(user_zero["decision"]["decision_version"], 2)
        self.assertEqual(coordinator.status("round")["decision_version"], 2)
        self.assertEqual(len(service.states), 2)

        for user_id in range(2):
            coordinator.submit_measurements(
                "round",
                user_id,
                {
                    "decision_id": "round:v2",
                    "decision_version": 2,
                    "measurements": [
                        {
                            "request_id": f"v1-user-{user_id}",
                            "decision_id": "round:v1",
                            "decision_version": 1,
                            "T_total": 1.0,
                            "is_correct": True,
                        },
                        {
                            "request_id": f"v2-user-{user_id}",
                            "decision_id": "round:v2",
                            "decision_version": 2,
                            "T_total": 0.5,
                            "is_correct": True,
                        },
                    ],
                },
            )
        self.assertEqual(coordinator.status("round")["status"], "completed")
        self.assertEqual([row["decision_id"] for row in service.reported], ["round:v1", "round:v2"])

    def test_fixed_mode_keeps_legacy_decision_id(self):
        service = FakeAlgoService()
        edge = {**owner("edge"), "BW_e2c": 50.0}
        cloud = {**owner("cloud"), "BW_e2c": 50.0}
        coordinator = RoundCoordinator(
            service,
            expected_users=1,
            node_state_provider=lambda: (edge, cloud),
            dynamic_bandwidth=False,
        )
        payload = registration(0, dynamic=False)
        coordinator.register("legacy-round", payload)
        decision = coordinator.decision_for_user("legacy-round", 0)
        self.assertEqual(decision["decision_id"], "legacy-round")
        self.assertEqual(decision["decision_version"], 1)

    def test_stale_e2c_is_calibrated_only_at_request_barrier(self):
        service = FakeAlgoService()
        edge = {**owner("edge"), "BW_e2c": 50.0}
        cloud = {**owner("cloud"), "BW_e2c": 50.0}
        clock = [100.0]
        calibrations = []

        def calibrate(duration_s):
            calibrations.append(duration_s)
            return {
                "link": "e2c",
                "source": "iperf",
                "bw_mbps": 80.0,
                "filtered_bw_mbps": 80.0,
            }

        coordinator = RoundCoordinator(
            service,
            expected_users=1,
            node_state_provider=lambda: (edge, cloud),
            edge_bandwidth_calibrator=calibrate,
            dynamic_bandwidth=True,
            bandwidth_stale_after_s=30.0,
            bandwidth_debounce_s=3600.0,
            clock=lambda: clock[0],
        )
        coordinator.register("round", registration(0))
        self.finish_calibration(coordinator, "round", 1)
        clock[0] = 200.0
        coordinator.report_device_bandwidth(
            "round",
            0,
            {
                "source": "passive",
                "link": "d2e",
                "bw_mbps": 10.0,
                "filtered_bw_mbps": 10.0,
                "payload_bytes": 300_000,
                "decision_version": 1,
            },
        )
        barrier = coordinator.ready_request("round", 0, 0)
        self.assertEqual(calibrations, [3.0])
        self.assertIsNotNone(barrier["release_at"])
        self.assertEqual(
            coordinator._round.edge_bandwidth_sample["filtered_bw_mbps"], 80.0
        )

    def test_expired_and_failed_leases_requeue_then_fall_back(self):
        service = FakeAlgoService()
        edge = {**owner("edge"), "BW_e2c": 50.0}
        cloud = {**owner("cloud"), "BW_e2c": 50.0}
        clock = [10.0]
        coordinator = RoundCoordinator(
            service,
            expected_users=1,
            node_state_provider=lambda: (edge, cloud),
            dynamic_bandwidth=True,
            iperf_calibration_timeout_s=8.0,
            clock=lambda: clock[0],
        )
        coordinator.register("round", registration(0))
        first = coordinator.acquire_bandwidth_lease("round", 0)
        clock[0] = 19.0
        second = coordinator.acquire_bandwidth_lease("round", 0)
        self.assertEqual(second["attempt"], 2)
        coordinator.report_device_bandwidth(
            "round",
            0,
            {
                "source": "iperf",
                "link": "d2e",
                "status": "failed",
                "lease_token": second["lease_token"],
            },
        )
        third = coordinator.acquire_bandwidth_lease("round", 0)
        self.assertEqual(third["attempt"], 3)
        coordinator.report_device_bandwidth(
            "round",
            0,
            {
                "source": "iperf",
                "link": "d2e",
                "status": "failed",
                "lease_token": third["lease_token"],
            },
        )
        self.assertEqual(coordinator.status("round")["decision_version"], 1)
        self.assertEqual(len(service.states), 1)


if __name__ == "__main__":
    unittest.main()
