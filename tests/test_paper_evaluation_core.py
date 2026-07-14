import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Src.Phase2_Scheduler.Objective.evaluator import (
    EvaluationBudgetExceeded,
    ObjectiveEvaluator,
)
from Src.Phase2_Scheduler.Objective.compute_latency import compute_latency_breakdown
from Src.Phase2_Scheduler.Objective.shared_resources import (
    LinkJob,
    LinkResource,
    SharedResourceModel,
    WorkerJob,
    WorkerPoolResource,
)
from Src.Phase2_Scheduler.Optimizer.DSCI.agent import PPOAgent
from Src.Phase2_Scheduler.Service.decision_solver import DecisionSpec, solve_decision
from Src.Phase2_Scheduler.Service.reward_adapter import compute_round_reward
from Src.Phase2_Scheduler.Service.round_coordinator import (
    DeviceRegistration,
    RoundCoordinator,
    SchedulingRound,
)
from Src.Phase3_Runtime.Shared.request_trace import finalize_request_trace
from Src.Shared.Partitioning.split_actions import decode_split_row
from Src.Shared.Partitioning.split_actions import encode_split_row


def fake_paras(n=2):
    rates = np.zeros((101, 4), dtype=float)
    accs = np.zeros((101, 4), dtype=float)
    accs[:, 1] = 70.0
    accs[:, 2] = 75.0
    accs[:, 3] = 80.0
    return SimpleNamespace(
        n=n,
        m=4,
        E=[1, 2],
        exit_ids=["exit1", "exit2"],
        rates=rates,
        accs=accs,
        alpha=1.0,
        beta=5.0,
        resource_mode="fixed_worker_pool",
        partition_boundary_ids=[0, 1, 2, 3],
        f_e_max=1.0,
        f_c_max=1.0,
        bundle_id="fake-bundle",
        B_u=np.full(n, 10.0),
        segment_latency_u=np.ones((n, 4)),
        segment_latency_e=np.ones(4),
        segment_latency_c=np.ones(4),
        b_c=10.0,
        protocol_overhead_d2e_s=0.0,
        protocol_overhead_e2c_s=0.0,
        edge_worker_count=1,
        cloud_worker_count=1,
    )


class SharedResourceTests(unittest.TestCase):
    @staticmethod
    def fixed_worker_paras(n, shared):
        manifest = SimpleNamespace(
            segment_ids=[0, 1, 2],
            early_exits=[],
            final_boundary_id=3,
            validate_boundary_pair=lambda _first, _second: None,
        )
        return SimpleNamespace(
            n=n,
            m=4,
            E=[],
            partition_manifest=manifest,
            boundary_bytes=[100, 100, 100, 100],
            transport_byte_scale=1.0,
            segment_latency_u=np.full((n, 3), 0.1),
            segment_latency_e=np.full(3, 0.2),
            segment_latency_c=np.full(3, 0.3),
            exit_head_latency_u=[{} for _ in range(n)],
            exit_head_latency_e={},
            exit_head_latency_c={},
            B_u=np.full(n, 10.0),
            b_c=10.0,
            protocol_overhead_d2e_s=0.0,
            protocol_overhead_e2c_s=0.0,
            resource_mode="fixed_worker_pool",
            shared_resource_model=shared,
            d2e_link_ids=[f"u{i}" for i in range(n)],
            d2e_link_capacities_mbps={f"u{i}": 10.0 for i in range(n)},
            e2c_link_id="wan",
            e2c_link_capacity_mbps=10.0,
            edge_worker_count=1,
            cloud_worker_count=1,
        )

    def test_link_processor_sharing(self):
        jobs = [LinkJob(0.0, 10.0, 10.0), LinkJob(0.0, 10.0, 10.0)]
        np.testing.assert_allclose(LinkResource(10.0).schedule(jobs), [2.0, 2.0])

    def test_worker_queue(self):
        starts, completion = WorkerPoolResource(1).schedule(
            [WorkerJob(0.0, 1.0), WorkerJob(0.0, 1.0)]
        )
        np.testing.assert_allclose(starts, [0.0, 1.0])
        np.testing.assert_allclose(completion, [1.0, 2.0])

    def test_single_user_degrades_to_component_sum(self):
        result = SharedResourceModel().evaluate(
            device_compute=np.array([0.2]),
            d2e_work_megabits=np.array([5.0]),
            d2e_max_rates_mbps=np.array([10.0]),
            d2e_link_ids=["wifi"],
            d2e_capacities_mbps={"wifi": 10.0},
            d2e_overhead_s=np.array([0.01]),
            edge_service_s=np.array([0.3]),
            edge_worker_count=1,
            e2c_work_megabits=np.array([2.0]),
            e2c_max_rates_mbps=np.array([10.0]),
            e2c_link_ids=["wan"],
            e2c_capacities_mbps={"wan": 10.0},
            e2c_overhead_s=np.array([0.02]),
            cloud_service_s=np.array([0.4]),
            cloud_worker_count=1,
        )
        self.assertAlmostEqual(float(result.edge_queue[0]), 0.0)
        self.assertAlmostEqual(float(result.cloud_queue[0]), 0.0)
        self.assertAlmostEqual(float(result.total[0]), 1.63)

    def test_more_jobs_than_workers_create_queue(self):
        n = 3
        result = SharedResourceModel().evaluate(
            device_compute=np.zeros(n),
            d2e_work_megabits=np.zeros(n),
            d2e_max_rates_mbps=np.ones(n),
            d2e_link_ids=[f"d{i}" for i in range(n)],
            d2e_capacities_mbps={f"d{i}": 1.0 for i in range(n)},
            d2e_overhead_s=np.zeros(n),
            edge_service_s=np.ones(n),
            edge_worker_count=1,
            e2c_work_megabits=np.zeros(n),
            e2c_max_rates_mbps=np.ones(n),
            e2c_link_ids=["wan"] * n,
            e2c_capacities_mbps={"wan": 1.0},
            e2c_overhead_s=np.zeros(n),
            cloud_service_s=np.zeros(n),
            cloud_worker_count=1,
        )
        self.assertGreater(float(result.edge_queue[1]), 0.0)
        self.assertGreater(float(result.edge_queue[2]), float(result.edge_queue[1]))

    def test_non_arriving_request_does_not_wait_in_remote_queues(self):
        result = SharedResourceModel().evaluate(
            device_compute=np.array([0.2, 0.0]),
            d2e_work_megabits=np.array([0.0, 1.0]),
            d2e_max_rates_mbps=np.ones(2),
            d2e_link_ids=["local", "remote"],
            d2e_capacities_mbps={"local": 1.0, "remote": 1.0},
            d2e_overhead_s=np.zeros(2),
            edge_service_s=np.array([0.0, 1.0]),
            edge_worker_count=1,
            e2c_work_megabits=np.zeros(2),
            e2c_max_rates_mbps=np.ones(2),
            e2c_link_ids=["wan", "wan"],
            e2c_capacities_mbps={"wan": 1.0},
            e2c_overhead_s=np.zeros(2),
            cloud_service_s=np.zeros(2),
            cloud_worker_count=1,
        )
        self.assertAlmostEqual(float(result.total[0]), 0.2)
        self.assertAlmostEqual(float(result.edge_queue[0]), 0.0)
        self.assertAlmostEqual(float(result.cloud_queue[0]), 0.0)

    def test_compute_latency_wiring_preserves_n1_and_couples_n2(self):
        X1 = np.stack([encode_split_row(1, 3, 4)])
        P1 = np.zeros((1, 4))
        P1[:, 3] = 1.0
        legacy = compute_latency_breakdown(
            X1, P1, self.fixed_worker_paras(1, False)
        )
        shared = compute_latency_breakdown(
            X1, P1, self.fixed_worker_paras(1, True)
        )
        np.testing.assert_allclose(shared["total"], legacy["total"])

        X2 = np.vstack([X1, X1])
        P2 = np.zeros((2, 4))
        P2[:, 3] = 1.0
        coupled = compute_latency_breakdown(
            X2, P2, self.fixed_worker_paras(2, True)
        )
        self.assertGreater(float(coupled["edge_queue"][1]), 0.0)


class DecisionContractTests(unittest.TestCase):
    def test_static_requires_fixed_variables(self):
        with self.assertRaises(ValueError):
            DecisionSpec(optimizer="static").validate()

    def test_ppo_action_obeys_fixed_split_and_disabled_exit(self):
        paras = fake_paras(2)
        spec = DecisionSpec(
            split_rule="fixed",
            fixed_split=((3, 3), (0, 3)),
            exit_rule="disabled",
            evaluation_budget=10,
        )
        params = {
            "lr": 1e-4,
            "entropy_coef": 0.01,
            "entropy_decay": 0.99,
        }
        agent = PPOAgent(paras, params, decision_spec=spec)
        X = np.zeros((2, 4), dtype=np.float32)
        Y = np.zeros((2, 4), dtype=np.float32)
        X, Y = agent._apply_action_to_XY(X, Y, 1, 0, np.array([]))
        self.assertEqual(decode_split_row(X[1]), (0, 3))
        np.testing.assert_allclose(Y[1], np.ones(4))

    def test_objective_budget_and_normalized_trace(self):
        paras = fake_paras(1)
        components = {
            "device_compute": np.array([0.1]),
            "d2e_transfer": np.array([0.0]),
            "edge_queue": np.array([0.0]),
            "edge_compute": np.array([0.0]),
            "e2c_transfer": np.array([0.0]),
            "cloud_queue": np.array([0.0]),
            "cloud_compute": np.array([0.0]),
            "total": np.array([0.1]),
        }
        evaluator = ObjectiveEvaluator(paras, evaluation_budget=1)
        X = np.array([[0, 0, 0, 1]], dtype=np.float32)
        Y = np.ones((1, 4), dtype=np.float32)
        F = np.zeros((1, 1), dtype=np.float32)
        with patch(
            "Src.Phase2_Scheduler.Objective.evaluator.compute_latency_breakdown",
            return_value=components,
        ):
            value = evaluator.evaluate(X, Y, F, F)
            row = evaluator.record("random", 0, value, value)
            self.assertEqual(row["objective_evaluations"], 1)
            self.assertAlmostEqual(row["expected_latency"], 0.1)
            with self.assertRaises(EvaluationBudgetExceeded):
                evaluator.evaluate(X, Y, F, F)

    def test_synchronous_static_solver_never_returns_default(self):
        state = {
            "round_id": "r1",
            "bundle_id": "fake-bundle",
            "resource_mode": "fixed_worker_pool",
            "users": [{"user_id": 7, "BW_d2e": 10.0}],
            "edge": {},
            "cloud": {"BW_e2c": 10.0},
        }
        spec = DecisionSpec(
            optimizer="static",
            split_rule="fixed",
            fixed_split=(3, 3),
            exit_rule="disabled",
            evaluation_budget=1,
        )
        components = {
            "device_compute": np.array([0.1]),
            "d2e_transfer": np.array([0.0]),
            "edge_queue": np.array([0.0]),
            "edge_compute": np.array([0.0]),
            "e2c_transfer": np.array([0.0]),
            "cloud_queue": np.array([0.0]),
            "cloud_compute": np.array([0.0]),
            "total": np.array([0.1]),
        }
        with patch(
            "Src.Phase2_Scheduler.Service.decision_solver.Paras.from_state",
            return_value=fake_paras(1),
        ), patch(
            "Src.Phase2_Scheduler.Service.decision_solver.encode",
            return_value={"users": [{"user_id": 7}]},
        ), patch(
            "Src.Phase2_Scheduler.Objective.evaluator.compute_latency_breakdown",
            return_value=components,
        ):
            result = solve_decision(state, spec)
        self.assertEqual(result.decision["decision_source"], "synchronous:static:joint")
        self.assertEqual(result.objective_evaluations, 1)
        self.assertEqual(decode_split_row(result.X[0]), (3, 3))
        self.assertEqual(len(result.state_signature["sha256"]), 64)

    def test_independent_matches_separate_single_user_solves(self):
        state = {
            "round_id": "r2",
            "bundle_id": "fake-bundle",
            "resource_mode": "fixed_worker_pool",
            "users": [
                {"user_id": 4, "BW_d2e": 10.0},
                {"user_id": 9, "BW_d2e": 10.0},
            ],
            "edge": {},
            "cloud": {"BW_e2c": 10.0},
        }
        spec = DecisionSpec(
            coordination="independent",
            optimizer="random",
            exit_rule="disabled",
            allowed_split_pairs=((3, 3), (0, 3), (1, 3)),
            evaluation_budget=2,
            seed=11,
        )

        def paras_for_state(value):
            return fake_paras(len(value["users"]))

        def latency_for(X, _P, _paras, *_resources):
            n = len(X)
            zeros = np.zeros(n)
            return {
                "device_compute": np.full(n, 0.1),
                "d2e_transfer": zeros.copy(),
                "edge_queue": zeros.copy(),
                "edge_compute": zeros.copy(),
                "e2c_transfer": zeros.copy(),
                "cloud_queue": zeros.copy(),
                "cloud_compute": zeros.copy(),
                "total": np.full(n, 0.1),
            }

        def fake_encode(_X, _Y, _Fe, _Fc, _paras, **kwargs):
            return {"users": [{"user_id": value} for value in kwargs["user_ids"]]}

        with patch(
            "Src.Phase2_Scheduler.Service.decision_solver.Paras.from_state",
            side_effect=paras_for_state,
        ), patch(
            "Src.Phase2_Scheduler.Service.decision_solver.encode",
            side_effect=fake_encode,
        ), patch(
            "Src.Phase2_Scheduler.Objective.evaluator.compute_latency_breakdown",
            side_effect=latency_for,
        ):
            combined = solve_decision(state, spec)
            singles = []
            for index, user in enumerate(state["users"]):
                child = {**state, "users": [user]}
                singles.append(solve_decision(child, spec.for_user(index, 2)))
        expected_X = np.concatenate([item.X for item in singles], axis=0)
        np.testing.assert_allclose(combined.X, expected_X)
        self.assertTrue(all(decode_split_row(row)[1] == 3 for row in combined.X))
        self.assertEqual(combined.objective_evaluations, 5)
        self.assertEqual(
            combined.optimizer_trace[-1]["stage"], "merged_shared_evaluation"
        )


class RuntimeProtocolTests(unittest.TestCase):
    def test_request_barrier_releases_every_user_at_same_time(self):
        coordinator = RoundCoordinator(
            service=SimpleNamespace(),
            expected_users=2,
            node_state_provider=lambda: ({}, {}),
            request_release_delay_s=0.0,
            clock=lambda: 100.0,
        )
        coordinator._round = SchedulingRound(
            round_id="r",
            expected_users=2,
            created_at=100.0,
            status="decision_ready",
            registered_devices={
                0: DeviceRegistration(0, {}, 100.0),
                1: DeviceRegistration(1, {}, 100.0),
            },
        )
        first = coordinator.ready_request("r", 0, 0)
        second = coordinator.ready_request("r", 1, 0)
        again = coordinator.ready_request("r", 0, 0)
        self.assertIsNone(first["release_at"])
        self.assertEqual(second["release_at"], again["release_at"])
        self.assertEqual(second["ready_users"], [0, 1])
        self.assertEqual(second["ready_at"], {"0": 100.0, "1": 100.0})

    def test_request_trace_closes_and_preserves_route(self):
        result = {
            "request_id": "q",
            "user_id": 1,
            "prediction": 2,
            "exit_id": "final",
            "exit_boundary_id": 4,
            "exit_location": "cloud",
            "confidence": 0.9,
            "T_compute_device": 0.1,
            "T_device_edge_roundtrip": 1.0,
            "T_node_edge": 0.2,
            "T_edge_cloud_roundtrip": 0.5,
            "T_node_cloud": 0.2,
            "T_total": 1.1,
            "node_trace": [
                {"node": "device", "executed_segments": [0]},
                {
                    "node": "edge",
                    "executed_segments": [1],
                    "queue_s": 0.05,
                    "segment_compute_s": 0.1,
                    "exit_head_compute_s": 0.01,
                    "exit_check_s": 0.01,
                },
                {
                    "node": "cloud",
                    "executed_segments": [2, 3],
                    "queue_s": 0.02,
                    "segment_compute_s": 0.15,
                    "exit_head_compute_s": 0.01,
                    "exit_check_s": 0.01,
                },
            ],
        }
        trace = finalize_request_trace(result)
        known = sum(
            trace[key]
            for key in (
                "device_compute",
                "d2e_transport",
                "edge_queue",
                "edge_segment_compute",
                "edge_exit_head_compute",
                "edge_exit_check",
                "e2c_transport",
                "cloud_queue",
                "cloud_segment_compute",
                "cloud_exit_head_compute",
                "cloud_exit_check",
                "unattributed_overhead",
            )
        )
        self.assertAlmostEqual(known, trace["total_latency"])
        self.assertEqual(trace["executed_segments_by_node"]["cloud"], [2, 3])

    def test_reward_reports_sum_and_mean(self):
        result = compute_round_reward(
            {
                "decision_id": "d",
                "measurements": [
                    {"user_id": 0, "is_correct": 1, "T_total": 0.1},
                    {"user_id": 1, "is_correct": 0, "T_total": 0.2},
                ],
            },
            alpha=1.0,
            beta=1.0,
        )
        self.assertAlmostEqual(result.utility_sum, 0.7)
        self.assertAlmostEqual(result.utility_mean, 0.35)
        self.assertNotIn("round_reward", result.to_dict())


if __name__ == "__main__":
    unittest.main()
