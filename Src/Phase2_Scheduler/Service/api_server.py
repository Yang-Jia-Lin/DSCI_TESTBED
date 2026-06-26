"""Flask HTTP server for testbed deploy ↔ algorithm communication."""

import os

import requests

from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig
from Src.Phase2_Scheduler.Service.decision_codec import DecisionCodecError
from Src.Phase2_Scheduler.Service.reward_adapter import RewardAdapterError
from Src.Phase2_Scheduler.Service.round_coordinator import (
    RoundConflictError,
    RoundCoordinator,
    RoundCoordinatorError,
)
from Src.Phase2_Scheduler.Service.state_adapter import to_paras
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import require_bundle_id

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Flask is required for the testbed API server. Install with: pip install flask"
    ) from exc


def _node_state_provider() -> tuple[dict, dict]:
    edge_response = requests.get(
        f"http://{TESTBED_CFG.edge_host}:{TESTBED_CFG.edge_status_port}/status",
        timeout=10,
    )
    cloud_response = requests.get(
        f"http://{TESTBED_CFG.cloud_host}:{TESTBED_CFG.cloud_status_port}/status",
        timeout=10,
    )
    edge_response.raise_for_status()
    cloud_response.raise_for_status()
    edge = edge_response.json()
    cloud = cloud_response.json()
    return edge, {**cloud, "BW_e2c": edge["BW_e2c"]}


def create_app(
    service: AlgoService | None = None,
    coordinator: RoundCoordinator | None = None,
) -> Flask:
    """Create Flask app wired to a shared :class:`AlgoService`."""
    app = Flask(__name__)
    svc = service or AlgoService()
    rounds = coordinator or RoundCoordinator(
        svc,
        expected_users=int(os.environ.get("DSCI_EXPECTED_USERS", TESTBED_CFG.num_users)),
        node_state_provider=_node_state_provider,
    )

    @app.route("/api/v1/health", methods=["GET"])
    def health():
        return jsonify(svc.health())

    @app.route("/api/v1/decision", methods=["POST"])
    def get_decision():
        state = request.get_json(silent=True)
        if not isinstance(state, dict):
            return _error("Request body must be a JSON object", 400)

        try:
            _validate_state_payload(state)
            decision = svc.make_decision(state)
            return jsonify(decision)
        except KeyError as exc:
            return _error(f"Invalid state payload: {exc}", 400)
        except DecisionCodecError as exc:
            return _error(str(exc), 400)
        except ValueError as exc:
            return _error(str(exc), 400)
        except RuntimeError as exc:
            return _error(str(exc), 409)
        except Exception as exc:  # pragma: no cover
            return _error(f"Internal error: {exc}", 500)

    @app.route("/api/v1/measurements", methods=["POST"])
    def report_measurements():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("Request body must be a JSON object", 400)

        try:
            result = svc.report_measurements(payload)
            return jsonify(result)
        except RewardAdapterError as exc:
            return _error(str(exc), 400)
        except Exception as exc:  # pragma: no cover
            return _error(f"Internal error: {exc}", 500)

    @app.route("/api/v2/rounds/<round_id>/devices/register", methods=["POST"])
    def register_device(round_id: str):
        payload = request.get_json(silent=True)
        try:
            return jsonify(rounds.register(round_id, payload))
        except RoundCoordinatorError as exc:
            return _error(str(exc), 400)
        except RoundConflictError as exc:
            return _error(str(exc), 409)
        except Exception as exc:  # pragma: no cover
            return _error(f"Internal error: {exc}", 500)

    @app.route(
        "/api/v2/rounds/<round_id>/devices/<int:user_id>/heartbeat",
        methods=["POST"],
    )
    def heartbeat(round_id: str, user_id: int):
        try:
            return jsonify(rounds.heartbeat(round_id, user_id))
        except RoundCoordinatorError as exc:
            return _error(str(exc), 404)
        except RoundConflictError as exc:
            return _error(str(exc), 409)

    @app.route(
        "/api/v2/rounds/<round_id>/decisions/<int:user_id>", methods=["GET"]
    )
    def get_user_decision(round_id: str, user_id: int):
        try:
            decision = rounds.decision_for_user(round_id, user_id)
            if decision is None:
                return jsonify(rounds.status(round_id)), 202
            return jsonify(decision)
        except RoundCoordinatorError as exc:
            return _error(str(exc), 404)
        except RoundConflictError as exc:
            return _error(str(exc), 409)

    @app.route(
        "/api/v2/rounds/<round_id>/measurements/<int:user_id>", methods=["POST"]
    )
    def report_user_measurements(round_id: str, user_id: int):
        payload = request.get_json(silent=True)
        try:
            return jsonify(rounds.submit_measurements(round_id, user_id, payload))
        except RoundCoordinatorError as exc:
            return _error(str(exc), 400)
        except RoundConflictError as exc:
            return _error(str(exc), 409)

    @app.route("/api/v2/rounds/<round_id>/status", methods=["GET"])
    def round_status(round_id: str):
        try:
            return jsonify(rounds.status(round_id))
        except RoundCoordinatorError as exc:
            return _error(str(exc), 404)

    return app


def _error(message: str, code: int):
    return jsonify({"status": "error", "message": message}), code


def _validate_state_payload(state: dict) -> None:
    """Fail fast on missing required fields before building Paras."""
    require_bundle_id(state)
    if "users" not in state or not isinstance(state["users"], list):
        raise KeyError("users")
    if len(state["users"]) == 0:
        raise KeyError("users must be non-empty")

    for key in ("edge", "cloud"):
        if key not in state or not isinstance(state[key], dict):
            raise KeyError(key)

    edge = state["edge"]
    cloud = state["cloud"]
    resource_mode = str(
        state.get("resource_mode")
        or edge.get("resource_mode")
        or cloud.get("resource_mode")
        or "simulation_resource_mode"
    )
    if resource_mode == "fixed_worker_pool":
        all_owners = [*state["users"], edge, cloud]
        if any(owner.get("bundle_id") != state["bundle_id"] for owner in all_owners):
            raise KeyError("all nodes must report the selected bundle_id")
        if any(not owner.get("manifest_id") for owner in all_owners):
            raise KeyError("every node must report manifest_id")
        manifest_ids = {str(owner["manifest_id"]) for owner in all_owners}
        if len(manifest_ids) != 1:
            raise KeyError("all nodes must report the same manifest_id")
        if any(not owner.get("model_hash") for owner in all_owners):
            raise KeyError("every node must report model_hash")
        if len({str(owner["model_hash"]) for owner in all_owners}) != 1:
            raise KeyError("all nodes must report the same model_hash")
        owners = [
            *[(f"users[{i}]", user) for i, user in enumerate(state["users"])],
            ("edge", edge),
            ("cloud", cloud),
        ]
        for owner_name, owner in owners:
            for key in (
                "execution_profile_id",
                "backend",
                "worker_count",
                "threads_per_worker",
            ):
                if key not in owner:
                    raise KeyError(f"{owner_name}.{key}")
        if "BW_e2c" not in cloud:
            raise KeyError("cloud.BW_e2c")
        for i, user in enumerate(state["users"]):
            if "BW_d2e" not in user:
                raise KeyError(f"users[{i}].BW_d2e")
        to_paras(state)
        return
    for key in ("f_e_max",):
        if key not in edge:
            raise KeyError(f"edge.{key}")
    if "compute_profile_id" not in edge:
        raise KeyError("edge.compute_profile_id")
    for key in ("f_c_max", "BW_e2c"):
        if key not in cloud:
            raise KeyError(f"cloud.{key}")
    if "compute_profile_id" not in cloud:
        raise KeyError("cloud.compute_profile_id")

    for i, user in enumerate(state["users"]):
        if not isinstance(user, dict):
            raise KeyError(f"users[{i}]")
        for key in ("f_u", "BW_d2e"):
            if key not in user:
                raise KeyError(f"users[{i}].{key}")
        if "compute_profile_id" not in user:
            raise KeyError(f"users[{i}].compute_profile_id")

    to_paras(state)


def run_server(
    host: str = TESTBED_CFG.listen_host,
    port: int | None = None,
    service: AlgoService | None = None,
    coordinator: RoundCoordinator | None = None,
    debug: bool = False,
) -> None:
    """Blocking entrypoint for the testbed algorithm HTTP server."""
    app = create_app(service, coordinator)
    listen_port = int(port if port is not None else TESTBED_CFG.algo_server_port)
    app.run(host=host, port=listen_port, debug=debug, threaded=True)


def build_service_from_env(
    *,
    checkpoint: str | None = None,
    enable_training: bool = False,
    auto_train: bool = True,
    deterministic: bool = True,
    buffer_size: int | None = None,
    fixed_split: tuple[int, int] | None = None,
    fixed_threshold: float | None = None,
) -> AlgoService:
    cfg = AlgoServiceConfig(
        checkpoint_path=checkpoint,
        enable_training=enable_training,
        auto_train=auto_train,
        deterministic=deterministic,
        fixed_split=fixed_split,
        fixed_threshold=fixed_threshold,
    )
    if buffer_size is not None:
        cfg.buffer_size = int(buffer_size)
    return AlgoService(config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the testbed algorithm API server.")
    parser.add_argument("--host", default=TESTBED_CFG.listen_host)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--expected-users",
        type=int,
        default=int(os.environ.get("DSCI_EXPECTED_USERS", TESTBED_CFG.num_users)),
    )
    parser.add_argument(
        "--fixed-split",
        nargs=2,
        type=int,
        metavar=("S1", "S2"),
        help=(
            "Return this fixed partition_s1/partition_s2 for every decision "
            "request, for example: --fixed-split 0 1"
        ),
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        metavar="VALUE",
        help=(
            "Set every early-exit threshold in Y to this value for every "
            "decision request, for example: --fixed-threshold 0.7"
        ),
    )
    parser.add_argument("--no-auto-train", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    service = build_service_from_env(
        auto_train=not args.no_auto_train,
        fixed_split=tuple(args.fixed_split) if args.fixed_split else None,
        fixed_threshold=args.fixed_threshold,
    )
    coordinator = RoundCoordinator(
        service,
        expected_users=args.expected_users,
        node_state_provider=_node_state_provider,
    )
    run_server(
        host=args.host,
        port=args.port,
        service=service,
        coordinator=coordinator,
        debug=args.debug,
    )
