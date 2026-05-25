"""Flask HTTP server for testbed deploy ↔ algorithm communication."""

from Src.Algorithm.Interface.algo_service import AlgoService, AlgoServiceConfig
from Src.Algorithm.Interface.decision_codec import DecisionCodecError
from Src.Algorithm.Interface.reward_adapter import RewardAdapterError
from Src.Algorithm.Interface.state_adapter import to_paras
from Src.Configs.model_config import get_model_config
from Src.Configs.testbed_config import DEFAULT as TESTBED_CFG

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Flask is required for the testbed API server. Install with: pip install flask"
    ) from exc


def create_app(service: AlgoService | None = None) -> Flask:
    """Create Flask app wired to a shared :class:`AlgoService`."""
    app = Flask(__name__)
    svc = service or AlgoService()

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
        except ValueError as exc:
            return _error(str(exc), 400)
        except DecisionCodecError as exc:
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

    return app


def _error(message: str, code: int):
    return jsonify({"status": "error", "message": message}), code


def _validate_state_payload(state: dict) -> None:
    """Fail fast on missing required fields before building Paras."""
    if "users" not in state or not isinstance(state["users"], list):
        raise KeyError("users")
    if len(state["users"]) == 0:
        raise KeyError("users must be non-empty")

    for key in ("edge", "cloud"):
        if key not in state or not isinstance(state[key], dict):
            raise KeyError(key)

    edge = state["edge"]
    cloud = state["cloud"]
    for key in ("f_e_max",):
        if key not in edge:
            raise KeyError(f"edge.{key}")
    for key in ("f_c_max", "BW_e2c"):
        if key not in cloud:
            raise KeyError(f"cloud.{key}")

    for i, user in enumerate(state["users"]):
        if not isinstance(user, dict):
            raise KeyError(f"users[{i}]")
        for key in ("f_u", "BW_d2e"):
            if key not in user:
                raise KeyError(f"users[{i}].{key}")

    model_name = state.get("model_name")
    if model_name is not None:
        get_model_config(model_name)

    to_paras(state)


def run_server(
    host: str = "0.0.0.0",
    port: int | None = None,
    service: AlgoService | None = None,
    debug: bool = False,
) -> None:
    """Blocking entrypoint for the testbed algorithm HTTP server."""
    app = create_app(service)
    listen_port = int(
        port if port is not None else TESTBED_CFG.algo_server_port)
    app.run(host=host, port=listen_port, debug=debug, threaded=True)


def build_service_from_env(
    *,
    checkpoint: str | None = None,
    enable_training: bool = False,
    auto_train: bool = True,
    deterministic: bool = True,
    buffer_size: int | None = None,
) -> AlgoService:
    cfg = AlgoServiceConfig(
        checkpoint_path=checkpoint,
        enable_training=enable_training,
        auto_train=auto_train,
        deterministic=deterministic,
    )
    if buffer_size is not None:
        cfg.buffer_size = int(buffer_size)
    return AlgoService(config=cfg)


if __name__ == '__main__':
    service = AlgoService()
    run_server(service=service, port=8000)
