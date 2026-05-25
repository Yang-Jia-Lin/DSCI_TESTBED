"""Interface adapters between testbed JSON payloads and algorithm objects."""

__all__ = [
    "AlgoService",
    "AlgoServiceConfig",
    "compute_round_reward",
    "create_app",
    "encode",
    "infer_one_round",
    "run_server",
    "to_paras",
    "validate_decision",
]

_LAZY_EXPORTS = {
    "AlgoService": ("Src.Algorithm.Interface.algo_service", "AlgoService"),
    "AlgoServiceConfig": ("Src.Algorithm.Interface.algo_service", "AlgoServiceConfig"),
    "compute_round_reward": (
        "Src.Algorithm.Interface.reward_adapter",
        "compute_round_reward",
    ),
    "create_app": ("Src.Algorithm.Interface.api_server", "create_app"),
    "encode": ("Src.Algorithm.Interface.decision_codec", "encode"),
    "infer_one_round": ("Src.Algorithm.Optimizer.DSCI.run_DSCI", "infer_one_round"),
    "run_server": ("Src.Algorithm.Interface.api_server", "run_server"),
    "to_paras": ("Src.Algorithm.Interface.state_adapter", "to_paras"),
    "validate_decision": ("Src.Algorithm.Interface.decision_codec", "validate_decision"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
