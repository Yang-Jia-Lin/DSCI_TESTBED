"""Run controlled PPO convergence, cache-update, and immediate-quality experiments.

The S0 cold run for each seed is shared by E1 (convergence) and E2 (startup
mode overhead). E3 compares the profile-based immediate and final decisions.
No real inference requests are issued by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from Scripts.EvaluationCommon.solutions import state_from_solution_meta
from Src.Phase2_Scheduler.Service.algo_service import (
    AlgoService,
    AlgoServiceConfig,
)


DEFAULT_STATE_META = (
    REPO_ROOT
    / "Data"
    / "Runtime"
    / "SolutionCache"
    / "solution_20260722175910020_meta.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Scripts"
    / "Exp4_System-Overhead"
    / "result_data"
)
MODE_SPECS = (
    ("S0", "cold", None),
    ("S1", "medium", 0.10),
    ("S2", "near", 0.025),
    ("S3", "reuse", 0.0025),
)
UTILITY_EPSILON = 1e-9


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _machine_manifest() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
    }


def _experiment_id(smoke: bool, timestamp: str) -> str:
    prefix = "smoke" if smoke else "controlled"
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _load_canonical_state(meta_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = _load_json(meta_path)
    state = state_from_solution_meta(meta)
    state["shared_resource_model"] = len(state["users"]) > 1
    if state.get("bundle_id") != "resnet50-cifar10":
        raise ValueError("Canonical experiment requires bundle resnet50-cifar10")
    if len(state.get("users") or []) != 4:
        raise ValueError("Canonical experiment requires exactly four users")
    if state.get("resource_mode") != "fixed_worker_pool":
        raise ValueError("Canonical experiment requires fixed_worker_pool")
    return state, meta


def _with_bandwidth_multiplier(state: dict[str, Any], multiplier: float) -> dict[str, Any]:
    candidate = copy.deepcopy(state)
    for user in candidate["users"]:
        user["BW_d2e"] = float(user["BW_d2e"]) * float(multiplier)
    candidate["cloud"]["BW_e2c"] = (
        float(candidate["cloud"]["BW_e2c"]) * float(multiplier)
    )
    if candidate.get("edge", {}).get("BW_e2c") is not None:
        candidate["edge"]["BW_e2c"] = (
            float(candidate["edge"]["BW_e2c"]) * float(multiplier)
        )
    return candidate


def _signature_for(service: AlgoService, state: dict[str, Any]) -> dict[str, Any]:
    paras = service._paras_for_state(state)
    return service._state_signature(state, paras)


def _distance_between(
    service: AlgoService,
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    left_vector = service._state_vector(_signature_for(service, left))
    right_vector = service._state_vector(_signature_for(service, right))
    return service._state_distance(left_vector, right_vector)


def _state_at_distance(
    service: AlgoService,
    previous: dict[str, Any],
    target: float,
) -> tuple[dict[str, Any], float]:
    low, high = 1.0, 2.0
    while _distance_between(
        service, previous, _with_bandwidth_multiplier(previous, high)
    ) < target:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError(f"Cannot reach state distance {target}")
    for _ in range(80):
        middle = (low + high) / 2.0
        candidate = _with_bandwidth_multiplier(previous, middle)
        if _distance_between(service, previous, candidate) < target:
            low = middle
        else:
            high = middle
    candidate = _with_bandwidth_multiplier(previous, high)
    return candidate, _distance_between(service, previous, candidate)


def build_state_sequence(
    service: AlgoService,
    canonical_state: dict[str, Any],
) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    previous = copy.deepcopy(canonical_state)
    base_compat = service._compat_key(_signature_for(service, previous))
    for state_id, planned_mode, target_distance in MODE_SPECS:
        if target_distance is None:
            state = copy.deepcopy(previous)
            actual_distance = None
        else:
            state, actual_distance = _state_at_distance(
                service, previous, target_distance
            )
            actual_mode = service._training_mode_for_distance(actual_distance)
            if actual_mode != planned_mode:
                raise AssertionError(
                    f"{state_id}: target {target_distance} produced "
                    f"{actual_distance} ({actual_mode}, expected {planned_mode})"
                )
        compat = service._compat_key(_signature_for(service, state))
        if compat != base_compat:
            raise AssertionError(f"{state_id}: compatibility key changed")
        sequence.append(
            {
                "state_id": state_id,
                "planned_mode": planned_mode,
                "target_distance": target_distance,
                "step_distance": actual_distance,
                "state": state,
            }
        )
        previous = state
    return sequence


def _count_epochs(path: str | Path | None) -> int:
    if not path:
        return 0
    convergence_path = Path(path)
    if not convergence_path.exists():
        return 0
    count = 0
    with convergence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                count += json.loads(line).get("event") == "ppo_epoch"
            except json.JSONDecodeError:
                continue
    return int(count)


def _wait_for_training(
    service: AlgoService,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while True:
        health = service.health()
        if health["training_status"] in {"idle", "error"}:
            return health
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Background training exceeded {timeout_s:.1f} seconds"
            )
        time.sleep(poll_interval_s)


def _relative(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return str(value.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(value)


def _quality_fields(
    immediate: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, float | None]:
    immediate_utility = float(immediate["objective"])
    final_utility = float(final["objective"])
    immediate_accuracy = float(immediate["expected_accuracy"])
    final_accuracy = float(final["expected_accuracy"])
    immediate_latency = float(immediate["expected_latency"])
    final_latency = float(final["expected_latency"])
    retention = (
        immediate_utility / final_utility
        if (
            immediate_utility > UTILITY_EPSILON
            and final_utility > UTILITY_EPSILON
        )
        else None
    )
    return {
        "immediate_objective": immediate_utility,
        "immediate_expected_accuracy": immediate_accuracy,
        "immediate_expected_latency_s": immediate_latency,
        "final_objective": final_utility,
        "final_expected_accuracy": final_accuracy,
        "final_expected_latency_s": final_latency,
        "objective_gap": final_utility - immediate_utility,
        "accuracy_gap_pp": 100.0 * (final_accuracy - immediate_accuracy),
        "latency_gap_ms": 1000.0 * (immediate_latency - final_latency),
        "utility_immediate": immediate_utility,
        "utility_final": final_utility,
        "utility_gap": final_utility - immediate_utility,
        "utility_retention": retention,
    }


def _mode_record(
    *,
    experiment_id: str,
    seed: int,
    spec: dict[str, Any],
    actual_distance: float | None,
    immediate: dict[str, Any],
    final: dict[str, Any],
    foreground_service_ms: float,
    final_lookup_ms: float,
    background_update_s: float,
    background_epochs: int,
    actual_training_mode: str,
    health: dict[str, Any],
    convergence_path: str | Path | None,
    immediate_path: Path,
    final_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}:seed-{seed}:{spec['state_id']}",
        "seed": seed,
        "state_id": spec["state_id"],
        "bundle_id": immediate["bundle_id"],
        "n_users": len(spec["state"]["users"]),
        "objective_alpha": float(immediate["objective_alpha"]),
        "objective_beta": float(immediate["objective_beta"]),
        "state_distance": actual_distance,
        "target_state_distance": spec["target_distance"],
        "planned_mode": spec["planned_mode"],
        "actual_training_mode": actual_training_mode,
        "decision_source": immediate["decision_source"],
        "final_decision_source": final["decision_source"],
        "has_compatible_cache": spec["planned_mode"] != "cold",
        "has_policy_source": bool(health.get("last_warm_start_source")),
        "foreground_service_ms": foreground_service_ms,
        "foreground_http_ms": None,
        "final_lookup_ms": final_lookup_ms,
        "background_update_s": background_update_s,
        "background_epochs": background_epochs,
        **_quality_fields(immediate, final),
        "training_status": health["training_status"],
        "training_error": health.get("last_error"),
        "convergence_path": _relative(convergence_path, experiment_root),
        "immediate_decision_path": _relative(immediate_path, experiment_root),
        "final_decision_path": _relative(final_path, experiment_root),
    }


def _run_mode(
    *,
    service: AlgoService,
    experiment_id: str,
    experiment_root: Path,
    seed: int,
    spec: dict[str, Any],
    previous_state: dict[str, Any] | None,
    events_path: Path,
    mode_dir: Path,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    state = copy.deepcopy(spec["state"])
    state["round_id"] = f"{experiment_id}:seed-{seed}:{spec['state_id']}"
    actual_distance = (
        None
        if previous_state is None
        else _distance_between(service, previous_state, state)
    )
    if actual_distance is not None:
        actual_mode = service._training_mode_for_distance(actual_distance)
        if actual_mode != spec["planned_mode"]:
            raise AssertionError(
                f"{spec['state_id']}: actual mode {actual_mode}, "
                f"expected {spec['planned_mode']}"
            )

    _append_jsonl(
        events_path,
        {
            "event": "mode_start",
            "experiment_id": experiment_id,
            "seed": seed,
            "state_id": spec["state_id"],
            "planned_mode": spec["planned_mode"],
            "state_distance": actual_distance,
        },
    )
    started = time.perf_counter()
    immediate = service.make_decision(state)
    foreground_service_ms = (time.perf_counter() - started) * 1000.0
    immediate_health = service.health()
    immediate_path = mode_dir / "immediate_decision.json"
    _write_json_once(immediate_path, immediate)
    _append_jsonl(
        events_path,
        {
            "event": "immediate_decision",
            "experiment_id": experiment_id,
            "seed": seed,
            "state_id": spec["state_id"],
            "planned_mode": spec["planned_mode"],
            "state_distance": actual_distance,
            "foreground_service_ms": foreground_service_ms,
            "decision": immediate,
            "health": immediate_health,
        },
    )

    if spec["planned_mode"] == "reuse":
        if immediate_health["training_status"] == "running":
            raise AssertionError("reuse unexpectedly started background training")
        final = copy.deepcopy(immediate)
        final_lookup_ms = 0.0
        terminal_health = immediate_health
        background_update_s = 0.0
        background_epochs = 0
        convergence_path = None
        actual_training_mode = "reuse"
    else:
        terminal_health = _wait_for_training(
            service,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        if terminal_health["training_status"] != "idle":
            raise RuntimeError(
                terminal_health.get("last_error") or "Background training failed"
            )
        convergence_path = terminal_health.get("last_training_convergence_path")
        background_update_s = float(
            terminal_health.get("last_training_duration_s") or 0.0
        )
        background_epochs = _count_epochs(convergence_path)
        actual_training_mode = str(terminal_health.get("last_training_mode"))

        lookup_started = time.perf_counter()
        final = service.make_decision(state)
        final_lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
        exact_health = service.health()
        if exact_health["training_status"] == "running":
            raise AssertionError("Exact post-training lookup restarted PPO")
        if final.get("decision_source") != "cached_dsci:exact":
            raise AssertionError(
                "Post-training lookup did not return the exact cached solution: "
                f"{final.get('decision_source')}"
            )
        terminal_health = exact_health

    final_path = mode_dir / "final_decision.json"
    _write_json_once(final_path, final)
    record = _mode_record(
        experiment_id=experiment_id,
        seed=seed,
        spec=spec,
        actual_distance=actual_distance,
        immediate=immediate,
        final=final,
        foreground_service_ms=foreground_service_ms,
        final_lookup_ms=final_lookup_ms,
        background_update_s=background_update_s,
        background_epochs=background_epochs,
        actual_training_mode=actual_training_mode,
        health=terminal_health,
        convergence_path=convergence_path,
        immediate_path=immediate_path,
        final_path=final_path,
        experiment_root=experiment_root,
    )
    _write_json_once(mode_dir / "record.json", record)
    _append_jsonl(
        events_path,
        {
            "event": "mode_complete",
            **record,
        },
    )
    return record


def _smoke_hyperparams() -> dict[str, Any]:
    return {
        "max_epochs": 2,
        "min_epochs": 0,
        "target_steps": 8,
        "k_epochs": 1,
        "patience": 1,
        "outer_ema": 1.0,
    }


def _service_for_seed(
    seed_dir: Path,
    *,
    alpha: float,
    beta: float,
    smoke: bool,
) -> AlgoService:
    cache_dir = seed_dir / "cache"
    config = AlgoServiceConfig(
        objective_alpha=alpha,
        objective_beta=beta,
        latest_solution_path=cache_dir / "latest_solution.npz",
        latest_meta_path=cache_dir / "latest_solution_meta.json",
        training_events_path=cache_dir / "training_events.jsonl",
        training_convergence_dir=seed_dir / "convergence",
        custom_ppo_hyperparams=_smoke_hyperparams() if smoke else None,
        auto_train=True,
        max_cached_solutions=10,
    )
    return AlgoService(config=config)


def _effective_hyperparameters(
    experiment_root: Path,
    *,
    alpha: float,
    beta: float,
    smoke: bool,
) -> dict[str, dict[str, Any] | None]:
    probe = _service_for_seed(
        experiment_root / "_manifest_probe",
        alpha=alpha,
        beta=beta,
        smoke=smoke,
    )
    return {
        "cold": probe._training_params(None),
        "medium": probe._training_params(
            SimpleNamespace(training_mode="medium")
        ),
        "near": probe._training_params(SimpleNamespace(training_mode="near")),
        "reuse": None,
    }


def _prepare_experiment(
    args: argparse.Namespace,
    canonical_state: dict[str, Any],
    source_meta: dict[str, Any],
) -> tuple[str, Path]:
    run_timestamp = (
        args.run_timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    experiment_id = _experiment_id(args.smoke, run_timestamp)
    experiment_root = Path(args.output_dir).resolve()
    manifest_path = experiment_root / f"manifest_{run_timestamp}.json"
    manifest = {
        "experiment_id": experiment_id,
        "run_timestamp": run_timestamp,
        "created_at": time.time(),
        "smoke": bool(args.smoke),
        "state_meta_path": str(Path(args.state_meta).resolve()),
        "bundle_id": canonical_state["bundle_id"],
        "manifest_id": source_meta["state_signature"]["manifest_id"],
        "model_hash": source_meta["state_signature"]["model_hash"],
        "n_users": len(canonical_state["users"]),
        "objective_alpha": float(args.objective_alpha),
        "objective_beta": float(args.objective_beta),
        "resource_mode": canonical_state["resource_mode"],
        "seeds": list(args.seeds),
        "target_distances": {
            mode: distance for _, mode, distance in MODE_SPECS
        },
        "ppo_hyperparameters": _effective_hyperparameters(
            experiment_root / f"_manifest_probe_{run_timestamp}",
            alpha=float(args.objective_alpha),
            beta=float(args.objective_beta),
            smoke=bool(args.smoke),
        ),
        "canonical_state": canonical_state,
        "hardware": _machine_manifest(),
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists for timestamp {run_timestamp}: "
            f"{manifest_path}"
        )
    _write_json_once(manifest_path, manifest)
    return experiment_id, experiment_root


def run_experiment(args: argparse.Namespace) -> Path:
    canonical_state, source_meta = _load_canonical_state(Path(args.state_meta))
    experiment_id, experiment_root = _prepare_experiment(
        args, canonical_state, source_meta
    )
    print(f"Result directory: {experiment_root}")

    for seed in args.seeds:
        seed = int(seed)
        seed_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed_dir = experiment_root / f"seed_{seed}_{seed_timestamp}"
        status_path = seed_dir / "status.json"
        seed_dir.mkdir(parents=True, exist_ok=False)
        events_path = seed_dir / "run_events.jsonl"
        _seed_everything(seed)
        service = _service_for_seed(
            seed_dir,
            alpha=float(args.objective_alpha),
            beta=float(args.objective_beta),
            smoke=bool(args.smoke),
        )
        sequence = build_state_sequence(service, canonical_state)
        _write_json_once(
            seed_dir / "states.json",
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "states": sequence,
            },
        )
        _append_jsonl(
            events_path,
            {
                "event": "seed_start",
                "experiment_id": experiment_id,
                "seed": seed,
                "seed_timestamp": seed_timestamp,
            },
        )
        records: list[dict[str, Any]] = []
        try:
            previous_state = None
            for spec in sequence:
                mode_dir = (
                    seed_dir
                    / "modes"
                    / f"{spec['state_id']}_{spec['planned_mode']}"
                )
                mode_dir.mkdir(parents=True, exist_ok=False)
                record = _run_mode(
                    service=service,
                    experiment_id=experiment_id,
                    experiment_root=experiment_root,
                    seed=seed,
                    spec=spec,
                    previous_state=previous_state,
                    events_path=events_path,
                    mode_dir=mode_dir,
                    timeout_s=float(args.training_timeout_s),
                    poll_interval_s=float(args.poll_interval_s),
                )
                records.append(record)
                previous_state = spec["state"]
                print(
                    f"[seed {seed}] {spec['state_id']} "
                    f"{record['actual_training_mode']}: "
                    f"response={record['foreground_service_ms']:.3f} ms, "
                    f"update={record['background_update_s']:.3f} s"
                )
            _write_json_once(
                status_path,
                {
                    "status": "complete",
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "completed_at": time.time(),
                    "modes": [row["actual_training_mode"] for row in records],
                },
            )
            _append_jsonl(
                events_path,
                {
                    "event": "seed_complete",
                    "experiment_id": experiment_id,
                    "seed": seed,
                },
            )
        except Exception as exc:
            _write_json_once(
                status_path,
                {
                    "status": "failed",
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "failed_at": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            _append_jsonl(
                events_path,
                {
                    "event": "seed_error",
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
    return experiment_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-meta", type=Path, default=DEFAULT_STATE_META)
    parser.add_argument("--objective-alpha", type=float, default=1.0)
    parser.add_argument("--objective-beta", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-timestamp",
        "--experiment-id",
        dest="run_timestamp",
        help=(
            "Optional YYYYMMDD-HHMMSS manifest timestamp. "
            "--experiment-id is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--training-timeout-s", type=float, default=3600.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.objective_alpha <= 0.0 or args.objective_beta <= 0.0:
        raise SystemExit("Objective weights must be positive")
    if args.training_timeout_s <= 0.0 or args.poll_interval_s <= 0.0:
        raise SystemExit("Timeout and poll interval must be positive")
    experiment_root = run_experiment(args)
    print(f"Completed experiment: {experiment_root}")


if __name__ == "__main__":
    main()
