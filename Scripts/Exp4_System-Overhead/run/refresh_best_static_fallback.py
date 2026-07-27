"""Refresh only cold foreground fallback measurements in a completed Exp4 run.

The completed PPO convergence logs, final decisions, and background-update
measurements are retained. Before any overwrite, the manifest, summaries,
figures, seed event streams, and complete S0 mode directories are copied into a
timestamped backup directory under the experiment root.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_controlled_policy_update import (  # noqa: E402
    _append_jsonl,
    _load_json,
    _quality_fields,
    _seed_everything,
)
from Src.Phase2_Scheduler.Service.algo_service import (  # noqa: E402
    AlgoService,
    AlgoServiceConfig,
)


DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT / "Scripts" / "Exp4_System-Overhead" / "result_data"
)
DEFAULT_FIGURE_DIR = (
    REPO_ROOT / "Scripts" / "Paper_figures"
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.refresh-tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _manifest_path(experiment_dir: Path) -> Path:
    legacy = experiment_dir / "manifest.json"
    manifests = sorted(experiment_dir.glob("manifest_*.json"))
    if legacy.is_file():
        return legacy
    if len(manifests) != 1:
        raise ValueError(
            f"Expected exactly one manifest under {experiment_dir}, "
            f"found {len(manifests)}"
        )
    return manifests[0]


def _seed_dir(experiment_dir: Path, seed: int) -> Path:
    matches = sorted(experiment_dir.glob(f"seed_{seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one directory for seed {seed}, found {len(matches)}"
        )
    return matches[0]


def _cold_state(seed_dir: Path) -> dict[str, Any]:
    payload = _load_json(seed_dir / "states.json")
    matches = [
        item for item in payload["states"] if item.get("state_id") == "S0"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one S0 state under {seed_dir}")
    return copy.deepcopy(matches[0]["state"])


def _backup(
    *,
    experiment_dir: Path,
    figure_dir: Path,
    manifest_path: Path,
    seeds: list[int],
    refresh_id: str,
) -> Path:
    backup_dir = experiment_dir / "backups" / f"before_{refresh_id}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, backup_dir / manifest_path.name)
    summary_dir = experiment_dir / "summary"
    if summary_dir.is_dir():
        shutil.copytree(summary_dir, backup_dir / "summary")
    for seed in seeds:
        seed_dir = _seed_dir(experiment_dir, seed)
        seed_backup = backup_dir / f"seed_{seed}"
        seed_backup.mkdir(parents=True, exist_ok=False)
        shutil.copytree(
            seed_dir / "modes" / "S0_cold",
            seed_backup / "S0_cold",
        )
        shutil.copy2(seed_dir / "run_events.jsonl", seed_backup / "run_events.jsonl")
    figure_backup = backup_dir / "result_figure"
    for pattern in ("4_*.pdf", "4_*.png"):
        for path in sorted(figure_dir.glob(pattern)):
            figure_backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, figure_backup / path.name)
    return backup_dir


def _service(
    measurement_dir: Path,
    *,
    alpha: float,
    beta: float,
) -> AlgoService:
    cache_dir = measurement_dir / "cache"
    return AlgoService(
        config=AlgoServiceConfig(
            objective_alpha=alpha,
            objective_beta=beta,
            default_fallback_strategy="best_static",
            auto_train=False,
            latest_solution_path=cache_dir / "latest_solution.npz",
            latest_meta_path=cache_dir / "latest_solution_meta.json",
            training_events_path=cache_dir / "training_events.jsonl",
            training_convergence_dir=measurement_dir / "convergence",
        )
    )


def refresh(
    experiment_dir: Path,
    figure_dir: Path,
    refresh_id: str,
) -> Path:
    experiment_dir = experiment_dir.resolve()
    figure_dir = figure_dir.resolve()
    manifest_path = _manifest_path(experiment_dir)
    manifest = _load_json(manifest_path)
    seeds = [int(seed) for seed in manifest["seeds"]]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected formal seeds [0, 1, 2, 3, 4], got {seeds}")
    alpha = float(manifest["objective_alpha"])
    beta = float(manifest["objective_beta"])
    if alpha != 1.0 or beta != 1.0:
        raise ValueError("This refresh requires objective_alpha=objective_beta=1")

    backup_dir = _backup(
        experiment_dir=experiment_dir,
        figure_dir=figure_dir,
        manifest_path=manifest_path,
        seeds=seeds,
        refresh_id=refresh_id,
    )
    refresh_dir = experiment_dir / "foreground_refreshes" / refresh_id
    refresh_dir.mkdir(parents=True, exist_ok=False)
    refreshed_at = time.time()
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        _seed_everything(seed)
        seed_dir = _seed_dir(experiment_dir, seed)
        mode_dir = seed_dir / "modes" / "S0_cold"
        immediate_path = mode_dir / "immediate_decision.json"
        final_path = mode_dir / "final_decision.json"
        record_path = mode_dir / "record.json"
        events_path = seed_dir / "run_events.jsonl"
        old_immediate = _load_json(immediate_path)
        final = _load_json(final_path)
        old_record = _load_json(record_path)
        state = _cold_state(seed_dir)
        state["round_id"] = str(old_record["run_id"])

        measurement_dir = refresh_dir / f"seed_{seed}"
        service = _service(measurement_dir, alpha=alpha, beta=beta)
        started = time.perf_counter()
        immediate = service.make_decision(state)
        foreground_service_ms = (time.perf_counter() - started) * 1000.0
        health = service.health()
        if health["training_status"] != "idle":
            raise AssertionError("Foreground-only refresh unexpectedly started PPO")
        if immediate.get("decision_source") != "default:best_static:edge":
            raise AssertionError(
                "Canonical fallback did not select pure edge: "
                f"{immediate.get('decision_source')}"
            )

        quality = _quality_fields(immediate, final)
        provenance = {
            "refresh_id": refresh_id,
            "refreshed_at": refreshed_at,
            "strategy": "best_static",
            "measurement_scope": "foreground_only",
            "background_metrics_retained": True,
            "backup_dir": str(backup_dir.relative_to(experiment_dir)),
        }
        immediate["foreground_refresh"] = provenance
        new_record = copy.deepcopy(old_record)
        new_record.update(
            {
                "decision_source": immediate["decision_source"],
                "foreground_service_ms": foreground_service_ms,
                **quality,
                "foreground_refresh": provenance,
            }
        )
        measurement = {
            "experiment_id": manifest["experiment_id"],
            "seed": seed,
            "old_foreground_service_ms": float(
                old_record["foreground_service_ms"]
            ),
            "new_foreground_service_ms": foreground_service_ms,
            "old_decision_source": old_immediate["decision_source"],
            "new_decision_source": immediate["decision_source"],
            "old_immediate_objective": float(old_record["immediate_objective"]),
            "new_immediate_objective": float(quality["immediate_objective"]),
            "final_objective": float(quality["final_objective"]),
            "old_utility_retention": old_record["utility_retention"],
            "new_utility_retention": quality["utility_retention"],
            "fallback_candidates": immediate.get("fallback_candidates"),
            "background_update_s_retained": float(
                old_record["background_update_s"]
            ),
            "background_epochs_retained": int(old_record["background_epochs"]),
        }
        _write_json_atomic(measurement_dir / "measurement.json", measurement)
        _write_json_atomic(measurement_dir / "immediate_decision.json", immediate)
        _write_json_atomic(immediate_path, immediate)
        _write_json_atomic(record_path, new_record)
        _append_jsonl(
            events_path,
            {
                "event": "cold_foreground_refresh",
                "supersedes_event": "immediate_decision",
                "experiment_id": manifest["experiment_id"],
                "seed": seed,
                "state_id": "S0",
                "refresh_id": refresh_id,
                "foreground_service_ms": foreground_service_ms,
                "decision": immediate,
                "quality": quality,
                "background_metrics_retained": True,
                "backup_dir": str(backup_dir.relative_to(experiment_dir)),
            },
        )
        rows.append(measurement)

    refresh_manifest = {
        "refresh_id": refresh_id,
        "created_at": refreshed_at,
        "experiment_id": manifest["experiment_id"],
        "strategy": "best_static",
        "scope": "cold_foreground_only",
        "seeds": seeds,
        "objective_alpha": alpha,
        "objective_beta": beta,
        "background_metrics_retained": True,
        "backup_dir": str(backup_dir.relative_to(experiment_dir)),
        "measurements": rows,
    }
    _write_json_atomic(refresh_dir / "refresh_manifest.json", refresh_manifest)
    manifest.setdefault("foreground_refreshes", []).append(
        {
            key: value
            for key, value in refresh_manifest.items()
            if key != "measurements"
        }
    )
    manifest["default_fallback_strategy"] = "best_static"
    _write_json_atomic(manifest_path, manifest)
    print(f"Backup: {backup_dir}")
    print(f"Refresh: {refresh_dir}")
    return refresh_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
    )
    parser.add_argument(
        "--refresh-id",
        default=f"best-static-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    refresh(args.experiment_dir, args.figure_dir, str(args.refresh_id))


if __name__ == "__main__":
    main()
