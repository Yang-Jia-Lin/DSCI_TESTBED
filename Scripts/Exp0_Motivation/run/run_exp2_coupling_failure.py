"""Prepare Figure 2 data for split-threshold coupling failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Scripts.Exp0_Motivation.run.config import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_curve_path,
    data_dir,
    prepare_result_dirs,
    required_curve_columns,
    save_config,
    update_paper_numbers,
)
from Src.Phase2_Scheduler.Objective.compute_latency import compute_total_latency  # noqa: E402
from Src.Phase2_Scheduler.Utils.parsing_data import parsing_rate_and_acc  # noqa: E402
from Src.Phase2_Scheduler.paras import Paras  # noqa: E402
from Src.Shared.Config.model_config import get_bundle  # noqa: E402
from Src.Shared.Partitioning.manifest import load_partition_manifest  # noqa: E402
from Src.Shared.Partitioning.split_actions import (  # noqa: E402
    encode_split_row,
    enumerate_deployment_pairs,
)


def _profile_owner(
    profile_id: str,
    manifest,
    *,
    bw_d2e: float | None = None,
    bw_e2c: float | None = None,
) -> dict:
    owner = {
        "resource_mode": "fixed_worker_pool",
        "bundle_id": manifest.bundle_id,
        "manifest_id": manifest.manifest_id,
        "model_hash": manifest.model_hash,
        "execution_profile_id": profile_id,
        "backend": "pytorch",
        "worker_count": 1,
        "threads_per_worker": 1,
        "protocol_overhead_s": DEFAULT_CONFIG.protocol_overhead_s,
        "tensor_transport_dtype": DEFAULT_CONFIG.tensor_transport_dtype,
    }
    if bw_d2e is not None:
        owner["BW_d2e"] = float(bw_d2e)
    if bw_e2c is not None:
        owner["BW_e2c"] = float(bw_e2c)
    return owner


def _state_for_bandwidth(bw_d2e: float) -> dict:
    cfg = DEFAULT_CONFIG
    bundle = get_bundle(cfg.bundle_id)
    manifest = load_partition_manifest(bundle.bundle_id)
    user = _profile_owner(cfg.device_profile_id, manifest, bw_d2e=bw_d2e)
    edge = _profile_owner(cfg.edge_profile_id, manifest)
    cloud = _profile_owner(
        cfg.cloud_profile_id, manifest, bw_e2c=cfg.bandwidth_e2c_mbps
    )
    return {
        "resource_mode": "fixed_worker_pool",
        "bundle_id": bundle.bundle_id,
        "tensor_transport_dtype": cfg.tensor_transport_dtype,
        "users": [user],
        "edge": edge,
        "cloud": cloud,
    }


def _paras_for_bandwidth(bw_d2e: float, curve_path: Path) -> Paras:
    paras = Paras.from_state(_state_for_bandwidth(bw_d2e))
    paras.rates, paras.accs = parsing_rate_and_acc(paras, curve_path)
    return paras


def _no_exit_probs(paras: Paras) -> np.ndarray:
    probs = np.zeros((1, paras.m), dtype=np.float64)
    probs[0, paras.m - 1] = 1.0
    return probs


def _threshold_probs(paras: Paras, candidate: dict) -> np.ndarray:
    if candidate["tau"] is None:
        return _no_exit_probs(paras)
    probs = np.zeros((1, paras.m), dtype=np.float64)
    for boundary, rate_pct in zip(paras.E, candidate["sequential_exit_rates_pct"]):
        probs[0, int(boundary)] = float(rate_pct) / 100.0
    probs[0, paras.m - 1] = float(candidate["final_rate_pct"]) / 100.0
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError(
            f"Exit probabilities do not sum to 1 for tau={candidate['tau_label']}"
        )
    return probs


def _latency_ms(paras: Paras, pair: tuple[int, int], probs: np.ndarray) -> float:
    x = np.asarray([encode_split_row(pair[0], pair[1], paras.m, dtype=np.float64)])
    f_e = np.zeros((1, 1), dtype=np.float64)
    f_c = np.zeros((1, 1), dtype=np.float64)
    return float(compute_total_latency(x, probs, f_e, f_c, paras)[0] * 1000.0)


def _choose_best_split_only(paras: Paras, pairs: Iterable[tuple[int, int]]) -> dict:
    probs = _no_exit_probs(paras)
    best = None
    for pair in pairs:
        latency = _latency_ms(paras, pair, probs)
        candidate = {"pair": pair, "latency_ms": latency}
        if best is None or latency < best["latency_ms"]:
            best = candidate
    assert best is not None
    return best


def _threshold_candidates(
    curves: pd.DataFrame, target_accuracy: float, main_accuracy: float
) -> list[dict]:
    candidates = [
        {
            "tau": None,
            "tau_label": "no_exit",
            "accuracy_pct": float(main_accuracy),
            "feasible": True,
            "flow_policy": "no_exit",
        }
    ]
    for _, row in curves.iterrows():
        accuracy = float(row["overall_accuracy"])
        if accuracy + 1e-12 >= target_accuracy:
            candidates.append({
                "tau": float(row["threshold"]),
                "tau_label": f"{float(row['threshold']):.2f}",
                "accuracy_pct": accuracy,
                "feasible": True,
                "flow_policy": "sequential",
                "sequential_exit_rates_pct": (
                    float(row["after_layer2_sequential_rate"]),
                    float(row["after_layer3_sequential_rate"]),
                ),
                "final_rate_pct": float(row["final_rate"]),
            })
    return candidates


def _choose_best_threshold_for_pair(
    paras: Paras,
    pair: tuple[int, int],
    threshold_candidates: list[dict],
) -> dict:
    best = None
    for candidate in threshold_candidates:
        probs = _threshold_probs(paras, candidate)
        latency = _latency_ms(paras, pair, probs)
        record = {**candidate, "pair": pair, "latency_ms": latency}
        if best is None or latency < best["latency_ms"]:
            best = record
    assert best is not None
    return best


def _choose_joint(
    paras: Paras,
    pairs: Iterable[tuple[int, int]],
    threshold_candidates: list[dict],
) -> dict:
    best = None
    for pair in pairs:
        record = _choose_best_threshold_for_pair(paras, pair, threshold_candidates)
        if best is None or record["latency_ms"] < best["latency_ms"]:
            best = record
    assert best is not None
    return best


def _result_row(
    bandwidth: float,
    strategy: str,
    pair: tuple[int, int],
    latency_ms: float,
    accuracy_pct: float,
    tau_label: str,
    feasible: bool = True,
) -> dict:
    return {
        "bandwidth_d2e_mbps": float(bandwidth),
        "strategy": strategy,
        "latency_ms": float(latency_ms),
        "accuracy_pct": float(accuracy_pct),
        "b1": int(pair[0]),
        "b2": int(pair[1]),
        "tau": tau_label,
        "feasible": bool(feasible),
    }


def _summarize(results: pd.DataFrame, run_dir: Path, main_accuracy: float) -> dict:
    pivot = results.pivot(
        index="bandwidth_d2e_mbps", columns="strategy", values="latency_ms"
    )
    inversion = pivot[pivot["Decoupled"] > pivot["Local-full"]]
    if inversion.empty:
        inversion_info = {
            "occurred": False,
            "first_bandwidth_mbps": None,
            "max_decoupled_minus_local_ms": float(
                (pivot["Decoupled"] - pivot["Local-full"]).max()
            ),
        }
    else:
        inversion_info = {
            "occurred": True,
            "first_bandwidth_mbps": float(inversion.index.min()),
            "max_decoupled_minus_local_ms": float(
                (inversion["Decoupled"] - inversion["Local-full"]).max()
            ),
        }
    improvement = (pivot["Decoupled"] - pivot["Joint"]) / pivot["Decoupled"] * 100.0
    positive_improvement = improvement[improvement > 1e-9]
    joint_over_ee_only = (pivot["EE-only"] - pivot["Joint"]) / pivot["EE-only"] * 100.0
    decoupled_over_ee_only = (
        (pivot["Decoupled"] - pivot["EE-only"]) / pivot["EE-only"] * 100.0
    )
    split_rows = results[results["strategy"].isin(["Split-only", "Decoupled", "Joint"])]
    split_pivot = split_rows.pivot(
        index="bandwidth_d2e_mbps", columns="strategy", values="b1"
    )
    divergence = (split_pivot["Decoupled"] - split_pivot["Joint"]).abs()
    return {
        "output_csv": str(data_dir(run_dir) / "exp2_coupling_failure.csv"),
        "accuracy_constraint_pct": float(
            main_accuracy - DEFAULT_CONFIG.accuracy_drop_tolerance_pp
        ),
        "main_exit_accuracy_pct": float(main_accuracy),
        "decoupled_performance_inversion": inversion_info,
        "max_joint_latency_reduction_over_decoupled_pct": float(improvement.max()),
        "first_bandwidth_with_joint_gain_mbps": (
            None
            if positive_improvement.empty
            else float(positive_improvement.index.min())
        ),
        "max_joint_latency_reduction_over_ee_only_pct": float(joint_over_ee_only.max()),
        "max_decoupled_latency_increase_over_ee_only_pct": float(
            decoupled_over_ee_only.max()
        ),
        "max_abs_b1_divergence_decoupled_vs_joint": int(divergence.max()),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    run_dir = prepare_result_dirs()
    save_config(run_dir, cfg)
    curve_path = canonical_curve_path(run_dir)
    if not curve_path.is_file():
        raise FileNotFoundError(f"Canonical curve not found: {curve_path}")
    curves = pd.read_csv(curve_path)
    missing = sorted(required_curve_columns(cfg).difference(curves.columns))
    if missing:
        raise ValueError(f"Canonical curve misses columns: {missing}")
    main_accuracy = float(curves["final_accuracy"].iloc[0])
    target_accuracy = main_accuracy - cfg.accuracy_drop_tolerance_pp
    candidates = _threshold_candidates(curves, target_accuracy, main_accuracy)
    if not candidates:
        raise AssertionError("No feasible threshold candidate found")

    all_rows = []
    for bandwidth in cfg.bandwidth_d2e_mbps:
        paras = _paras_for_bandwidth(float(bandwidth), curve_path)
        pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
        final = paras.m - 1
        local_pair = (final, final)
        cloud_pair = (0, 0)
        no_exit = _no_exit_probs(paras)
        local_latency = _latency_ms(paras, local_pair, no_exit)
        cloud_latency = _latency_ms(paras, cloud_pair, no_exit)
        split = _choose_best_split_only(paras, pairs)
        ee_only = _choose_best_threshold_for_pair(paras, local_pair, candidates)
        decoupled = _choose_best_threshold_for_pair(paras, split["pair"], candidates)
        joint = _choose_joint(paras, pairs, candidates)

        all_rows.extend([
            _result_row(
                bandwidth,
                "Local-full",
                local_pair,
                local_latency,
                main_accuracy,
                "no_exit",
            ),
            _result_row(
                bandwidth,
                "Cloud-full",
                cloud_pair,
                cloud_latency,
                main_accuracy,
                "no_exit",
            ),
            _result_row(
                bandwidth,
                "Split-only",
                split["pair"],
                split["latency_ms"],
                main_accuracy,
                "no_exit",
            ),
            _result_row(
                bandwidth,
                "EE-only",
                ee_only["pair"],
                ee_only["latency_ms"],
                ee_only["accuracy_pct"],
                ee_only["tau_label"],
            ),
            _result_row(
                bandwidth,
                "Decoupled",
                decoupled["pair"],
                decoupled["latency_ms"],
                decoupled["accuracy_pct"],
                decoupled["tau_label"],
            ),
            _result_row(
                bandwidth,
                "Joint",
                joint["pair"],
                joint["latency_ms"],
                joint["accuracy_pct"],
                joint["tau_label"],
            ),
        ])

    results = pd.DataFrame(all_rows)
    if not results["feasible"].all():
        raise AssertionError("At least one strategy is infeasible")
    out_path = data_dir(run_dir) / "exp2_coupling_failure.csv"
    results.to_csv(out_path, index=False)
    update_paper_numbers(
        run_dir, "figure2", _summarize(results, run_dir, main_accuracy)
    )
    print(f"Figure 2 data: {out_path}")


if __name__ == "__main__":
    main()
