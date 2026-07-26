"""Build canonical tables for Exp0 motivation studies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Scripts.Exp0_Motivation.run.config import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_curve_path,
    prepare_result_dirs,
    save_config,
    update_paper_numbers,
)
from Src.Phase1_Offline.LookupTables.generate_exit_curves import (  # noqa: E402
    build_exit_curve_frame,
)
from Src.Shared.Config.model_config import get_bundle  # noqa: E402
from Src.Shared.Config.paths import bundle_paths  # noqa: E402


def summarize_curve(frame: pd.DataFrame) -> dict:
    final_accuracy = float(frame["final_accuracy"].iloc[0])
    return {
        "final_accuracy_pct": final_accuracy,
        "overall_accuracy_max_pct": float(frame["overall_accuracy"].max()),
        "overall_accuracy_at_tau_1_pct": float(frame["overall_accuracy"].iloc[-1]),
        "final_rate_at_tau_1_pct": float(frame["final_rate"].iloc[-1]),
        "num_rows": int(len(frame)),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG.curve_batch_size
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    run_dir = prepare_result_dirs()
    save_config(run_dir, cfg)
    bundle = get_bundle(cfg.bundle_id)
    out_path = canonical_curve_path(run_dir)

    if args.reuse_existing:
        source_path = bundle_paths(bundle.bundle_id).offline_table_path
        frame = pd.read_csv(source_path)
    else:
        frame = build_exit_curve_frame(
            bundle,
            data_root=args.data_root,
            batch_size=args.batch_size,
            download=args.download,
            split=cfg.dataset_split,
        )
    frame.to_csv(out_path, index=False)
    summary = summarize_curve(frame)
    update_paper_numbers(
        run_dir,
        "tables",
        {
            "canonical_curve_csv": str(out_path),
            **summary,
        },
    )

    print(f"Experiment directory: {run_dir}")
    print(f"Canonical curve: {out_path}")
    print(f"Final accuracy: {summary['final_accuracy_pct']:.2f}%")


if __name__ == "__main__":
    main()
