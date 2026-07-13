"""Generate a bundle-scoped exit rate/accuracy curve table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Src.Shared.Config.model_config import ModelBundleSpec, get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Utils.phase_timing import timed_event


def build_exit_curve_frame(
    bundle: ModelBundleSpec | str | None = None,
    *,
    data_root: str | Path | None = None,
    batch_size: int = 64,
    download: bool = False,
    split: str = "val",
    device=None,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Build the threshold curve table without writing it to disk."""
    import torch

    from Src.Shared.Data.registry import build_loader
    from Src.Shared.Models.factory import build_model

    bundle = get_bundle(bundle) if not isinstance(bundle, ModelBundleSpec) else bundle
    paths = bundle_paths(bundle.bundle_id)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(bundle).to(selected_device)
    model.load_state_dict(
        torch.load(paths.weight_path, map_location=selected_device, weights_only=True)
    )
    model.eval()
    loader = build_loader(
        bundle,
        split,
        batch_size=batch_size,
        data_root=data_root,
        download=download,
        num_workers=num_workers,
    )
    confidences = {item.exit_id: [] for item in bundle.exits}
    correct = {item.exit_id: [] for item in bundle.exits}
    final_correct = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(selected_device), labels.to(selected_device)
            features = model.forward_features(images)
            for item in bundle.exits:
                logits = model.classify_exit(item.exit_id, features[item.attach_point])
                conf, pred = torch.softmax(logits, 1).max(1)
                confidences[item.exit_id].extend(conf.cpu().tolist())
                correct[item.exit_id].extend((pred == labels).cpu().tolist())
            final_correct.extend((model(images).argmax(1) == labels).cpu().tolist())

    rows = []
    sample_count = len(final_correct)
    if sample_count == 0:
        raise ValueError("Cannot build exit curves from an empty dataset split")
    for index in range(101):
        threshold = index / 100
        row = {
            "threshold": threshold,
            "final_accuracy": 100 * sum(final_correct) / sample_count,
        }
        sequential_counts = {item.exit_id: 0 for item in bundle.exits}
        sequential_correct = {item.exit_id: 0 for item in bundle.exits}
        final_count = 0
        total_correct = 0
        for item in bundle.exits:
            mask = [value >= threshold for value in confidences[item.exit_id]]
            row[f"{item.exit_id}_isolated_accuracy"] = 100 * sum(correct[item.exit_id]) / sample_count
            row[f"{item.exit_id}_rate"] = 100 * sum(mask) / len(mask)
            selected = [ok for ok, keep in zip(correct[item.exit_id], mask) if keep]
            row[f"{item.exit_id}_accuracy"] = 100 * sum(selected) / max(len(selected), 1)
        for sample_index, final_ok in enumerate(final_correct):
            chosen_exit = None
            for item in bundle.exits:
                if confidences[item.exit_id][sample_index] >= threshold:
                    chosen_exit = item.exit_id
                    break
            if chosen_exit is None:
                final_count += 1
                total_correct += int(final_ok)
            else:
                sequential_counts[chosen_exit] += 1
                ok = correct[chosen_exit][sample_index]
                sequential_correct[chosen_exit] += int(ok)
                total_correct += int(ok)
        for item in bundle.exits:
            count = sequential_counts[item.exit_id]
            row[f"{item.exit_id}_sequential_rate"] = 100 * count / sample_count
            row[f"{item.exit_id}_sequential_accuracy"] = (
                100 * sequential_correct[item.exit_id] / count if count else 0.0
            )
        row["final_rate"] = 100 * final_count / sample_count
        row["overall_accuracy"] = 100 * total_correct / sample_count
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    output_path = Path(args.output_csv) if args.output_csv else paths.offline_table_path
    with timed_event(phase="phase1", step="exit_curves", bundle_id=bundle.bundle_id):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(output_path)
        frame = build_exit_curve_frame(
            bundle,
            data_root=args.data_root,
            batch_size=args.batch_size,
            download=args.download,
            split=args.split,
            num_workers=args.num_workers,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        analysis_path = None
        if output_path == paths.offline_table_path:
            paths.analysis_root.mkdir(parents=True, exist_ok=True)
            analysis_path = paths.analysis_root / "threshold_curves.csv"
            frame.to_csv(analysis_path, index=False)
    print(f"Saved exit curves: {output_path}")
    if analysis_path is not None:
        print(f"Saved analysis copy: {analysis_path}")


if __name__ == "__main__":
    main()
