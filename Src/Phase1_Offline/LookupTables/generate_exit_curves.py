"""Generate a bundle-scoped exit rate/accuracy curve table."""

import argparse

import pandas as pd
import torch

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    if paths.offline_table_path.exists() and not args.overwrite:
        raise FileExistsError(paths.offline_table_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    model.eval()
    loader = build_loader(
        bundle, "val", batch_size=args.batch_size, data_root=args.data_root, download=args.download
    )
    confidences = {item.exit_id: [] for item in bundle.exits}
    correct = {item.exit_id: [] for item in bundle.exits}
    final_correct = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = model.forward_features(images)
            for item in bundle.exits:
                logits = model.classify_exit(item.exit_id, features[item.attach_point])
                conf, pred = torch.softmax(logits, 1).max(1)
                confidences[item.exit_id].extend(conf.cpu().tolist())
                correct[item.exit_id].extend((pred == labels).cpu().tolist())
            final_correct.extend((model(images).argmax(1) == labels).cpu().tolist())
    rows = []
    for index in range(101):
        threshold = index / 100
        row = {"threshold": threshold, "final_accuracy": 100 * sum(final_correct) / len(final_correct)}
        sequential_counts = {item.exit_id: 0 for item in bundle.exits}
        sequential_correct = {item.exit_id: 0 for item in bundle.exits}
        final_count = 0
        total_correct = 0
        for item in bundle.exits:
            mask = [value >= threshold for value in confidences[item.exit_id]]
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
            row[f"{item.exit_id}_sequential_rate"] = 100 * count / len(final_correct)
            row[f"{item.exit_id}_sequential_accuracy"] = (
                100 * sequential_correct[item.exit_id] / count if count else 0.0
            )
        row["final_rate"] = 100 * final_count / len(final_correct)
        row["overall_accuracy"] = 100 * total_correct / len(final_correct)
        rows.append(row)
    paths.root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(paths.offline_table_path, index=False)
    paths.analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_path = paths.analysis_root / "threshold_curves.csv"
    frame.to_csv(analysis_path, index=False)
    print(f"Saved exit curves: {paths.offline_table_path}")
    print(f"Saved analysis copy: {analysis_path}")


if __name__ == "__main__":
    main()
