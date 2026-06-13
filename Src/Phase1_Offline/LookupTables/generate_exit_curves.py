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
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    if paths.offline_table_path.exists() and not args.overwrite:
        raise FileExistsError(paths.offline_table_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    model.eval()
    loader = build_loader(bundle, "val", batch_size=args.batch_size, data_root=args.data_root)
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
        for item in bundle.exits:
            mask = [value >= threshold for value in confidences[item.exit_id]]
            row[f"{item.exit_id}_rate"] = 100 * sum(mask) / len(mask)
            selected = [ok for ok, keep in zip(correct[item.exit_id], mask) if keep]
            row[f"{item.exit_id}_accuracy"] = 100 * sum(selected) / max(len(selected), 1)
        rows.append(row)
    paths.root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.offline_table_path, index=False)


if __name__ == "__main__":
    main()
