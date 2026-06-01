from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Models.model_config import RESNET50 as MODEL_CFG


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"Profile {MODEL_CFG.dataset_name} sample difficulty with the final "
            f"{MODEL_CFG.name} head."
        )
    )
    parser.add_argument(
        "--model_path",
        default=str(MODEL_CFG.resolve_weight_path()),
        help="Path to the MultiEEResNet50 state_dict.",
    )
    parser.add_argument(
        "--data_root",
        default=str(MODEL_CFG.data_dir / MODEL_CFG.dataset_name),
        help="Root passed to torchvision.datasets.CIFAR10.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "Data" / "OfflineTables"),
        help="Directory for the raw CSV and confidence stats JSON.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not present under data_root.",
    )
    parser.add_argument("--suggest_easy_min", type=float, default=0.90)
    parser.add_argument("--suggest_hard_max", type=float, default=0.60)
    return parser.parse_args()


def resolve_device(choice):
    import torch

    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def load_model(model_path, device):
    import torch

    from Src.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50

    model = MultiEEResNet50(
        Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
    ).to(device)

    try:
        payload = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]

    if isinstance(payload, dict):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}

    model.load_state_dict(payload)
    model.eval()
    return model


def write_outputs(rows, confidences, correct_flags, output_dir, easy_min, hard_max):
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / f"{MODEL_CFG.artifact_prefix}_difficulty_raw.csv"
    stats_json = output_dir / f"{MODEL_CFG.artifact_prefix}_confidence_stats.json"

    df = pd.DataFrame(rows)
    df.to_csv(raw_csv, index=False)

    conf_array = np.asarray(confidences, dtype=np.float64)
    percentiles = {
        f"p{p}": float(np.percentile(conf_array, p)) for p in (10, 25, 50, 75, 90)
    }
    total = int(len(conf_array))
    easy_count = int((conf_array >= easy_min).sum())
    hard_count = int((conf_array <= hard_max).sum())
    medium_count = int(((conf_array > hard_max) & (conf_array < easy_min)).sum())
    overall_accuracy = float(np.mean(correct_flags)) if total else 0.0

    stats = {
        "total_samples": total,
        "overall_accuracy": overall_accuracy,
        "confidence_percentiles": percentiles,
        "suggested_thresholds": {
            "easy_min": float(easy_min),
            "hard_max": float(hard_max),
        },
        "suggested_counts": {
            "easy": easy_count,
            "medium": medium_count,
            "hard": hard_count,
        },
    }
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== Confidence Distribution ===")
    print("  ".join(f"{name}={value:.4f}" for name, value in percentiles.items()))
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"\nSuggested thresholds (easy_min={easy_min:.2f}, hard_max={hard_max:.2f}):")
    print(f"  easy   (conf >= {easy_min:.2f}): ~{easy_count} samples")
    print(f"  medium ({hard_max:.2f} < conf < {easy_min:.2f}): ~{medium_count} samples")
    print(f"  hard   (conf <= {hard_max:.2f}): ~{hard_count} samples")
    print("\nAdjust thresholds with assign_difficulty_labels.py")
    print(f"\nSaved raw table: {raw_csv}")
    print(f"Saved stats: {stats_json}")


def main():
    args = parse_args()
    if not 0.0 <= args.suggest_hard_max < args.suggest_easy_min <= 1.0:
        raise ValueError(
            "Suggested thresholds must satisfy "
            "0.0 <= suggest_hard_max < suggest_easy_min <= 1.0, "
            f"got suggest_hard_max={args.suggest_hard_max}, "
            f"suggest_easy_min={args.suggest_easy_min}"
        )

    import torch
    import torch.nn.functional as F
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from Src.Algorithm.Utils.difficulty_dataset import build_cifar10_test_transform

    device = resolve_device(args.device)
    model = load_model(Path(args.model_path), device)
    transform = build_cifar10_test_transform()
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    confidences = []
    correct_flags = []
    seen = 0

    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=f"Profiling {MODEL_CFG.dataset_name}", ncols=80
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images, stage="final")
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

            batch_size = int(labels.size(0))
            for offset in range(batch_size):
                confidence = float(conf[offset].item())
                predicted = int(pred[offset].item())
                true_label = int(labels[offset].item())
                correct = bool(predicted == true_label)
                rows.append(
                    {
                        "image_id": seen + offset,
                        "true_label": true_label,
                        "pred_label": predicted,
                        "correct": correct,
                        "confidence": confidence,
                        "entropy": float(entropy[offset].item()),
                    }
                )
                confidences.append(confidence)
                correct_flags.append(correct)
            seen += batch_size

    write_outputs(
        rows=rows,
        confidences=confidences,
        correct_flags=correct_flags,
        output_dir=Path(args.output_dir),
        easy_min=args.suggest_easy_min,
        hard_max=args.suggest_hard_max,
    )


if __name__ == "__main__":
    main()
