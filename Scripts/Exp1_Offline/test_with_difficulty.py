from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from Src.Models.model_config import RESNET50 as MODEL_CFG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXIT1_LAYER = 57
EXIT2_LAYER = 103
FINAL_LAYER = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate early-exit inference and summarize results by difficulty."
    )
    parser.add_argument(
        "--model_path",
        default=str(MODEL_CFG.resolve_weight_path()),
        help="Path to the MultiEEResNet50 state_dict.",
    )
    parser.add_argument(
        "--table_path",
        default=str(
            PROJECT_ROOT
            / "Data"
            / "OfflineTables"
            / f"{MODEL_CFG.artifact_prefix}_difficulty_labeled.csv"
        ),
        help="Labeled difficulty CSV from assign_difficulty_labels.py.",
    )
    parser.add_argument(
        "--data_root",
        default=str(MODEL_CFG.data_dir / MODEL_CFG.dataset_name),
        help="CIFAR-10 root, either Data/CIFAR10 or Data/CIFAR10/cifar-10-batches-py.",
    )
    parser.add_argument(
        "--partition_idx",
        type=int,
        default=3,
        help=(
            "Local partition stage 0..4, or a model layer boundary such as 57/103. "
            "Stage 2 owns exit layer 57; stage 3 owns exit layers 57 and 103."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "Scripts" / "Results" / "Exp0_Offline"),
        help=(
            "Directory for "
            f"{MODEL_CFG.artifact_prefix}_difficulty_results_{{timestamp}}.csv."
        ),
    )
    parser.add_argument("--exit_threshold_57", type=float, default=0.80)
    parser.add_argument("--exit_threshold_103", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--difficulty",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Optional subset to evaluate. Defaults to all labeled samples.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not present under data_root.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    return parser.parse_args()


def resolve_device(choice):
    import torch

    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def resolve_partition_stage(partition_idx: int) -> int:
    if partition_idx < 0:
        raise ValueError(f"partition_idx must be non-negative, got {partition_idx}")
    if partition_idx <= 4:
        return int(partition_idx)
    if partition_idx <= 27:
        return 1
    if partition_idx <= EXIT1_LAYER:
        return 2
    if partition_idx <= EXIT2_LAYER:
        return 3
    return 4


def validate_thresholds(threshold_57: float, threshold_103: float):
    for name, value in {
        "exit_threshold_57": threshold_57,
        "exit_threshold_103": threshold_103,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


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


def forward_all_logits(model, images):
    import torch

    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x1 = model.layer1(x)

    x2 = model.layer2(x1)
    exit1_logits = model.fc2(torch.flatten(model.avgpool(x2), 1))

    x3 = model.layer3(x2)
    exit2_logits = model.fc3(torch.flatten(model.avgpool(x3), 1))

    x4 = model.layer4(x3)
    final_logits = model.fc(torch.flatten(model.avgpool(x4), 1))

    return final_logits, exit1_logits, exit2_logits


def choose_prediction(
    offset,
    partition_stage,
    thresholds,
    final_conf,
    final_pred,
    exit1_conf,
    exit1_pred,
    exit2_conf,
    exit2_pred,
):
    if (
        partition_stage >= 2
        and float(exit1_conf[offset].item()) >= thresholds[EXIT1_LAYER]
    ):
        return (
            int(exit1_pred[offset].item()),
            float(exit1_conf[offset].item()),
            EXIT1_LAYER,
            False,
        )

    if (
        partition_stage >= 3
        and float(exit2_conf[offset].item()) >= thresholds[EXIT2_LAYER]
    ):
        return (
            int(exit2_pred[offset].item()),
            float(exit2_conf[offset].item()),
            EXIT2_LAYER,
            False,
        )

    return (
        int(final_pred[offset].item()),
        float(final_conf[offset].item()),
        FINAL_LAYER,
        partition_stage < 4,
    )


def print_summary(rows, partition_idx, partition_stage):
    print(
        f"\n=== Results Summary (partition_idx={partition_idx}, "
        f"partition_stage={partition_stage}) ==="
    )
    print("Difficulty | Samples | Accuracy | Exit Rate (local) | Avg Exit Layer")
    print("-----------+---------+----------+-------------------+---------------")

    def summarise(label, subset):
        samples = len(subset)
        if samples == 0:
            print(f"{label:<10} | {0:7d} | {'n/a':>8} | {'n/a':>17} | {'n/a':>13}")
            return
        correct = sum(1 for row in subset if row["correct"])
        local = sum(1 for row in subset if not row["transmitted_to_cloud"])
        avg_exit_layer = sum(row["exit_layer"] for row in subset) / samples
        print(
            f"{label:<10} | {samples:7d} | {correct / samples * 100:7.2f}% | "
            f"{local / samples * 100:16.2f}% | {avg_exit_layer:13.2f}"
        )

    for label in ("easy", "medium", "hard"):
        summarise(label, [row for row in rows if row["difficulty"] == label])
    summarise("all", rows)


def main():
    args = parse_args()
    validate_thresholds(args.exit_threshold_57, args.exit_threshold_103)

    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from Src.Deploy.Shared.dataloader import (
        DifficultyAwareDataset,
        build_cifar10_test_transform,
    )

    device = resolve_device(args.device)
    partition_stage = resolve_partition_stage(args.partition_idx)
    thresholds = {
        EXIT1_LAYER: args.exit_threshold_57,
        EXIT2_LAYER: args.exit_threshold_103,
    }

    model = load_model(Path(args.model_path), device)
    dataset = DifficultyAwareDataset(
        data_root=args.data_root,
        difficulty_table_path=args.table_path,
        difficulty=args.difficulty,
        train=False,
        transform=build_cifar10_test_transform(),
        download=args.download,
        include_image_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    processed = 0
    with torch.no_grad():
        for images, labels, difficulties, _profile_conf, image_ids in tqdm(
            loader, desc="Testing by difficulty", ncols=80
        ):
            images = images.to(device, non_blocking=True)
            final_logits, exit1_logits, exit2_logits = forward_all_logits(model, images)

            final_conf, final_pred = F.softmax(final_logits, dim=1).max(dim=1)
            exit1_conf, exit1_pred = F.softmax(exit1_logits, dim=1).max(dim=1)
            exit2_conf, exit2_pred = F.softmax(exit2_logits, dim=1).max(dim=1)

            batch_size = int(labels.size(0))
            limit = batch_size
            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                limit = min(batch_size, remaining)

            for offset in range(limit):
                pred, confidence, exit_layer, transmitted = choose_prediction(
                    offset=offset,
                    partition_stage=partition_stage,
                    thresholds=thresholds,
                    final_conf=final_conf,
                    final_pred=final_pred,
                    exit1_conf=exit1_conf,
                    exit1_pred=exit1_pred,
                    exit2_conf=exit2_conf,
                    exit2_pred=exit2_pred,
                )
                true_label = int(labels[offset].item())
                rows.append(
                    {
                        "image_id": int(image_ids[offset].item()),
                        "true_label": true_label,
                        "pred_label": pred,
                        "correct": bool(pred == true_label),
                        "confidence": confidence,
                        "difficulty": str(difficulties[offset]),
                        "exit_layer": int(exit_layer),
                        "transmitted_to_cloud": bool(transmitted),
                    }
                )
            processed += limit

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv = (
        output_dir / f"{MODEL_CFG.artifact_prefix}_difficulty_results_{timestamp}.csv"
    )
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print_summary(rows, args.partition_idx, partition_stage)
    print(f"\nSaved per-sample results: {output_csv}")


if __name__ == "__main__":
    main()
