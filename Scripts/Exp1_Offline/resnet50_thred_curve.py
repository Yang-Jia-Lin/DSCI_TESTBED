from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

from Src.Models.model_config import RESNET50 as MODEL_CFG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"Generate {MODEL_CFG.name} early-exit rate/accuracy lookup tables for "
            "Data/OfflineTables."
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
        help=(f"Directory for {MODEL_CFG.rate_csv.name} and {MODEL_CFG.acc_csv.name}."),
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=1.0)
    parser.add_argument("--num_thresholds", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not present under data_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            f"Overwrite existing {MODEL_CFG.rate_csv.name} and "
            f"{MODEL_CFG.acc_csv.name}."
        ),
    )
    parser.add_argument(
        "--timestamped_copy",
        action="store_true",
        help="Also save timestamped copies for audit/debug runs.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 <= args.threshold_min <= args.threshold_max <= 1.0:
        raise ValueError(
            "Threshold range must satisfy 0.0 <= threshold_min <= threshold_max <= 1.0."
        )
    if args.num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2.")


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


def save_tables(rates, accs, output_dir, overwrite, timestamped_copy):
    output_dir.mkdir(parents=True, exist_ok=True)
    rates_path = output_dir / MODEL_CFG.rate_csv.name
    accs_path = output_dir / MODEL_CFG.acc_csv.name

    existing = [path for path in (rates_path, accs_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing offline tables: {names}. "
            "Pass --overwrite to refresh the canonical tables."
        )

    rates.to_csv(rates_path, index=False)
    accs.to_csv(accs_path, index=False)
    saved = [rates_path, accs_path]

    if timestamped_copy:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        rates_copy = output_dir / f"{MODEL_CFG.artifact_prefix}_rates_{stamp}.csv"
        accs_copy = output_dir / f"{MODEL_CFG.artifact_prefix}_accs_{stamp}.csv"
        rates.to_csv(rates_copy, index=False)
        accs.to_csv(accs_copy, index=False)
        saved.extend([rates_copy, accs_copy])

    return saved


def main():
    args = parse_args()
    validate_args(args)

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from Src.Algorithm.Utils.difficulty_dataset import build_cifar10_test_transform

    device = resolve_device(args.device)
    if device.type == "cpu":
        warnings.filterwarnings("ignore", message=".*pin_memory.*")

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=build_cifar10_test_transform(),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(Path(args.model_path), device)
    thresholds = np.linspace(
        args.threshold_min, args.threshold_max, args.num_thresholds
    )

    # ── Step 1: Run inference once, cache all per-sample results ──────────────
    # The model outputs (conf, pred) are threshold-independent, so there is no
    # need to re-run the model for every threshold value.
    all_conf1, all_conf2 = [], []
    all_pred1, all_pred2, all_pred_full = [], [], []
    all_true = []

    print(
        f"Running inference on {len(dataset)} {MODEL_CFG.dataset_name} test samples "
        f"(once, then sweep {len(thresholds)} thresholds)."
    )
    with torch.no_grad():
        for images, true_labels in tqdm(test_loader, desc="Inference", ncols=80):
            images = images.to(device, non_blocking=True)

            out_full, out_e1, out_e2 = model(images, stage=None)

            conf1, pred1 = F.softmax(out_e1, dim=1).max(dim=1)
            conf2, pred2 = F.softmax(out_e2, dim=1).max(dim=1)

            all_conf1.append(conf1.cpu())
            all_conf2.append(conf2.cpu())
            all_pred1.append(pred1.cpu())
            all_pred2.append(pred2.cpu())
            all_pred_full.append(out_full.argmax(dim=1).cpu())
            all_true.append(true_labels)

    # Concatenate into flat tensors: shape (N,)
    conf1 = torch.cat(all_conf1)
    conf2 = torch.cat(all_conf2)
    pred1 = torch.cat(all_pred1)
    pred2 = torch.cat(all_pred2)
    pred_full = torch.cat(all_pred_full)
    true_labels = torch.cat(all_true)

    # ── Step 2: Vectorized threshold sweep (no model calls) ───────────────────
    exit1_rates, exit2_rates = [], []
    exit1_accs, exit2_accs, full_accs = [], [], []

    for threshold in tqdm(thresholds, desc="Threshold sweep", ncols=80):
        t = float(threshold)

        # Independent exit masks (same semantics as original double-if logic)
        e1_mask = conf1 >= t  # shape (N,) bool
        e2_mask = conf2 >= t

        # Exit rates
        exit1_rate = 100.0 * e1_mask.float().mean().item()
        exit2_rate = 100.0 * e2_mask.float().mean().item()

        # Per-head accuracy (independent, not cascade)
        exit1_correct = e1_mask & (pred1 == true_labels)
        exit2_correct = e2_mask & (pred2 == true_labels)

        exit1_acc = (
            (100.0 * exit1_correct.float().sum() / e1_mask.float().sum()).item()
            if e1_mask.any()
            else 0.0
        )

        exit2_acc = (
            (100.0 * exit2_correct.float().sum() / e2_mask.float().sum()).item()
            if e2_mask.any()
            else 0.0
        )

        # Cascade accuracy: exit1 → exit2 → full  (same as original if/elif/else)
        chosen_pred = torch.where(
            e1_mask, pred1, torch.where(e2_mask, pred2, pred_full)
        )
        accuracy = 100.0 * (chosen_pred == true_labels).float().mean().item()

        exit1_rates.append(exit1_rate)
        exit2_rates.append(exit2_rate)
        exit1_accs.append(exit1_acc)
        exit2_accs.append(exit2_acc)
        full_accs.append(accuracy)

    # ── Step 3: Build DataFrames and save ─────────────────────────────────────
    rates = pd.DataFrame(
        {
            "threshold": thresholds,
            "exit1_rate": exit1_rates,
            "exit2_rate": exit2_rates,
        }
    )
    accs = pd.DataFrame(
        {
            "threshold": thresholds,
            "exit1_accuracy": exit1_accs,
            "exit2_accuracy": exit2_accs,
            "accuracy": full_accs,
        }
    )

    saved = save_tables(
        rates=rates,
        accs=accs,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        timestamped_copy=args.timestamped_copy,
    )
    print("\nSaved offline tables:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
