"""Train the ResNet-50 backbone and final classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from Src.Algorithm.Utils.utils_function import get_data_loaders, get_device
from Src.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50, freeze_layers
from Src.Models.model_config import RESNET50 as MODEL_CFG

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=str(MODEL_CFG.weights_dir / "resnet50_cifar10_backbone.pth"))
    parser.add_argument(
        "--log-output",
        default=str(PROJECT_ROOT / "Scripts" / "Results" / "OfflinePipeline" / "train_model.csv"),
    )
    return parser.parse_args(argv)


def _run_epoch(model, loader, device, criterion, optimizer=None):
    model.train(optimizer is not None)
    total_loss = total_correct = total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images, stage="final")
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total += int(labels.numel())
    return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total, 1)


def main(argv=None):
    args = parse_args(argv)
    device = get_device()
    train_loader, valid_loader, _ = get_data_loaders(
        root=str(PROJECT_ROOT / "Data" / "CIFAR10"),
        batch_size=args.batch_size,
        valid_size=0.1,
        random_seed=42,
        num_workers=args.num_workers,
    )
    model = MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True).to(device)
    freeze_layers(model, freeze_x2_fc=True, freeze_x3_fc=True)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, valid_loader, device, criterion)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(f"epoch={epoch} train_acc={train_acc:.2f} val_acc={val_acc:.2f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    log_output = Path(args.log_output)
    log_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(log_output, index=False)
    print(f"Saved weights: {output}")
    print(f"Saved log: {log_output}")


if __name__ == "__main__":
    main()
