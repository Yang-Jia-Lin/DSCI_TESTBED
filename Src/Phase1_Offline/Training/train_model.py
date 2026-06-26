"""Train the final classifier of a selected model bundle."""

import argparse
import copy

import torch
import torch.nn as nn
import pandas as pd

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    train_loader = build_loader(
        bundle, "train", batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, download=args.download
    )
    val_loader = build_loader(
        bundle, "val", batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, download=args.download
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    log_rows = []
    for epoch in range(args.epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        log_rows.append(
            {
                "stage": "final",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_acc,
            }
        )
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.2f} best_val_acc={best_acc:.2f}"
        )
    path = bundle_paths(bundle.bundle_id).weight_path
    path.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), path)
    print(f"Saved weights: {path}")
    log_path = bundle_paths(bundle.bundle_id).analysis_root / "train_model_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
