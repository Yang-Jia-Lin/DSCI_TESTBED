"""Fine-tune every early-exit head in a selected model bundle."""

import argparse
import copy

import torch
import torch.nn as nn
import pandas as pd

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model, freeze_for_exit


def _run_exit_epoch(model, loader, criterion, device, exit_id, optimizer=None):
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
            logits = model(images, exit_id=exit_id)
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
    parser.add_argument("--epochs-per-exit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    train_loader = build_loader(
        bundle, "train", batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, download=args.download
    )
    val_loader = build_loader(
        bundle, "val", batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, download=args.download
    )
    criterion = nn.CrossEntropyLoss()
    log_rows = []
    for exit_spec in bundle.exits:
        freeze_for_exit(model, exit_spec.exit_id)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr
        )
        best_acc = -1.0
        best_state = copy.deepcopy(model.exit_heads[exit_spec.exit_id].state_dict())
        for epoch in range(args.epochs_per_exit):
            train_loss, train_acc = _run_exit_epoch(
                model, train_loader, criterion, device, exit_spec.exit_id, optimizer
            )
            val_loss, val_acc = _run_exit_epoch(
                model, val_loader, criterion, device, exit_spec.exit_id
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.exit_heads[exit_spec.exit_id].state_dict())
            log_rows.append(
                {
                    "stage": exit_spec.exit_id,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_acc,
                }
            )
            print(
                f"exit_id={exit_spec.exit_id} epoch={epoch + 1} "
                f"train_loss={train_loss:.6f} train_acc={train_acc:.2f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.2f} best_val_acc={best_acc:.2f}"
            )
        model.exit_heads[exit_spec.exit_id].load_state_dict(best_state)
    torch.save(model.state_dict(), paths.weight_path)
    log_path = paths.analysis_root / "finetune_exits_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Saved exit fine-tune log: {log_path}")


if __name__ == "__main__":
    main()
