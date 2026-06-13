"""Fine-tune every early-exit head in a selected model bundle."""

import argparse

import torch
import torch.nn as nn

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model, freeze_for_exit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs-per-exit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    loader = build_loader(
        bundle, "train", batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root
    )
    criterion = nn.CrossEntropyLoss()
    for exit_spec in bundle.exits:
        freeze_for_exit(model, exit_spec.exit_id)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr
        )
        for epoch in range(args.epochs_per_exit):
            for images, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(images.to(device), exit_id=exit_spec.exit_id), labels.to(device))
                loss.backward()
                optimizer.step()
            print(f"exit_id={exit_spec.exit_id} epoch={epoch + 1} loss={float(loss.item()):.6f}")
    torch.save(model.state_dict(), paths.weight_path)


if __name__ == "__main__":
    main()
