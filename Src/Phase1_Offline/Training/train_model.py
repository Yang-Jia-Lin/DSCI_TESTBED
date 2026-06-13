"""Train the final classifier of a selected model bundle."""

import argparse

import torch
import torch.nn as nn

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    loader = build_loader(
        bundle, "train", batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} loss={float(loss.item()):.6f}")
    path = bundle_paths(bundle.bundle_id).weight_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved weights: {path}")


if __name__ == "__main__":
    main()
