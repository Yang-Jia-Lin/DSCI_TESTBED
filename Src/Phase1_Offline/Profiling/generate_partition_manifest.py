"""Generate the shared executable partition manifest for ResNet50."""

from __future__ import annotations

import argparse

from Src.Shared.Partitioning.manifest import (
    DEFAULT_MANIFEST_ID,
    build_resnet50_manifest,
    write_partition_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", default=DEFAULT_MANIFEST_ID)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_resnet50_manifest(manifest_id=args.manifest_id)
    path = write_partition_manifest(manifest, args.output, overwrite=args.overwrite)
    print(f"Saved partition manifest: {path}")
    print(f"boundaries={len(manifest.boundaries)}, segments={len(manifest.segments)}")


if __name__ == "__main__":
    main()
