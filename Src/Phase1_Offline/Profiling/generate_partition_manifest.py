"""Generate an executable partition manifest for a model bundle."""

import argparse

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Partitioning.manifest import build_partition_manifest, write_partition_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    manifest = build_partition_manifest(bundle)
    path = write_partition_manifest(manifest, args.output, overwrite=args.overwrite)
    print(f"Saved partition manifest: {path}")
    print(f"boundaries={len(manifest.boundaries)}, segments={len(manifest.segments)}")


if __name__ == "__main__":
    main()
