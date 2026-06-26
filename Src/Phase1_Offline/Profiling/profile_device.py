"""Create a simulation compute profile from bundle-aligned measurements."""

import argparse

import pandas as pd

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Profiles.compute_profile import write_compute_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id")
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--latencies-csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    stats = pd.read_csv(bundle_paths(bundle.bundle_id).layer_stats_path)
    measured = pd.read_csv(args.latencies_csv)
    if "raw_latency_s" not in measured:
        raise ValueError("latencies CSV requires raw_latency_s")
    profile = write_compute_profile(
        profile_id=args.profile_id,
        layer_names=stats["name"],
        theoretical_flops=stats["approx_flops"],
        raw_latencies_s=measured["raw_latency_s"],
        total_latency_s=float(measured["raw_latency_s"].sum()),
        bundle_id=bundle.bundle_id,
        backend=args.backend,
        overwrite=args.overwrite,
    )
    print(profile.profile_dir)


if __name__ == "__main__":
    main()
