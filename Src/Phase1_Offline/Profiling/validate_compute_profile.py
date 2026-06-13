"""Validate one calibrated compute profile."""

from __future__ import annotations

import argparse

from Src.Shared.Profiles.compute_profile import load_compute_profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id")
    parser.add_argument("--backend")
    parser.add_argument("--bundle-id")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = load_compute_profile(
        args.profile_id,
        expected_backend=args.backend,
        expected_bundle=args.bundle_id,
    )
    print(f"Profile valid: {profile.profile_id}")
    print(f"  backend: {profile.metadata['backend']}")
    print(f"  layers: {len(profile.layers)}")
    print(f"  total_latency_s: {profile.total_latency_s:.9f}")
    print(f"  theta_flops_per_s: {profile.theta:.6f}")


if __name__ == "__main__":
    main()
