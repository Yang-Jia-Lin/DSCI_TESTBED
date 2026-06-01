from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assign_difficulty(confidence: float, easy_min: float, hard_max: float) -> str:
    if confidence >= easy_min:
        return "easy"
    if confidence <= hard_max:
        return "hard"
    return "medium"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign easy/medium/hard labels to CIFAR-10 profiling rows."
    )
    parser.add_argument(
        "--input_path",
        default=str(PROJECT_ROOT / "Data" / "OfflineTables" / "cifar10_difficulty_table.csv"),
        help="Raw profiling CSV from profile_difficulty.py.",
    )
    parser.add_argument(
        "--output_path",
        default=str(
            PROJECT_ROOT
            / "Data"
            / "OfflineTables"
            / "cifar10_difficulty_table_labeled.csv"
        ),
        help="Labeled difficulty CSV used by downstream experiments.",
    )
    parser.add_argument("--easy_min", type=float, default=0.90)
    parser.add_argument("--hard_max", type=float, default=0.60)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.hard_max < args.easy_min <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0.0 <= hard_max < easy_min <= 1.0, "
            f"got hard_max={args.hard_max}, easy_min={args.easy_min}"
        )

    import pandas as pd

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    df = pd.read_csv(input_path)
    required = {"image_id", "true_label", "pred_label", "correct", "confidence", "entropy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    df["difficulty"] = [
        assign_difficulty(float(conf), args.easy_min, args.hard_max)
        for conf in df["confidence"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    counts = df["difficulty"].value_counts().to_dict()
    print("\n=== Difficulty Distribution ===")
    for label in ("easy", "medium", "hard"):
        count = int(counts.get(label, 0))
        ratio = (count / total * 100.0) if total else 0.0
        print(f"{label:<6}: {count:5d} samples ({ratio:5.1f}%)")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
