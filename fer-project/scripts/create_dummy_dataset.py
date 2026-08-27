"""Script to generate a small synthetic RAF-DB dataset for testing the full pipeline."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RAF-DB dataset for testing"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory path",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50,
        help="Number of synthetic train samples",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=20,
        help="Number of synthetic test samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.data_dir) / "raw" / "rafdb"
    img_dir = base_dir / "Image" / "aligned"
    label_dir = base_dir / "EmoLabel"

    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    label_file = label_dir / "list_patition_label.txt"
    lines = []

    np.random.seed(42)

    # Generate train images
    for i in range(1, args.num_train + 1):
        fname = f"train_{i:05d}_aligned.jpg"
        label = int(np.random.randint(1, 8))  # RAF-DB raw labels 1..7

        # Create a synthetic 100x100 RGB image with random colors
        img_array = np.random.randint(50, 220, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_dir / fname, quality=90)

        lines.append(f"{fname} {label}\n")

    # Generate test images
    for i in range(1, args.num_test + 1):
        fname = f"test_{i:05d}_aligned.jpg"
        label = int(np.random.randint(1, 8))

        img_array = np.random.randint(50, 220, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_dir / fname, quality=90)

        lines.append(f"{fname} {label}\n")

    with open(label_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("=" * 60)
    print("SYNTHETIC DATASET GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Location:           {base_dir.resolve()}")
    print(f"Train samples:      {args.num_train}")
    print(f"Test samples:       {args.num_test}")
    print(f"Annotation file:    {label_file.resolve()}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
