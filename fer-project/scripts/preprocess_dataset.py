"""CLI script for offline dataset preprocessing."""

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

from fer.data.label_mapping import is_excluded_label, map_label
from fer.data.preprocessing import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Preprocessing for FER Datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["rafdb", "fer2013", "affectnet"],
        help="Dataset name to process.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory path.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Output image size (height & width).",
    )
    parser.add_argument(
        "--bypass-face-detection",
        action="store_true",
        help="Skip face detection during preprocessing.",
    )
    return parser.parse_args()


def preprocess_rafdb(data_dir: Path, target_size: int, bypass_face_detection: bool):
    """Preprocess RAF-DB dataset."""
    anno_candidates = [
        data_dir / "raw" / "rafdb" / "EmoLabel" / "list_patition_label.txt",
        data_dir / "raw" / "rafdb" / "list_patition_label.txt",
        data_dir / "EmoLabel" / "list_patition_label.txt",
        data_dir / "list_patition_label.txt",
    ]

    anno_file = None
    for candidate in anno_candidates:
        if candidate.exists():
            anno_file = candidate
            break

    if anno_file is None:
        print(f"Error: RAF-DB annotation file not found under {data_dir}")
        sys.exit(1)

    img_dir_candidates = [
        data_dir / "raw" / "rafdb" / "Image" / "aligned",
        data_dir / "Image" / "aligned",
        data_dir / "aligned",
    ]

    img_dir = None
    for candidate in img_dir_candidates:
        if candidate.exists():
            img_dir = candidate
            break

    if img_dir is None:
        print(f"Error: RAF-DB aligned image directory not found under {data_dir}")
        sys.exit(1)

    image_paths = []
    labels = []
    splits = []
    original_labels = []

    with open(anno_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                fname, orig_lbl_str = parts[0], parts[1]
                if not fname.endswith(".jpg"):
                    fname = f"{fname}.jpg"
                img_path = img_dir / fname
                if not img_path.exists():
                    continue

                try:
                    orig_lbl = int(orig_lbl_str)
                    mapped_lbl = map_label("rafdb", orig_lbl)
                except ValueError:
                    continue

                split = "test" if fname.startswith("test_") else "train"

                image_paths.append(img_path)
                labels.append(mapped_lbl)
                splits.append(split)
                original_labels.append(orig_lbl)

    print(f"Found {len(image_paths)} valid RAF-DB images to process.")

    preprocessor = Preprocessor(
        target_size=target_size, bypass_face_detection=bypass_face_detection
    )
    output_dir = data_dir / "processed" / "rafdb"
    df = preprocessor.process_dataset(
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        dataset_name="rafdb",
        output_dir=output_dir,
        original_labels=original_labels,
    )
    preprocessor.close()
    print(f"RAF-DB preprocessing complete. Output: {output_dir}")


def preprocess_fer2013(data_dir: Path, target_size: int, bypass_face_detection: bool):
    """Preprocess FER2013 dataset (supports fer2013.csv or folder image layout)."""
    image_paths = []
    labels = []
    splits = []
    original_labels = []

    folder_to_label = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6,
    }

    raw_fer_dir = data_dir / "raw" / "fer2013"
    archive_dir = raw_fer_dir / "archive" if (raw_fer_dir / "archive").exists() else raw_fer_dir

    if (archive_dir / "train").exists() or (archive_dir / "test").exists():
        for split_name in ["train", "test"]:
            split_dir = archive_dir / split_name
            if not split_dir.exists():
                continue
            for em_name, orig_lbl in folder_to_label.items():
                em_dir = split_dir / em_name
                if not em_dir.exists():
                    continue
                mapped_lbl = map_label("fer2013", orig_lbl)
                for img_p in em_dir.glob("*"):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        image_paths.append(img_p)
                        labels.append(mapped_lbl)
                        splits.append(split_name)
                        original_labels.append(orig_lbl)

    if not image_paths:
        csv_candidates = [
            raw_fer_dir / "fer2013.csv",
            data_dir / "fer2013.csv",
        ]
        csv_file = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            print(f"Error: FER2013 image directory or CSV file not found under {data_dir}")
            sys.exit(1)

        df_raw = pd.read_csv(csv_file)
        print(f"Loaded FER2013 CSV with {len(df_raw)} rows.")

        usage_to_split = {
            "Training": "train",
            "PublicTest": "val",
            "PrivateTest": "test",
        }

        temp_img_dir = data_dir / "processed" / "fer2013" / "tmp_raw"
        temp_img_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df_raw.iterrows():
            orig_lbl = int(row["emotion"])
            mapped_lbl = map_label("fer2013", orig_lbl)
            usage = str(row["Usage"])
            split = usage_to_split.get(usage, "train")

            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            rgb = np.stack([pixels, pixels, pixels], axis=2)

            tmp_path = temp_img_dir / f"raw_{idx:06d}.jpg"
            Image.fromarray(rgb).save(tmp_path)

            image_paths.append(tmp_path)
            labels.append(mapped_lbl)
            splits.append(split)
            original_labels.append(orig_lbl)

    print(f"Found {len(image_paths)} valid FER2013 images to process.")

    preprocessor = Preprocessor(
        target_size=target_size, bypass_face_detection=bypass_face_detection
    )
    output_dir = data_dir / "processed" / "fer2013"
    df = preprocessor.process_dataset(
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        dataset_name="fer2013",
        output_dir=output_dir,
        original_labels=original_labels,
    )
    preprocessor.close()

    temp_img_dir = data_dir / "processed" / "fer2013" / "tmp_raw"
    if temp_img_dir.exists():
        shutil.rmtree(temp_img_dir, ignore_errors=True)

    print(f"FER2013 preprocessing complete. Output: {output_dir}")


def preprocess_affectnet(data_dir: Path, target_size: int, bypass_face_detection: bool):
    """Preprocess AffectNet dataset recursively across any directory layout."""
    image_paths = []
    labels = []
    splits = []
    original_labels = []

    folder_to_label = {
        "anger": 0, "angry": 0, "0_anger": 0, "0": 0,
        "disgust": 1, "disgusted": 1, "1_disgust": 1, "1": 1,
        "fear": 2, "fearful": 2, "2_fear": 2, "2": 2,
        "happy": 3, "happiness": 3, "3_happy": 3, "3": 3,
        "sad": 4, "sadness": 4, "4_sad": 4, "4": 4,
        "surprise": 5, "surprised": 5, "5_surprise": 5, "5": 5,
        "neutral": 6, "6_neutral": 6, "6": 6,
    }

    raw_aff_dir = data_dir / "raw" / "affectnet"
    if not raw_aff_dir.exists():
        raw_aff_dir = data_dir / "affectnet"

    # 1. Search for image directories recursively anywhere under raw_aff_dir
    all_img_files = list(raw_aff_dir.glob("**/*.jpg")) + list(raw_aff_dir.glob("**/*.png")) + list(raw_aff_dir.glob("**/*.jpeg"))

    for img_p in all_img_files:
        parent_name = img_p.parent.name.lower()
        path_parts = [p.lower() for p in img_p.parts]

        # Check if parent or any path part matches emotion label
        found_label = None
        for part in path_parts:
            if part in folder_to_label:
                found_label = folder_to_label[part]
                break

        if found_label is not None:
            if is_excluded_label("affectnet", found_label):
                continue
            mapped_lbl = map_label("affectnet", found_label)
            split_name = "val" if ("val" in path_parts or "test" in path_parts) else "train"

            image_paths.append(img_p)
            labels.append(mapped_lbl)
            splits.append(split_name)
            original_labels.append(found_label)

    # 2. Fallback to CSV annotations if folder scan found no images
    if not image_paths:
        csv_files = list(raw_aff_dir.glob("**/*.csv"))
        for csv_file in csv_files:
            split = "val" if ("val" in csv_file.name.lower() or "test" in csv_file.name.lower()) else "train"
            try:
                df_raw = pd.read_csv(csv_file)
                img_base = csv_file.parent
                for _, row in df_raw.iterrows():
                    expr_col = "expression" if "expression" in row else ("label" if "label" in row else None)
                    if expr_col is None or expr_col not in row:
                        continue
                    try:
                        orig_lbl = int(row[expr_col])
                    except (ValueError, TypeError):
                        continue

                    if is_excluded_label("affectnet", orig_lbl):
                        continue

                    mapped_lbl = map_label("affectnet", orig_lbl)
                    rel_file = str(row.get("subDirectory_filePath", row.get("filePath", row.get("image_path", ""))))
                    img_p = img_base / rel_file
                    if not img_p.exists():
                        img_p = raw_aff_dir / rel_file

                    if img_p.exists() and img_p.is_file():
                        image_paths.append(img_p)
                        labels.append(mapped_lbl)
                        splits.append(split)
                        original_labels.append(orig_lbl)
            except Exception as e:
                print(f"Warning: CSV parse error for {csv_file}: {e}")

    if not image_paths:
        print(f"Warning: No AffectNet images found under {raw_aff_dir}. Creating empty placeholder for pipeline safety.")
        output_dir = data_dir / "processed" / "affectnet"
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=["image_path", "original_label", "mapped_label", "dataset_source", "split"])
        empty_df.to_csv(output_dir / "metadata.csv", index=False)
        return

    print(f"Found {len(image_paths)} valid AffectNet images to process.")

    preprocessor = Preprocessor(
        target_size=target_size, bypass_face_detection=bypass_face_detection
    )
    output_dir = data_dir / "processed" / "affectnet"
    df = preprocessor.process_dataset(
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        dataset_name="affectnet",
        output_dir=output_dir,
        original_labels=original_labels,
    )
    preprocessor.close()
    print(f"AffectNet preprocessing complete. Output: {output_dir}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"Dataset to process: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Target image size: {args.target_size}x{args.target_size}")
    print(f"Bypass face detection: {args.bypass_face_detection}")

    if args.dataset == "rafdb":
        preprocess_rafdb(
            data_dir,
            target_size=args.target_size,
            bypass_face_detection=args.bypass_face_detection,
        )
    elif args.dataset == "fer2013":
        preprocess_fer2013(
            data_dir,
            target_size=args.target_size,
            bypass_face_detection=args.bypass_face_detection,
        )
    elif args.dataset == "affectnet":
        preprocess_affectnet(
            data_dir,
            target_size=args.target_size,
            bypass_face_detection=args.bypass_face_detection,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
