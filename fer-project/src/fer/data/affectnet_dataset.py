"""AffectNet dataset adapter."""

from pathlib import Path
from typing import Optional

import albumentations as A
import pandas as pd

from fer.data.base_dataset import BaseFERDataset
from fer.data.label_mapping import is_excluded_label, map_label


class AffectNetDataset(BaseFERDataset):
    """AffectNet dataset adapter extending BaseFERDataset.

    Filters out Contempt (label 7) and maps remaining 7 expression classes.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[A.Compose] = None,
    ):
        """Initialize AffectNetDataset.

        Args:
            data_dir: Root data directory path.
            split: Data split ("train", "val", "test").
            transform: Albumentations transform pipeline.
        """
        super().__init__(data_dir, split, transform)

    def load_metadata(self) -> pd.DataFrame:
        """Parse AffectNet annotations and return standardized metadata DataFrame.

        Returns:
            pd.DataFrame containing image_path, original_label, mapped_label, etc.
        """
        processed_csv = self.data_dir / "processed" / "affectnet" / "metadata.csv"
        if processed_csv.exists():
            df = pd.read_csv(processed_csv)
            filtered = df[df["split"] == self.split].reset_index(drop=True)
            return filtered

        # Fallback: Read raw AffectNet CSVs
        csv_filename = "training.csv" if self.split == "train" else "validation.csv"

        csv_candidates = [
            self.data_dir / "raw" / "affectnet" / "labels" / csv_filename,
            self.data_dir / "raw" / "affectnet" / csv_filename,
            self.data_dir / "labels" / csv_filename,
            self.data_dir / csv_filename,
        ]

        csv_file = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            raise FileNotFoundError(
                f"AffectNet CSV file '{csv_filename}' not found under {self.data_dir}"
            )

        df_raw = pd.read_csv(csv_file)

        # Image subfolder prefix
        sub_folder = "train_set" if self.split == "train" else "val_set"

        rows = []
        for _, row in df_raw.iterrows():
            # Check expression column name (can be expression or label)
            expr_col = "expression" if "expression" in row else "label"
            if expr_col not in row:
                continue

            orig_label = int(row[expr_col])

            # CRITICAL: Filter out Contempt (label 7)
            if is_excluded_label("affectnet", orig_label):
                continue

            mapped_lbl = map_label("affectnet", orig_label)

            rel_file_path = str(row.get("subDirectory_filePath", row.get("filePath", "")))
            if not rel_file_path.startswith("raw/affectnet/"):
                rel_path = f"raw/affectnet/{sub_folder}/images/{rel_file_path}"
            else:
                rel_path = rel_file_path

            rows.append(
                {
                    "image_path": rel_path,
                    "original_label": orig_label,
                    "mapped_label": mapped_lbl,
                    "dataset_source": "affectnet",
                    "split": self.split,
                }
            )

        return pd.DataFrame(rows)
