"""RAF-DB (Real-world Affective Faces Database) dataset adapter."""

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split

from fer.data.base_dataset import BaseFERDataset
from fer.data.label_mapping import map_label


class RAFDBDataset(BaseFERDataset):
    """RAF-DB dataset adapter extending BaseFERDataset.

    Handles RAF-DB aligned images and official train/test splits.
    Generates a stratified validation split from official training data.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[A.Compose] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize RAFDBDataset.

        Args:
            data_dir: Root data directory path.
            split: Data split ("train", "val", "test").
            transform: Albumentations transform pipeline.
            val_ratio: Proportion of training data to use for validation split.
            seed: Random seed for stratified validation split.
        """
        self.val_ratio = val_ratio
        self.seed = seed
        super().__init__(data_dir, split, transform)

    def load_metadata(self) -> pd.DataFrame:
        """Parse RAF-DB annotations and return standardized metadata DataFrame.

        Returns:
            pd.DataFrame containing image_path, original_label, mapped_label, etc.
        """
        # Check if preprocessed metadata CSV exists
        processed_csv = self.data_dir / "processed" / "rafdb" / "metadata.csv"
        if processed_csv.exists():
            df = pd.read_csv(processed_csv)

            if self.split == "test":
                return df[df["split"] == "test"].reset_index(drop=True)

            # Split official train into train and val splits
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            if len(train_df) > 1:
                try:
                    tr_df, val_df = train_test_split(
                        train_df,
                        test_size=self.val_ratio,
                        random_state=self.seed,
                        stratify=train_df["mapped_label"],
                    )
                except ValueError:
                    # Fallback non-stratified if a class has <2 samples
                    tr_df, val_df = train_test_split(
                        train_df,
                        test_size=self.val_ratio,
                        random_state=self.seed,
                    )
            else:
                tr_df, val_df = train_df, train_df

            if self.split == "train":
                return tr_df.reset_index(drop=True)
            else:
                return val_df.reset_index(drop=True)

        # Fallback: Parse raw RAF-DB files
        anno_candidates = [
            self.data_dir / "raw" / "rafdb" / "EmoLabel" / "list_patition_label.txt",
            self.data_dir / "raw" / "rafdb" / "list_patition_label.txt",
            self.data_dir / "EmoLabel" / "list_patition_label.txt",
            self.data_dir / "list_patition_label.txt",
        ]

        anno_file = None
        for candidate in anno_candidates:
            if candidate.exists():
                anno_file = candidate
                break

        if anno_file is None:
            raise FileNotFoundError(
                f"RAF-DB annotation file 'list_patition_label.txt' not found under {self.data_dir}"
            )

        items: List[Tuple[str, int]] = []
        with open(anno_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) == 2:
                    filename, raw_label_str = parts[0], parts[1]
                    try:
                        raw_label = int(raw_label_str)
                        items.append((filename, raw_label))
                    except ValueError:
                        continue

        img_dir_candidates = [
            self.data_dir / "raw" / "rafdb" / "Image" / "aligned",
            self.data_dir / "Image" / "aligned",
            self.data_dir / "aligned",
        ]

        image_base_rel = "raw/rafdb/Image/aligned"
        for candidate in img_dir_candidates:
            if candidate.exists():
                image_base_rel = str(candidate.relative_to(self.data_dir))
                break

        train_items = [it for it in items if it[0].startswith("train_")]
        test_items = [it for it in items if it[0].startswith("test_")]

        if train_items:
            filenames, labels = zip(*train_items)
            try:
                tr_files, val_files, tr_labs, val_labs = train_test_split(
                    filenames,
                    labels,
                    test_size=self.val_ratio,
                    random_state=self.seed,
                    stratify=labels,
                )
            except ValueError:
                tr_files, val_files, tr_labs, val_labs = train_test_split(
                    filenames,
                    labels,
                    test_size=self.val_ratio,
                    random_state=self.seed,
                )
            tr_split = list(zip(tr_files, tr_labs))
            val_split = list(zip(val_files, val_labs))
        else:
            tr_split, val_split = [], []

        if self.split == "train":
            selected_items = tr_split
        elif self.split == "val":
            selected_items = val_split
        elif self.split == "test":
            selected_items = test_items
        else:
            raise ValueError(f"Invalid split: {self.split}")

        rows = []
        for filename, orig_label in selected_items:
            clean_name = filename
            if not clean_name.endswith(".jpg"):
                clean_name = f"{clean_name}.jpg"

            mapped_label = map_label("rafdb", orig_label)
            rel_path = f"{image_base_rel}/{clean_name}"

            rows.append(
                {
                    "image_path": rel_path,
                    "original_label": orig_label,
                    "mapped_label": mapped_label,
                    "dataset_source": "rafdb",
                    "split": self.split,
                }
            )

        return pd.DataFrame(rows)
