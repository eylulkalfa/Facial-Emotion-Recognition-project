"""FER2013 dataset adapter."""

from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import torch

from fer.data.base_dataset import BaseFERDataset
from fer.data.label_mapping import map_label


class FER2013Dataset(BaseFERDataset):
    """FER2013 dataset adapter extending BaseFERDataset.

    Handles CSV pixel parsing and on-the-fly grayscale-to-RGB expansion.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[A.Compose] = None,
        csv_filename: str = "fer2013.csv",
    ):
        """Initialize FER2013Dataset.

        Args:
            data_dir: Root data directory path.
            split: Data split ("train", "val", "test").
            transform: Albumentations transform pipeline.
            csv_filename: Name of the FER2013 CSV file.
        """
        self.csv_filename = csv_filename
        self.has_processed_images = False
        super().__init__(data_dir, split, transform)

    def load_metadata(self) -> pd.DataFrame:
        """Parse FER2013 annotations and return standardized metadata DataFrame.

        Returns:
            pd.DataFrame containing image_path, original_label, mapped_label, etc.
        """
        processed_csv = self.data_dir / "processed" / "fer2013" / "metadata.csv"
        if processed_csv.exists():
            self.has_processed_images = True
            df = pd.read_csv(processed_csv)
            filtered = df[df["split"] == self.split].reset_index(drop=True)
            return filtered

        # Fallback: Read raw fer2013.csv
        csv_candidates = [
            self.data_dir / "raw" / "fer2013" / self.csv_filename,
            self.data_dir / self.csv_filename,
        ]

        csv_file = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            raise FileNotFoundError(
                f"FER2013 CSV file '{self.csv_filename}' not found under {self.data_dir}"
            )

        df_raw = pd.read_csv(csv_file)

        usage_map = {
            "train": "Training",
            "val": "PublicTest",
            "test": "PrivateTest",
        }
        target_usage = usage_map.get(self.split, "Training")
        filtered_raw = df_raw[df_raw["Usage"] == target_usage].reset_index(drop=True)

        rows = []
        for idx, row in filtered_raw.iterrows():
            orig_label = int(row["emotion"])
            mapped_lbl = map_label("fer2013", orig_label)
            rows.append(
                {
                    "image_path": "",  # Empty path indicates raw pixels stored in DataFrame
                    "original_label": orig_label,
                    "mapped_label": mapped_lbl,
                    "dataset_source": "fer2013",
                    "split": self.split,
                    "pixels": row["pixels"],
                }
            )

        return pd.DataFrame(rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Override __getitem__ to handle pixel string parsing for raw CSV."""
        if self.has_processed_images:
            return super().__getitem__(idx)

        row = self.metadata.iloc[idx]
        pixel_str = row["pixels"]
        label = int(row["mapped_label"])

        # Parse 48x48 pixel values string
        pixels = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        rgb = np.stack([pixels, pixels, pixels], axis=2)  # (48, 48, 3)

        # Resize to 224x224
        img_pil = Image.fromarray(rgb)
        resized_pil = img_pil.resize((224, 224), Image.LANCZOS)
        resized_np = np.array(resized_pil, dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=resized_np)
            image_out = augmented["image"]
        else:
            image_out = resized_np

        if isinstance(image_out, np.ndarray):
            image_tensor = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0
        elif isinstance(image_out, torch.Tensor):
            image_tensor = image_out
        else:
            raise TypeError(f"Unexpected transform output type: {type(image_out)}")

        return image_tensor, label
