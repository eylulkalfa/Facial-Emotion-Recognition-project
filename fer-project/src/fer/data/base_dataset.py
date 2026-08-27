"""Abstract base class for Facial Emotion Recognition (FER) datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from fer.data.label_mapping import NUM_CLASSES


class BaseFERDataset(Dataset, ABC):
    """Abstract base dataset for FER datasets.

    Subclasses must implement `load_metadata()` to return a DataFrame containing
    at least 'image_path' and 'mapped_label' columns.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[A.Compose] = None,
    ):
        """Initialize BaseFERDataset.

        Args:
            data_dir: Base directory path for datasets.
            split: Data split ("train", "val", "test").
            transform: Albumentations Compose transform pipeline.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'val', 'test'.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.metadata = self.load_metadata()

        # Validate metadata schema
        required_cols = {"image_path", "mapped_label"}
        if not required_cols.issubset(set(self.metadata.columns)):
            raise ValueError(
                f"Metadata DataFrame must contain required columns {required_cols}. "
                f"Got: {list(self.metadata.columns)}"
            )

        # Validate label range
        self.labels = np.array(self.metadata["mapped_label"].values, dtype=np.int64)
        if len(self.labels) > 0:
            if self.labels.min() < 0 or self.labels.max() >= NUM_CLASSES:
                raise ValueError(
                    f"Label values must be in range [0, {NUM_CLASSES - 1}]. "
                    f"Found range [{self.labels.min()}, {self.labels.max()}]."
                )

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata and return a DataFrame.

        Must contain 'image_path' and 'mapped_label' columns.

        Returns:
            pd.DataFrame with metadata.
        """
        pass

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        image_path = self.data_dir / row["image_path"]

        img_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(img_pil, dtype=np.uint8)  # (H, W, 3)

        label = int(row["mapped_label"])

        if self.transform is not None:
            augmented = self.transform(image=image_np)
            image_out = augmented["image"]
        else:
            image_out = image_np

        if isinstance(image_out, np.ndarray):
            # Manual conversion if transform didn't return a torch.Tensor
            image_tensor = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0
        elif isinstance(image_out, torch.Tensor):
            image_tensor = image_out
        else:
            raise TypeError(f"Unexpected image type from transform: {type(image_out)}")

        return image_tensor, label

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array."""
        return self.labels.copy()

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function balancing.

        Returns:
            torch.FloatTensor of shape (NUM_CLASSES,).
        """
        total_samples = len(self.labels)
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)

        weights = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i in range(NUM_CLASSES):
            if counts[i] > 0:
                weights[i] = total_samples / (NUM_CLASSES * counts[i])
            else:
                weights[i] = 1.0

        return torch.from_numpy(weights).float()

    def get_class_distribution(self) -> dict:
        """Return dict mapping class index [0-6] to sample count."""
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        return {i: int(counts[i]) for i in range(NUM_CLASSES)}
