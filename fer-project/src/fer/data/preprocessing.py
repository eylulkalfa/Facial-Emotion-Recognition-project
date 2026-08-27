"""Preprocessing pipeline for offline image processing and metadata generation."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from fer.data.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessor pipeline that crops faces, resizes images, and builds metadata."""

    def __init__(
        self,
        target_size: int = 224,
        bypass_face_detection: bool = True,
    ):
        """Initialize Preprocessor.

        Args:
            target_size: Output image height and width in pixels.
            bypass_face_detection: Whether to skip face detection.
        """
        self.target_size = target_size
        self.face_detector = FaceDetector(bypass=bypass_face_detection)
        self.logger = logging.getLogger(__name__)

    def process_single_image(
        self, image_path: Union[str, Path]
    ) -> Optional[np.ndarray]:
        """Load, detect face, crop, and resize a single image to (target_size, target_size) RGB.

        Args:
            image_path: Path to the raw image file.

        Returns:
            Processed RGB numpy array of shape (target_size, target_size, 3) uint8,
            or None if processing fails.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                self.logger.warning(f"File not found: {image_path}")
                return None

            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil, dtype=np.uint8)

            # Face detection and cropping
            cropped_face = self.face_detector.detect_and_crop(img_np)

            # Resize with LANCZOS resampling
            face_pil = Image.fromarray(cropped_face)
            resized_pil = face_pil.resize(
                (self.target_size, self.target_size), Image.LANCZOS
            )
            resized_np = np.array(resized_pil, dtype=np.uint8)

            return resized_np

        except Exception as e:
            self.logger.warning(f"Failed to process image {image_path}: {e}")
            return None

    def compute_image_hash(self, image_path: Union[str, Path]) -> str:
        """Compute MD5 hash of raw image file for duplicate detection.

        Args:
            image_path: Path to the image file.

        Returns:
            32-character hexadecimal MD5 hash string.
        """
        path = Path(image_path)
        with open(path, "rb") as f:
            data = f.read()
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def is_grayscale(image: np.ndarray) -> bool:
        """Check if an image is originally grayscale.

        Args:
            image: Image numpy array.

        Returns:
            True if image is 2D or has identical RGB channels.
        """
        if image.ndim == 2:
            return True
        if image.shape[2] == 1:
            return True
        return bool(np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2]))

    def process_dataset(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        splits: List[str],
        dataset_name: str,
        output_dir: Union[str, Path],
        original_labels: Optional[List[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        """Process a batch of dataset images, save to disk as JPEG, and create metadata CSV.

        Args:
            image_paths: Raw image file paths.
            labels: Canonical integer labels [0-6].
            splits: Split names ("train", "val", "test").
            dataset_name: Dataset name identifier ("rafdb", "fer2013", etc.).
            output_dir: Output root directory for processed files.
            original_labels: Optional raw unmapped labels.

        Returns:
            pd.DataFrame with processed metadata matching schema.
        """
        out_root = Path(output_dir)
        for split_name in {"train", "val", "test"}:
            (out_root / split_name).mkdir(parents=True, exist_ok=True)

        rows = []
        if original_labels is None:
            original_labels = labels

        for idx, (raw_path, label, split, orig_label) in enumerate(
            tqdm(
                zip(image_paths, labels, splits, original_labels),
                total=len(image_paths),
                desc=f"Processing {dataset_name}",
            )
        ):
            processed_img = self.process_single_image(raw_path)
            if processed_img is None:
                continue

            filename = f"{dataset_name}_{idx:06d}.jpg"
            rel_save_path = Path("processed") / dataset_name / split / filename
            abs_save_path = out_root / split / filename

            # Save JPEG using PIL
            Image.fromarray(processed_img).save(abs_save_path, "JPEG", quality=95)

            img_hash = self.compute_image_hash(raw_path)
            is_gray = self.is_grayscale(processed_img)

            rows.append(
                {
                    "image_path": str(rel_save_path),
                    "original_label": orig_label,
                    "mapped_label": label,
                    "dataset_source": dataset_name,
                    "split": split,
                    "image_hash": img_hash,
                    "is_grayscale_origin": is_gray,
                }
            )

        df = pd.DataFrame(rows)
        csv_path = out_root / "metadata.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved dataset metadata with {len(df)} samples to {csv_path}")

        return df

    def close(self) -> None:
        """Close face detector resources."""
        self.face_detector.close()
