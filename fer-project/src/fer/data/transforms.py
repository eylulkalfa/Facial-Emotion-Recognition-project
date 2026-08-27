"""Albumentations image augmentation and normalization transform pipelines."""

from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fer.config import AugmentationConfig

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_train_transforms(
    config: AugmentationConfig, input_size: int = 224
) -> A.Compose:
    """Create training transform pipeline with albumentations augmentations.

    Args:
        config: AugmentationConfig containing transform probabilities and limits.
        input_size: Target image height and width.

    Returns:
        albumentations.Compose pipeline.
    """
    min_hole = max(1, int(input_size * 0.05))
    max_hole = max(1, int(input_size * 0.15))

    return A.Compose(
        [
            A.HorizontalFlip(p=config.horizontal_flip_prob),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=config.rotation_limit,
                p=config.rotation_prob,
                border_mode=0,  # BORDER_CONSTANT (fill with black)
            ),
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=tuple(config.random_resized_crop_scale),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=0.1,
                hue=config.color_jitter_hue,
                p=config.color_jitter_prob,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(min_hole, max_hole),
                hole_width_range=(min_hole, max_hole),
                fill=0,
                p=config.random_erasing_prob,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(input_size: int = 224) -> A.Compose:
    """Create validation/testing transform pipeline (no augmentation).

    Args:
        input_size: Target image height and width.

    Returns:
        albumentations.Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=input_size, width=input_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
