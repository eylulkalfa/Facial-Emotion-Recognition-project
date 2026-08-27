import numpy as np
import pytest
import torch

from fer.config import AugmentationConfig
from fer.data.transforms import get_train_transforms, get_val_transforms


def test_train_transform_output_shape():
    aug_config = AugmentationConfig()
    tf = get_train_transforms(aug_config, 224)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = tf(image=image)["image"]
    assert result.shape == (3, 224, 224)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_val_transform_output_shape():
    tf = get_val_transforms(224)
    image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    result = tf(image=image)["image"]
    assert result.shape == (3, 224, 224)


def test_val_transform_no_randomness():
    tf = get_val_transforms(224)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    r1 = tf(image=image)["image"]
    r2 = tf(image=image)["image"]
    assert torch.equal(r1, r2)


def test_train_transform_produces_variation():
    aug_config = AugmentationConfig()
    tf = get_train_transforms(aug_config, 224)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    results = [tf(image=image)["image"] for _ in range(10)]
    all_same = all(torch.equal(results[0], r) for r in results[1:])
    assert not all_same
