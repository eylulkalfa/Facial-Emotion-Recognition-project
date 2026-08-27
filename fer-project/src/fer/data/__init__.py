"""Data loading, preprocessing, and augmentation for FER datasets."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from fer.config import ProjectConfig
from fer.data.affectnet_dataset import AffectNetDataset
from fer.data.combined_dataset import CombinedFERDataset
from fer.data.fer2013_dataset import FER2013Dataset
from fer.data.rafdb_dataset import RAFDBDataset
from fer.data.transforms import get_train_transforms, get_val_transforms
from fer.utils.seeding import get_worker_init_fn

_DATASET_REGISTRY = {
    "rafdb": RAFDBDataset,
    "fer2013": FER2013Dataset,
    "affectnet": AffectNetDataset,
    "combined": CombinedFERDataset,
}


def create_dataloaders(
    config: ProjectConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, val, and test DataLoaders from project config.

    Args:
        config: ProjectConfig instance.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader may be None if the dataset split fails to load.
    """
    dataset_name = config.data.dataset.lower()
    dataset_cls = _DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset: '{config.data.dataset}'. "
            f"Available datasets: {list(_DATASET_REGISTRY.keys())}"
        )

    train_transform = get_train_transforms(
        config.augmentation, config.data.input_size
    )
    val_transform = get_val_transforms(config.data.input_size)

    train_dataset = dataset_cls(
        config.data.data_dir, split="train", transform=train_transform
    )
    val_dataset = dataset_cls(
        config.data.data_dir, split="val", transform=val_transform
    )

    test_dataset = None
    try:
        test_dataset = dataset_cls(
            config.data.data_dir, split="test", transform=val_transform
        )
    except Exception:
        pass

    # Balanced sampling for training loader if configured
    sampler = None
    if config.data.balance_classes:
        labels = train_dataset.get_labels()
        if len(labels) > 0:
            class_counts = np.bincount(labels, minlength=7)
            class_counts = np.maximum(class_counts, 1)
            weights_per_class = 1.0 / class_counts
            sample_weights = weights_per_class[labels]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )

    worker_init_fn = get_worker_init_fn(config.experiment.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=sampler if config.data.balance_classes else None,
        shuffle=(not config.data.balance_classes),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

    return train_loader, val_loader, test_loader
