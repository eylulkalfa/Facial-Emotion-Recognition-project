"""Random seed management for reproducibility."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across all libraries for deterministic behavior.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


class WorkerInitFn:
    """Top-level callable object for DataLoader worker initialization.

    Avoids unpicklable nested closure errors in multiprocessing.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_worker_init_fn(seed: int) -> WorkerInitFn:
    """Returns a WorkerInitFn object for DataLoader to seed worker processes.

    Args:
        seed: Base random seed.

    Returns:
        WorkerInitFn callable instance.
    """
    return WorkerInitFn(seed)
