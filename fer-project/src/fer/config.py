"""Configuration system for Facial Emotion Recognition (FER) project.

Defines nested dataclasses and loading/saving utilities using dacite and PyYAML.
"""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dacite
import yaml


@dataclass
class PhaseConfig:
    name: str
    epochs: int
    lr: float
    freeze_backbone: bool


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    seed: int = 42
    output_dir: str = "./experiments"


@dataclass
class DataConfig:
    dataset: str = "rafdb"
    data_dir: str = "./data"
    input_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    bypass_face_detection: bool = True
    balance_classes: bool = True


@dataclass
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    rotation_limit: int = 10
    rotation_prob: float = 0.3
    random_resized_crop_scale: List[float] = field(default_factory=lambda: [0.92, 1.0])
    color_jitter_prob: float = 0.25
    color_jitter_brightness: float = 0.15
    color_jitter_contrast: float = 0.15
    color_jitter_hue: float = 0.02
    random_erasing_prob: float = 0.1
    random_erasing_scale: List[float] = field(default_factory=lambda: [0.02, 0.10])
    mixup_alpha: float = 0.0


@dataclass
class ModelConfig:
    backbone: str = "mobilenetv3_large_100"
    pretrained: bool = True
    num_classes: int = 7
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    phases: List[PhaseConfig] = field(
        default_factory=lambda: [
            PhaseConfig(name="head_warmup", epochs=5, lr=0.001, freeze_backbone=True),
            PhaseConfig(name="full_training", epochs=20, lr=0.0001, freeze_backbone=False),
            PhaseConfig(name="fine_tune", epochs=10, lr=0.00003, freeze_backbone=False),
        ]
    )
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    loss: str = "weighted_ce"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    label_smoothing: float = 0.0
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    mixed_precision: bool = True


@dataclass
class ExportConfig:
    opset_version: int = 17
    simplify: bool = True
    dynamic_batch: bool = True


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dictionary into base dictionary.

    Lists in override completely replace lists in base.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(*config_paths: str) -> ProjectConfig:
    """Load one or more YAML config files and parse into a ProjectConfig instance.

    Multiple configs are merged left-to-right (subsequent configs override earlier ones).
    """
    if not config_paths:
        raise ValueError("At least one config path must be provided.")

    merged_dict: Dict[str, Any] = {}
    for path_str in config_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path_str}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            merged_dict = _deep_merge(merged_dict, data)

    config_obj = dacite.from_dict(
        data_class=ProjectConfig,
        data=merged_dict,
        config=dacite.Config(strict=False),
    )
    return config_obj


def save_config(config: ProjectConfig, path: str) -> None:
    """Serialize a ProjectConfig instance to a YAML file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
