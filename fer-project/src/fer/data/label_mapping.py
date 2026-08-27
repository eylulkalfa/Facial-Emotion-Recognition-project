"""Canonical 7-class emotion label space and dataset-specific mappings."""

from typing import Dict, List, Set

NUM_CLASSES: int = 7

EMOTION_NAMES: List[str] = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

LABEL_TO_IDX: Dict[str, int] = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

IDX_TO_LABEL: Dict[int, str] = {idx: name for name, idx in LABEL_TO_IDX.items()}

# RAF-DB: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
RAFDB_LABEL_MAP: Dict[int, int] = {
    1: 5,  # Surprise -> 5
    2: 2,  # Fear -> 2
    3: 1,  # Disgust -> 1
    4: 3,  # Happiness -> 3
    5: 4,  # Sadness -> 4
    6: 0,  # Anger -> 0
    7: 6,  # Neutral -> 6
}

# FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
FER2013_LABEL_MAP: Dict[int, int] = {i: i for i in range(7)}

# AffectNet: 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger, 7=Contempt
AFFECTNET_LABEL_MAP: Dict[int, int] = {
    0: 6,  # Neutral -> 6
    1: 3,  # Happy -> 3
    2: 4,  # Sad -> 4
    3: 5,  # Surprise -> 5
    4: 2,  # Fear -> 2
    5: 1,  # Disgust -> 1
    6: 0,  # Anger -> 0
}

AFFECTNET_EXCLUDED_LABELS: Set[int] = {7}  # Contempt is excluded

DATASET_LABEL_MAPS: Dict[str, Dict[int, int]] = {
    "rafdb": RAFDB_LABEL_MAP,
    "fer2013": FER2013_LABEL_MAP,
    "affectnet": AFFECTNET_LABEL_MAP,
}

DATASET_EXCLUDED_LABELS: Dict[str, Set[int]] = {
    "rafdb": set(),
    "fer2013": set(),
    "affectnet": AFFECTNET_EXCLUDED_LABELS,
}


def map_label(dataset_name: str, original_label: int) -> int:
    """Map a dataset's original label to canonical 0-6 index.

    Args:
        dataset_name: Name of dataset ("rafdb", "fer2013", "affectnet").
        original_label: Raw integer label from dataset.

    Returns:
        Canonical label index [0-6].
    """
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_LABEL_MAPS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(DATASET_LABEL_MAPS.keys())}"
        )

    label_map = DATASET_LABEL_MAPS[dataset_key]
    if original_label not in label_map:
        raise ValueError(
            f"Unknown label {original_label} for dataset {dataset_name}."
        )

    return label_map[original_label]


def is_excluded_label(dataset_name: str, original_label: int) -> bool:
    """Check if a label should be filtered out (e.g. AffectNet Contempt=7).

    Args:
        dataset_name: Name of dataset.
        original_label: Raw integer label from dataset.

    Returns:
        True if label is excluded, False otherwise.
    """
    dataset_key = dataset_name.lower()
    excluded = DATASET_EXCLUDED_LABELS.get(dataset_key, set())
    return original_label in excluded


def get_class_names() -> List[str]:
    """Return a copy of the canonical emotion class names."""
    return EMOTION_NAMES.copy()


def get_num_classes() -> int:
    """Return the total number of classes (7)."""
    return NUM_CLASSES
