import pytest
from fer.data.label_mapping import (
    NUM_CLASSES,
    EMOTION_NAMES,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    map_label,
    is_excluded_label,
)


def test_num_classes():
    assert NUM_CLASSES == 7


def test_emotion_names_order():
    assert EMOTION_NAMES == [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]


def test_label_to_idx_consistency():
    for name, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == name


def test_rafdb_label_mapping():
    assert map_label("rafdb", 1) == 5  # Surprise
    assert map_label("rafdb", 2) == 2  # Fear
    assert map_label("rafdb", 3) == 1  # Disgust
    assert map_label("rafdb", 4) == 3  # Happiness
    assert map_label("rafdb", 5) == 4  # Sadness
    assert map_label("rafdb", 6) == 0  # Anger
    assert map_label("rafdb", 7) == 6  # Neutral


def test_fer2013_identity_mapping():
    for i in range(7):
        assert map_label("fer2013", i) == i


def test_affectnet_label_mapping():
    assert map_label("affectnet", 0) == 6  # Neutral
    assert map_label("affectnet", 1) == 3  # Happy
    assert map_label("affectnet", 6) == 0  # Anger


def test_affectnet_contempt_excluded():
    assert is_excluded_label("affectnet", 7) is True
    assert is_excluded_label("affectnet", 0) is False


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        map_label("unknown", 0)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        map_label("rafdb", 99)
