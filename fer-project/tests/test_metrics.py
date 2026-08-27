import pytest
import torch

from fer.training.metrics import MetricTracker


def test_metric_tracker_basic():
    tracker = MetricTracker()
    tracker.update(torch.randn(10, 7), torch.randint(0, 7, (10,)))
    metrics = tracker.compute()
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["confusion_matrix"].shape == (7, 7)


def test_metric_tracker_perfect():
    tracker = MetricTracker()
    logits = torch.zeros(7, 7)
    for i in range(7):
        logits[i, i] = 10.0
    tracker.update(logits, torch.arange(7))
    metrics = tracker.compute()
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_metric_tracker_reset():
    tracker = MetricTracker()
    tracker.update(torch.randn(5, 7), torch.randint(0, 7, (5,)))
    tracker.reset()
    tracker.update(torch.randn(3, 7), torch.randint(0, 7, (3,)))
    metrics = tracker.compute()
    assert metrics["confusion_matrix"].sum() == 3


def test_per_class_metrics():
    tracker = MetricTracker()
    tracker.update(torch.randn(20, 7), torch.randint(0, 7, (20,)))
    metrics = tracker.compute()
    assert len(metrics["per_class_f1"]) == 7
    assert len(metrics["per_class_precision"]) == 7
    assert len(metrics["per_class_recall"]) == 7
    for name in [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]:
        assert name in metrics["per_class_f1"]
