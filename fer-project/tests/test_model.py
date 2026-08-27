import pytest
import torch

from fer.models.fer_model import FERModel


def test_forward_shape():
    model = FERModel(backbone_name="mobilenetv3_large_100", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_forward_with_softmax():
    model = FERModel(backbone_name="mobilenetv3_large_100", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    probs = model.forward_with_softmax(x)
    assert probs.shape == (2, 7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_freeze_unfreeze():
    model = FERModel(backbone_name="mobilenetv3_large_100", pretrained=False)
    total = model.get_total_params()
    model.freeze_backbone()
    frozen = model.get_trainable_params()
    assert frozen < total
    model.unfreeze_backbone()
    assert model.get_trainable_params() == total


def test_model_size_constraint():
    for name in ["mobilenetv3_large_100", "efficientnet_b0", "mobilevit_xs"]:
        model = FERModel(backbone_name=name, pretrained=False)
        assert model.get_model_size_mb() < 100


def test_all_backbones_output_shape():
    for name in ["mobilenetv3_large_100", "efficientnet_b0", "mobilevit_xs"]:
        model = FERModel(backbone_name=name, pretrained=False)
        out = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 7)
