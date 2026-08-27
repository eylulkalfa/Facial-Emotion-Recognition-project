import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fer.evaluation.evaluator import Evaluator
from fer.export.onnx_exporter import export_to_onnx
from fer.export.onnx_verifier import verify_onnx
from fer.models.fer_model import FERModel
from fer.training.losses import create_loss


class TestFullPipeline:
    """End-to-end integration test verifying the full pipeline on synthetic data."""

    def _create_synthetic_loader(self, num_samples: int = 16, batch_size: int = 4):
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 7, (num_samples,))
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_end_to_end_pipeline(self):
        """Full pipeline: Model -> Train Step -> Evaluation -> ONNX Export -> Verification -> ONNX Inference."""
        model = FERModel(backbone_name="mobilenetv3_large_100", pretrained=False)
        train_loader = self._create_synthetic_loader(16, batch_size=4)
        val_loader = self._create_synthetic_loader(8, batch_size=4)

        # 1. Train Step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = create_loss("weighted_ce")

        images, targets = next(iter(train_loader))
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0.0

        # 2. Evaluation
        evaluator = Evaluator(model)
        metrics = evaluator.evaluate(val_loader)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

        # 3. ONNX Export
        tmp_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmp_dir, "test_pipeline.onnx")
        export_to_onnx(model, onnx_path, temperature=1.0)
        assert os.path.exists(onnx_path)

        # 4. ONNX Verification
        results = verify_onnx(
            model, onnx_path, temperature=1.0, num_samples=3
        )
        assert results["passed"] is True

        # 5. ONNX Runtime Inference
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        ort_out = session.run(None, {"input": dummy_np})[0]
        assert ort_out.shape == (1, 7)
        assert abs(float(ort_out.sum()) - 1.0) < 1e-3

    def test_model_size_constraint_all_backbones(self):
        for name in ["mobilenetv3_large_100", "efficientnet_b0", "mobilevit_xs"]:
            m = FERModel(backbone_name=name, pretrained=False)
            assert m.get_model_size_mb() < 100.0
