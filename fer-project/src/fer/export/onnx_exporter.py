"""ONNX exporter with temperature scaling baking and model simplification."""

import logging
from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F

from fer.config import ExportConfig
from fer.models.fer_model import FERModel

logger = logging.getLogger(__name__)


class ONNXExportModel(nn.Module):
    """Wrapper that bakes temperature scaling and softmax into the exported model.

    This ensures the exported ONNX model directly outputs calibrated probability distributions.
    """

    def __init__(self, model: FERModel, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting calibrated probabilities."""
        logits = self.model(x)
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=1)


def export_to_onnx(
    model: FERModel,
    output_path: str,
    config: Optional[ExportConfig] = None,
    temperature: float = 1.0,
    input_size: int = 224,
) -> str:
    """Export FERModel to ONNX format with simplification and verification."""
    logger = logging.getLogger(__name__)

    opset_version = config.opset_version if config else 17
    simplify = config.simplify if config else True
    dynamic_batch = config.dynamic_batch if config else True

    export_model = ONNXExportModel(model, temperature=temperature)
    export_model.to("cpu")
    export_model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, device="cpu")

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to ONNX (opset {opset_version})...")
    torch.onnx.export(
        export_model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    logger.info(f"Exported raw ONNX model to {out_p}")

    # Validate ONNX model with built-in checker
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX structural validation passed.")

    # Simplify ONNX graph if configured
    if simplify:
        logger.info("Simplifying ONNX graph using onnx-simplifier...")
        try:
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simp, str(out_p))
                logger.info("ONNX graph simplification successful.")
            else:
                logger.warning("ONNX simplification check failed, preserving unsimplified graph.")
        except Exception as e:
            logger.warning(f"ONNX simplification skipped due to error: {e}")

    file_size_mb = out_p.stat().st_size / (1024 * 1024)
    logger.info(f"Exported ONNX file size: {file_size_mb:.2f} MB")
    return str(out_p.resolve())
