"""Numerical verification module comparing PyTorch and ONNX Runtime predictions."""

import logging
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import torch

from fer.models.fer_model import FERModel

logger = logging.getLogger(__name__)


def verify_onnx(
    pytorch_model: FERModel,
    onnx_path: str,
    temperature: float = 1.0,
    num_samples: int = 10,
    tolerance: float = 1e-4,
    input_size: int = 224,
) -> Dict[str, Any]:
    """Compare PyTorch model outputs with ONNX Runtime outputs.

    Args:
        pytorch_model: Trained PyTorch FERModel instance.
        onnx_path: Path to the exported .onnx model file.
        temperature: Calibration temperature used during ONNX export.
        num_samples: Number of random test inputs to compare.
        tolerance: Maximum allowed absolute difference tolerance.
        input_size: Input image dimension.

    Returns:
        Dict with verification results (passed, max_abs_diff, mean_abs_diff, etc.).
    """
    logger = logging.getLogger(__name__)

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    temp_val = max(float(temperature), 1e-6)

    all_diffs = []
    max_abs_diff = 0.0

    for _ in range(num_samples):
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # PyTorch prediction (logits / T -> softmax)
        with torch.no_grad():
            pt_logits = pytorch_model(dummy_input.to(device))
            pt_scaled = pt_logits / temp_val
            pt_probs = torch.softmax(pt_scaled, dim=1).cpu().numpy()

        # ONNX Runtime prediction
        ort_input = {"input": dummy_input.numpy()}
        ort_probs = session.run(None, ort_input)[0]

        diff = float(np.abs(pt_probs - ort_probs).max())
        all_diffs.append(diff)
        max_abs_diff = max(max_abs_diff, diff)

    mean_abs_diff = float(np.mean(all_diffs))
    passed = bool(max_abs_diff < tolerance)

    results = {
        "passed": passed,
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": mean_abs_diff,
        "num_samples_tested": num_samples,
        "all_diffs": all_diffs,
    }

    if passed:
        logger.info(
            f"ONNX numerical verification PASSED (max_diff={max_abs_diff:.2e} < {tolerance:.2e})"
        )
    else:
        logger.error(
            f"ONNX numerical verification FAILED (max_diff={max_abs_diff:.2e} >= {tolerance:.2e})"
        )

    return results
