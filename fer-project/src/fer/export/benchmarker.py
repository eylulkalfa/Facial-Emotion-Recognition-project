"""Inference latency and model size benchmarking module."""

import logging
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import torch

from fer.models.fer_model import FERModel

logger = logging.getLogger(__name__)


def benchmark(
    pytorch_model: FERModel,
    onnx_path: str,
    input_size: int = 224,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark inference latency and model sizes for PyTorch vs ONNX Runtime.

    Args:
        pytorch_model: Instantiated PyTorch FERModel.
        onnx_path: Path to the exported .onnx model file.
        input_size: Image dimension size.
        num_warmup: Warmup iterations count.
        num_runs: Benchmark iterations count.

    Returns:
        Dict containing latency statistics and model sizes.
    """
    logger = logging.getLogger(__name__)

    # --- PyTorch Benchmark ---
    pytorch_model.eval()
    pt_device = torch.device("cpu")
    pytorch_model.to(pt_device)

    dummy_tensor = torch.randn(1, 3, input_size, input_size, device=pt_device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = pytorch_model(dummy_tensor)

    pt_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = pytorch_model(dummy_tensor)
            end = time.perf_counter()
            pt_times.append((end - start) * 1000.0)  # ms

    pt_mean = float(np.mean(pt_times))
    pt_std = float(np.std(pt_times))
    pt_p95 = float(np.percentile(pt_times, 95))

    # --- ONNX Runtime Benchmark ---
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    dummy_np = dummy_tensor.cpu().numpy()
    ort_input = {"input": dummy_np}

    for _ in range(num_warmup):
        _ = session.run(None, ort_input)

    ort_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, ort_input)
        end = time.perf_counter()
        ort_times.append((end - start) * 1000.0)  # ms

    ort_mean = float(np.mean(ort_times))
    ort_std = float(np.std(ort_times))
    ort_p95 = float(np.percentile(ort_times, 95))

    speedup = float(pt_mean / max(ort_mean, 1e-6))

    pt_size_mb = pytorch_model.get_model_size_mb()
    onnx_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    results = {
        "pytorch_mean_ms": pt_mean,
        "pytorch_std_ms": pt_std,
        "pytorch_p95_ms": pt_p95,
        "onnx_mean_ms": ort_mean,
        "onnx_std_ms": ort_std,
        "onnx_p95_ms": ort_p95,
        "speedup_ratio": speedup,
        "pytorch_size_mb": float(pt_size_mb),
        "onnx_size_mb": float(onnx_size_mb),
    }

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"PyTorch CPU Mean Latency:  {pt_mean:.2f} ms ± {pt_std:.2f} ms (p95: {pt_p95:.2f} ms)")
    logger.info(f"ONNX CPU Mean Latency:     {ort_mean:.2f} ms ± {ort_std:.2f} ms (p95: {ort_p95:.2f} ms)")
    logger.info(f"Speedup Ratio:             {speedup:.2f}x")
    logger.info(f"PyTorch Model Size:        {pt_size_mb:.2f} MB")
    logger.info(f"ONNX Model File Size:      {onnx_size_mb:.2f} MB")
    logger.info("=" * 60)

    return results
