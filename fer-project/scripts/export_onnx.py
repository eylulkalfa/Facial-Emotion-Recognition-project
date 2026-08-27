"""CLI entry point for exporting trained FER models to ONNX format with verification and benchmarking."""

import argparse
import logging
from pathlib import Path
import sys

import torch

from fer.config import load_config
from fer.export.benchmarker import benchmark
from fer.export.onnx_exporter import export_to_onnx
from fer.export.onnx_verifier import verify_onnx
from fer.models.fer_model import FERModel
from fer.utils.io import save_json
from fer.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Trained FER Model to ONNX Format"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more YAML config files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch .pt model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Target path for output .onnx file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Learned calibration temperature scaling parameter to bake into export.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip numerical verification between PyTorch and ONNX Runtime.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip latency benchmarking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("fer_export")

    logger.info(f"Loading config: {args.config}")
    config = load_config(*args.config)

    logger.info(f"Creating FERModel ({config.model.backbone}) and loading checkpoint: {args.checkpoint}")
    model = FERModel(config=config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Exporting to ONNX at {args.output} with temperature T={args.temperature:.4f}...")
    onnx_path = export_to_onnx(
        model=model,
        output_path=args.output,
        config=config.export,
        temperature=args.temperature,
        input_size=config.data.input_size,
    )

    verify_results = None
    if not args.skip_verify:
        logger.info("Verifying ONNX numerical equivalence...")
        verify_results = verify_onnx(
            pytorch_model=model,
            onnx_path=onnx_path,
            temperature=args.temperature,
            input_size=config.data.input_size,
        )

    bench_results = None
    if not args.skip_benchmark:
        logger.info("Running latency benchmark...")
        bench_results = benchmark(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_size=config.data.input_size,
        )

    out_json = Path(onnx_path).parent / "benchmark_summary.json"
    summary_data = {
        "onnx_path": onnx_path,
        "temperature": args.temperature,
        "verification": verify_results,
        "benchmark": bench_results,
    }
    save_json(summary_data, out_json)
    logger.info(f"Saved export summary to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
