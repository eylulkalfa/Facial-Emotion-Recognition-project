"""CLI script for comparing experiment results across different backbone architectures."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Experiment Results Across Backbones"
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to experiment directories to compare.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output comparison CSV file.",
    )
    return parser.parse_args()


def load_json_safe(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    args = parse_args()
    rows = []

    for exp_dir_str in args.experiment_dirs:
        exp_dir = Path(exp_dir_str)
        if not exp_dir.exists():
            print(f"Warning: Directory does not exist: {exp_dir_str}")
            continue

        cfg_file = exp_dir / "config.yaml"
        backbone_name = "unknown"
        dataset_name = "unknown"
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                    backbone_name = cfg_dict.get("model", {}).get(
                        "backbone", "unknown"
                    )
                    dataset_name = cfg_dict.get("data", {}).get("dataset", "unknown")
            except Exception:
                pass

        best_metrics = load_json_safe(exp_dir / "results" / "best_metrics.json")
        eval_metrics = load_json_safe(exp_dir / "results" / "metrics.json")
        bench_data = load_json_safe(exp_dir / "benchmark_summary.json")

        macro_f1 = best_metrics.get("macro_f1", eval_metrics.get("macro_f1", "N/A"))
        accuracy = best_metrics.get("accuracy", eval_metrics.get("accuracy", "N/A"))
        ece = eval_metrics.get(
            "ece_calibrated", eval_metrics.get("ece_uncalibrated", "N/A")
        )

        bench_info = bench_data.get("benchmark", {})
        pt_latency = bench_info.get("pytorch_mean_ms", "N/A")
        onnx_latency = bench_info.get("onnx_mean_ms", "N/A")
        onnx_size = bench_info.get("onnx_size_mb", "N/A")

        rows.append(
            {
                "Experiment": exp_dir.name,
                "Backbone": backbone_name,
                "Dataset": dataset_name,
                "Macro-F1": (
                    f"{macro_f1:.4f}" if isinstance(macro_f1, float) else macro_f1
                ),
                "Accuracy": (
                    f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy
                ),
                "ECE": f"{ece:.4f}" if isinstance(ece, float) else ece,
                "PT Latency (ms)": (
                    f"{pt_latency:.2f}"
                    if isinstance(pt_latency, float)
                    else pt_latency
                ),
                "ONNX Latency (ms)": (
                    f"{onnx_latency:.2f}"
                    if isinstance(onnx_latency, float)
                    else onnx_latency
                ),
                "ONNX Size (MB)": (
                    f"{onnx_size:.2f}"
                    if isinstance(onnx_size, float)
                    else onnx_size
                ),
            }
        )

    if not rows:
        print("No valid experiment directories found.")
        return 1

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("BACKBONE EXPERIMENT COMPARISON REPORT")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"Saved comparison report to {out_p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
