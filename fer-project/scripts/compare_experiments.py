"""Compare and log evaluation metrics across all trained model experiments."""

import argparse
import glob
import json
from pathlib import Path
import sys
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and Compare Model Experiment Results"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Path to experiments directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Path to output export directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.experiments_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print(f"Error: Experiments directory '{exp_dir}' not found.")
        sys.exit(1)

    records = []
    # Search for all best_metrics.json files in experiments directory
    metric_files = sorted(exp_dir.glob("**/best_metrics.json"))
    if not metric_files:
        metric_files = sorted(exp_dir.glob("**/metrics.json"))

    print(f"Found {len(metric_files)} experiment metric files to analyze.")

    for mf in metric_files:
        exp_folder = mf.parent.parent
        exp_name = exp_folder.name

        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            acc = data.get("accuracy", 0.0)
            macro_f1 = data.get("macro_f1", 0.0)
            weighted_f1 = data.get("weighted_f1", 0.0)
            roc_auc = data.get("roc_auc", 0.0)

            # Extract per-class F1 scores if available
            per_class_f1 = data.get("per_class_f1", {})

            record = {
                "experiment_id": exp_name,
                "accuracy": round(acc * 100, 2),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "roc_auc": round(roc_auc, 4),
                "f1_happy": round(per_class_f1.get("happiness", 0.0), 4),
                "f1_sad": round(per_class_f1.get("sadness", 0.0), 4),
                "f1_surprise": round(per_class_f1.get("surprise", 0.0), 4),
                "f1_anger": round(per_class_f1.get("anger", 0.0), 4),
                "f1_disgust": round(per_class_f1.get("disgust", 0.0), 4),
                "f1_fear": round(per_class_f1.get("fear", 0.0), 4),
                "f1_neutral": round(per_class_f1.get("neutral", 0.0), 4),
                "metric_file": str(mf),
            }
            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to read {mf}: {e}")

    if not records:
        print("No valid experiment records found to compare.")
        return 0

    df = pd.DataFrame(records)
    # Sort by Accuracy descending
    df = df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # Save to JSON and CSV in exports directory
    json_path = out_dir / "experiments_comparison.json"
    csv_path = out_dir / "experiments_comparison.csv"

    df.to_json(json_path, orient="records", indent=2)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("📊 EXPERIMENT COMPARISON SUMMARY TABLE")
    print("=" * 80)
    summary_cols = ["experiment_id", "accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    print(df[summary_cols].to_string(index=False))
    print("=" * 80)
    print(f"\nSaved JSON summary: {json_path}")
    print(f"Saved CSV summary:  {csv_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
