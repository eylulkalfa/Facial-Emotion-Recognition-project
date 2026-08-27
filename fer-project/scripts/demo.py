"""CLI script for launching Gradio web demo."""

import argparse
from pathlib import Path
import sys

from fer.inference.demo import create_demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Gradio Web Demo for Facial Emotion Recognition"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to exported .onnx model file.",
    )
    parser.add_argument(
        "--bypass-face-detection",
        action="store_true",
        help="Skip face detection during inference.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio shareable link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port number for Gradio application.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_p = Path(args.model)
    if not model_p.exists():
        print(f"Error: ONNX model file not found at {args.model}")
        sys.exit(1)

    print(f"Launching Gradio demo with model: {args.model}")
    print(f"Bypass face detection: {args.bypass_face_detection}")
    print(f"Server port: {args.port}")

    demo = create_demo(
        model_path=str(model_p),
        bypass_face_detection=args.bypass_face_detection,
    )
    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
