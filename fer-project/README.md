# Facial Emotion Recognition (FER) & Deployment Optimization

A PyTorch-based lightweight Facial Emotion Recognition system trained on RAF-DB, FER2013, and AffectNet datasets, with probability calibration and ONNX export for edge deployment.

## Features

- **Multi-Backbone Support**: MobileNetV3-Large, EfficientNet-B0, and MobileViT-XS via `timm` (< 100MB constraint).
- **Canonical 7-Class Space**: Standardized `anger, disgust, fear, happiness, sadness, surprise, neutral` across all datasets.
- **Class-Imbalance Handling**: Inverse-frequency Weighted Cross Entropy and Focal Loss ($\gamma=2.0$).
- **Multi-Phase Training**: Head warmup, full backbone training, and fine-tuning with AMP and TensorBoard logging.
- **Probability Calibration**: Temperature scaling to minimize Expected Calibration Error (ECE).
- **ONNX Deployment**: Export pipeline with temperature baking, numerical verification, and latency benchmarking.
- **Interactive Web Demo**: Gradio application with webcam support and ONNX Runtime inference.

## Quick Start

```bash
# 1. Setup virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run unit & integration tests
pytest tests/ -v

# 3. Train a model
python scripts/train.py --config configs/base.yaml configs/mobilenetv3.yaml

# 4. Evaluate & Calibrate
python scripts/evaluate.py --config configs/base.yaml configs/mobilenetv3.yaml --checkpoint experiments/<run>/checkpoints/best_model.pt --calibrate

# 5. Export to ONNX
python scripts/export_onnx.py --config configs/base.yaml configs/mobilenetv3.yaml --checkpoint experiments/<run>/checkpoints/best_model.pt --output exports/model.onnx --temperature 1.15

# 6. Launch Web Demo
python scripts/demo.py --model exports/model.onnx
```

## Documentation

- 📖 [Setup & Installation Guide](docs/setup.md)
- 🧪 [Experiment & Evaluation Guide](docs/experiments.md)
- 📐 [Architecture Overview](docs/architecture.md)

## License

MIT License
