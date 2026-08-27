# Experiment Guide

## 1. Training a Model

Train with default base configuration (MobileNetV3-Large on RAF-DB):

```bash
python scripts/train.py --config configs/base.yaml
```

Train with a specific backbone override:

```bash
# EfficientNet-B0
python scripts/train.py --config configs/base.yaml configs/efficientnet_b0.yaml

# MobileViT-XS
python scripts/train.py --config configs/base.yaml configs/mobilevit_xs.yaml
```

Override random seed:

```bash
python scripts/train.py --config configs/base.yaml --seed 123
```

## 2. Monitoring Training

Launch TensorBoard to monitor live metrics, loss curves, and learning rate schedules:

```bash
tensorboard --logdir experiments/
```

## 3. Evaluating & Calibrating

Run evaluation and temperature scaling calibration on the test set:

```bash
python scripts/evaluate.py \
  --config configs/base.yaml configs/mobilenetv3.yaml \
  --checkpoint experiments/mobilenetv3_rafdb_20260806_120000/checkpoints/best_model.pt \
  --split test \
  --calibrate
```

Generates:
- `metrics.json`
- `confusion_matrix.png`
- `reliability_diagram.png`
- `per_class_f1.png`
- `roc_curves.png`

## 4. Exporting to ONNX

Export trained PyTorch model to ONNX format with baked calibration temperature:

```bash
python scripts/export_onnx.py \
  --config configs/base.yaml configs/mobilenetv3.yaml \
  --checkpoint experiments/mobilenetv3_rafdb_20260806_120000/checkpoints/best_model.pt \
  --output exports/mobilenetv3.onnx \
  --temperature 1.15
```

Runs automatic numerical verification (`tolerance < 1e-4`) and latency benchmarking.

## 5. Comparing Backbones

Generate a comparison report across all backbone experiments:

```bash
python scripts/compare_backbones.py \
  --experiment-dirs experiments/mobilenetv3_* experiments/efficientnet_b0_* experiments/mobilevit_xs_* \
  --output experiments/backbone_comparison.csv
```

## 6. Running the Interactive Demo

Launch the Gradio web interface with webcam support:

```bash
python scripts/demo.py --model exports/mobilenetv3.onnx --port 7860
```
