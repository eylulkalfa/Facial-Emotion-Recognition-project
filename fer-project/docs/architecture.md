# Architecture Overview

## Project Architecture

The FER project follows a clean 8-module architecture:

```
fer-project/
├── configs/              # YAML configuration files
├── docs/                 # System documentation
├── scripts/              # CLI entry points
├── src/fer/
│   ├── data/            # Dataset adapters, face detection, preprocessing, transforms
│   ├── models/          # Backbone factory, classification head, FERModel
│   ├── training/        # Multi-phase trainer, loss functions, metrics, callbacks
│   ├── evaluation/      # Evaluator, temperature calibration, visualization
│   ├── export/          # ONNX exporter, verifier, latency benchmarker
│   ├── inference/       # ONNX predictor, Gradio web demo
│   └── utils/           # Logging, seeding, device detection, I/O
└── tests/               # Unit and integration test suite
```

## Key Technical Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Backbone Library** | `timm` | Provides pretrained MobileNetV3, EfficientNet-B0, MobileViT-XS |
| **Output Normalization** | `BackboneWrapper` | Normalizes 2D/3D/4D outputs to consistent 2D `[B, feature_dim]` |
| **Classification Head** | `Dropout -> Linear` | Architecture-independent, outputs raw logits |
| **Loss Functions** | Weighted CE / Focal Loss | Solves class imbalance in FER datasets |
| **Primary Metric** | `Macro-F1` | Evaluates multi-class imbalance performance fairly |
| **Calibration** | Temperature Scaling | Learns scalar $T$ on validation set to minimize ECE |
| **Export Format** | ONNX (opset 17+) | Device-agnostic deployment with baked temperature & softmax |
| **Model Size Limit** | `< 100 MB` | FP32 parameter footprint constraint |

## Canonical 7-Class Label Space

| Index | Emotion | RAF-DB Raw | FER2013 Raw | AffectNet Raw |
|-------|---------|------------|-------------|---------------|
| 0 | Anger | 6 | 0 | 6 |
| 1 | Disgust | 3 | 1 | 5 |
| 2 | Fear | 2 | 2 | 4 |
| 3 | Happiness | 4 | 3 | 1 |
| 4 | Sadness | 5 | 4 | 2 |
| 5 | Surprise | 1 | 5 | 3 |
| 6 | Neutral | 7 | 6 | 0 |

*(AffectNet Contempt=7 is filtered out during preprocessing)*
