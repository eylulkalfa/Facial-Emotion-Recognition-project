# 🎭 Facial Emotion Recognition (FER) & Deployment Optimization

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005BA1?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Web_UI-FF5500?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end, production-ready **Facial Emotion Recognition (FER)** system optimized for real-time CPU and edge inference. Features **MobileNetV3-Large** deep backbone, **OpenCV YuNet DNN** facial cropping, **Focal Loss** for class imbalance, **Temperature Scaling** probability calibration, and an **11.48x accelerated ONNX Runtime** deployment pipeline.

---

## 🚀 Key Highlights & Performance Benchmark

| Parameter | Specification | Achievement | Status |
| :--- | :--- | :--- | :---: |
| **Emotion Classes** | Minimum 5 categories | **7 Canonical Classes** (`Anger`, `Disgust`, `Fear`, `Happiness`, `Sadness`, `Surprise`, `Neutral`) | `PASSED` |
| **Model Size Limit** | Under **100 MB** | **16.05 MB** (MobileNetV3 FP32) — *Only 16% of budget* | `PASSED` |
| **Inference Latency** | Real-time CPU target | **3.32 ms / 300+ FPS** on CPU (*11.48x ONNX speedup*) | `PASSED` |
| **Model Verification** | PyTorch vs ONNX matching | **Max Abs Diff = $1.49 \times 10^{-6}$** (< $10^{-5}$ tolerance) | `PASSED` |
| **Calibration Error** | Low Expected Calibration Error | **ECE reduced from 10.25% down to 1.96%** ($T=1.7819$) | `PASSED` |
| **Dataset Scale** | Combined academic sources | **51,226 facial images** (RAF-DB + FER2013 + AffectNet integration) | `PASSED` |

---

## 📊 SOTA Architecture Comparison

We evaluated four State-of-the-Art architectures under identical data and constraint settings:

| Model Architecture | Dataset | Accuracy (%) | Macro-F1 | Weighted-F1 | ROC-AUC | Model Size | CPU Latency | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **RAF-DB + FER2013** | **74.92%** | **0.6850** | **0.7514** | **0.9376** | **16.05 MB** | **3.32 ms** | **🏆 SELECTED MODEL** |
| **ConvNeXt-Tiny** | RAF-DB + FER2013 | 73.91% | 0.6977 | 0.7250 | 0.9140 | ~114 MB | 18.45 ms | Eliminated (Heavy) |
| **EfficientNet-B0** | RAF-DB + FER2013 | 70.91% | 0.6634 | 0.6935 | 0.9126 | 16.50 MB | 8.12 ms | Eliminated (Slower) |
| **ViT-Base-Patch16** | RAF-DB + FER2013 | 43.12% | 0.3587 | 0.4528 | 0.7621 | ~343 MB | 85.30 ms | ❌ Eliminated (Overfitted) |

> **Why MobileNetV3-Large?** Inverted Residual Blocks, Hard-Swish activations, and Squeeze-and-Excitation (SE) attention modules allowed MobileNetV3 to achieve top accuracy while maintaining sub-17MB footprint and 3.32ms latency.

---

## ⚡ ONNX Runtime Acceleration

```
 +────────────────────────+     torch.onnx.export      +────────────────────────+
 │ Native PyTorch FP32    │ ─────────────────────────► │ Exported ONNX FP32     │
 │ Model (MobileNetV3)    │   opset_version = 17       │ (mobilenetv3.onnx)     │
 +────────────────────────+   dynamic_axes = {batch}   +────────────────────────+
             │                                                     │
             ▼                                                     ▼
 +────────────────────────+                            +────────────────────────+
 │ PyTorch CPU Execution  │                            │ ONNX Runtime CPU Exec  │
 │ (38.15 ms / 26.21 FPS) │                            │ (3.32 ms / 300.77 FPS) │
 +────────────────────────+                            +────────────────────────+
```

---

## 🛠️ Data Pipeline & Engineering

1. **Biometric Face Crop (OpenCV YuNet DNN):** Standard central crops carry background, clothing, and hair noise. Integrating OpenCV YuNet DNN (`cv2.FaceDetectorYN`) extracts 224x224 square face crops focused strictly on facial landmarks (eyebrows, eyes, nose, mouth).
2. **Class-Imbalance Handling:** Combined `WeightedRandomSampler` with **Focal Loss ($\gamma=2.0$)** to prevent minority classes (`Disgust`, `Fear`) from being suppressed by dominant classes (`Happiness`, `Neutral`).
3. **Probability Calibration:** Raw neural network logits suffer from overconfidence. Post-hoc **Temperature Scaling ($T=1.7819$)** reduced Expected Calibration Error (ECE) from **10.25% down to 1.96%**.

---

## 💻 Quick Start & Demo Instructions

### 1. Environment Setup
```bash
git clone https://github.com/eylulkalfa/Facial-Emotion-Recognition-project.git
cd Facial-Emotion-Recognition-project/fer-project

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Launch Web Application Demo
```bash
# Launch interactive Gradio Web UI with pre-trained ONNX model
python scripts/demo.py --model exports/mobilenetv3.onnx --port 7860
```
Open **`http://127.0.0.1:7860`** in your browser for live file upload or webcam snapshot prediction!

### 3. Run Evaluation & Calibration
```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config experiments/100k_veriseti/mobilenetv3_large_100_rafdb_20260824_181109/mobilenetv3_large_100_rafdb_20260824_181109/config.yaml \
  --checkpoint experiments/100k_veriseti/mobilenetv3_large_100_rafdb_20260824_181109/mobilenetv3_large_100_rafdb_20260824_181109/checkpoints/best_model.pt \
  --calibrate
```

### 4. Export to ONNX
```bash
PYTHONPATH=src python scripts/export_onnx.py \
  --config configs/base.yaml configs/mobilenetv3.yaml \
  --checkpoint experiments/100k_veriseti/mobilenetv3_large_100_rafdb_20260824_181109/mobilenetv3_large_100_rafdb_20260824_181109/checkpoints/best_model.pt \
  --output exports/mobilenetv3.onnx \
  --temperature 1.7819
```

---

## 📁 Repository Structure

```
.
├── deliverables/                # 📑 OFFICIAL PROJECT DELIVERABLES
│   ├── A_RD_Report/             #  Deliverable A: Research & Development Report
│   ├── B_ONNX_Report/           #  Deliverable B: ONNX Conversion & Benchmark Report
│   ├── C_Demo/                  #  Deliverable C: Demo Application Guide
│   ├── D_Presentation/          #  Deliverable D: Final Technical Presentation Deck
│   └── Optimized_Real_Time_Emotion_AI_(3).pdf  # 📊 Final Slide Deck Presentation PDF
│
├── docs/                        # 📚 Literature reviews & reference task specs
│   ├── references/              # Original task specs & Turkish literature reviews
│   └── FER_Literatur_Taramasi.md
│
└── fer-project/                 # ⚙️ MAIN PYTHON CODEBASE
    ├── configs/                 # YAML configuration files
    ├── exports/                 # Exported ONNX model & benchmark JSONs
    │   └── mobilenetv3.onnx     # Production ONNX model (16.05 MB)
    ├── experiments/             # Experiment runs (100k & 50k dataset runs)
    ├── scripts/                 # CLI scripts (train, evaluate, export_onnx, demo)
    ├── src/fer/                 # Modular Python package (models, data, eval, export)
    └── tests/                   # PyTest unit & integration test suite
```

---

## 📜 Deliverables Summary

- **Deliverable A:** [Research & Development Report](deliverables/A_RD_Report/RD_Report.md)
- **Deliverable B:** [ONNX Conversion & Performance Comparison Report](deliverables/B_ONNX_Report/ONNX_Report.md)
- **Deliverable C:** [Demo Application Guide](deliverables/C_Demo/Demo_Guide.md)
- **Deliverable D:** [Final Technical Presentation Deck](deliverables/D_Presentation/Presentation_Deck.md) & [PDF Deck](deliverables/Optimized_Real_Time_Emotion_AI_(3).pdf)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
