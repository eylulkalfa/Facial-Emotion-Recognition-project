# Facial Emotion Recognition (FER) — Complete Architecture & Implementation Blueprint

> **Document Type**: Technical Design Document  
> **Role**: Principal ML Engineer, Software Architect, Technical Lead  
> **Project Timeline**: 3 weeks  
> **Status**: Ready for Review

---

## Document References

This blueprint was designed after careful analysis of three workspace documents:

1. **Project Task Specification** (primary source of truth) — defines scope, constraints, deliverables, and weekly milestones
2. **Backbone Selection Literature Review** (Turkish) — evaluates 9 backbone candidates against project constraints
3. **Dataset, Preprocessing & Training Strategy Literature Review** (Turkish) — covers datasets, augmentation, training recipes, and evaluation

> [!IMPORTANT]
> Where any conflict exists between the literature reviews and the task specification, **the task specification takes precedence**. Trade-offs are explicitly noted throughout.

---

## Key Architectural Decisions (Summary)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Emotion classes | 7 (anger, disgust, fear, happiness, sadness, surprise, neutral) | Aligns with all major FER benchmarks; the task spec's minimum of 5 is exceeded |
| Framework | PyTorch only | Task spec allows both; PyTorch chosen for timm ecosystem, research flexibility |
| Backbones | MobileNetV3-Large (primary), EfficientNet-B0 (secondary), MobileViT-XS (experimental) | Literature review recommendation; all well under 100MB, strong ONNX support |
| Input resolution | 224×224 RGB | Matches ImageNet pretrained weights directly; no resolution mismatch |
| Config system | Plain YAML + Python dataclasses | Simpler than Hydra; appropriate for 3-week scope |
| Experiment tracking | TensorBoard | Zero external dependency; offline; sufficient for this project |
| Augmentation | albumentations | Faster than torchvision, richer transform set, reproducible |
| Demo | Gradio | Standard for ML demos; fast to implement |
| Dataset priority | RAF-DB first → FER2013 → AffectNet (incremental) | Common interface from day one; implement adapters incrementally |
| Face detection | Configurable bypass — MediaPipe/RetinaFace for raw data, bypass for pre-aligned | Flexible; most FER datasets already provide cropped faces |
| Primary metric | Macro-F1 | Fair across imbalanced classes; literature recommendation |
| Calibration | Temperature scaling + ECE | Task spec requires probability output — probabilities must be meaningful |

---

## 1. Project Architecture

### 1.1 Software Architecture Diagram

```
                                +─────────────────────────────+
                                │        Config Module        │
                                │  (YAML parsing, validation, │
                                │   dataclass mapping)        │
                                +─────────────────────────────+
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
               +────────────────────+  +────────────────+  +────────────────+
               │    Data Module     │  │  Models Module  │  │  Utils Module  │
               │ ─────────────────  │  │ ──────────────  │  │ ───────────────│
               │ • Dataset classes  │  │ • timm backbones│  │ • Logging      │
               │ • Preprocessing    │  │ • Custom heads  │  │ • Seeding      │
               │ • Face detection   │  │ • Model factory │  │ • Device mgmt  │
               │ • Augmentation     │  │ • FER model     │  │ • I/O helpers  │
               │ • Label mapping    │  │                 │  │                │
               │ • DataLoaders      │  │                 │  │                │
               +────────────────────+  +────────────────+  +────────────────+
                       │                       │                    │
                       ▼                       ▼                    │
               +───────────────────────────────────────────+        │
               │              Training Module              │◄───────┘
               │ ─────────────────────────────────────────  │
               │ • Trainer (multi-phase loop)               │
               │ • Optimizer (AdamW) + Scheduler            │
               │ • Loss functions (WCE, Focal)              │
               │ • MetricTracker                            │
               │ • Callbacks (Checkpointer, EarlyStopping)  │
               │ • Mixed precision (torch.amp)              │
               │ • TensorBoard logging                      │
               +───────────────────────────────────────────+
                                    │
                                    ▼
               +───────────────────────────────────────────+
               │            Evaluation Module              │
               │ ─────────────────────────────────────────  │
               │ • Evaluator (test set evaluation)          │
               │ • Calibration (Temperature scaling, ECE)   │
               │ • Visualization (confusion matrix, ROC,    │
               │   reliability diagrams, F1 bar charts)     │
               +───────────────────────────────────────────+
                                    │
                                    ▼
               +───────────────────────────────────────────+
               │              Export Module                 │
               │ ─────────────────────────────────────────  │
               │ • ONNX exporter (torch.onnx.export)        │
               │ • ONNX verifier (numerical comparison)     │
               │ • Benchmarker (latency, file size)         │
               +───────────────────────────────────────────+
                                    │
                                    ▼
               +───────────────────────────────────────────+
               │            Inference Module               │
               │ ─────────────────────────────────────────  │
               │ • Predictor (ONNX Runtime wrapper)         │
               │ • Gradio demo (image/webcam → bar chart)   │
               +───────────────────────────────────────────+
```

### 1.2 Module Responsibilities

| Module | Responsibility | Depends On |
|--------|---------------|------------|
| **Config** | Load YAML, validate via dataclasses, distribute to all modules | — |
| **Utils** | Logging, reproducibility (seeding), device management, I/O | Config |
| **Data** | Dataset classes, preprocessing, face detection, augmentation, label mapping, DataLoaders | Config, Utils |
| **Models** | timm backbone wrappers, classification heads, model factory, freeze/unfreeze | Config |
| **Training** | Multi-phase training loop, optimizer, scheduler, losses, metrics, checkpointing, early stopping, AMP, TensorBoard | Data, Models, Utils |
| **Evaluation** | Test evaluation, calibration (temperature scaling), visualization (plots) | Training outputs, Data |
| **Export** | ONNX conversion, numerical verification, latency benchmarking | Models, Evaluation |
| **Inference** | ONNX Runtime inference, preprocessing, Gradio demo | Export, Data |

### 1.3 Dependency Flow

```
Config → Utils → Data → Models → Training → Evaluation → Export → Inference
```

Each module depends only on those to its left. No circular dependencies. Scripts in `scripts/` are thin entry points that wire modules together.

---

## 2. Repository Structure

```
fer-project/
├── configs/                          # YAML configuration files
│   ├── base.yaml                     # Shared default configuration
│   ├── mobilenetv3.yaml              # MobileNetV3-Large overrides
│   ├── efficientnet_b0.yaml          # EfficientNet-B0 overrides
│   └── mobilevit_xs.yaml            # MobileViT-XS overrides
│
├── src/fer/                          # Main source package
│   ├── __init__.py
│   ├── config.py                     # Dataclass definitions + load/validate
│   │
│   ├── data/                         # Dataset & preprocessing
│   │   ├── __init__.py
│   │   ├── base_dataset.py           # Abstract base dataset (ABC)
│   │   ├── rafdb_dataset.py          # RAF-DB adapter
│   │   ├── fer2013_dataset.py        # FER2013 adapter
│   │   ├── affectnet_dataset.py      # AffectNet adapter
│   │   ├── preprocessing.py          # Face detection → alignment → crop → resize
│   │   ├── transforms.py            # albumentations train/val transforms
│   │   ├── face_detector.py          # MediaPipe/RetinaFace wrapper (bypass-capable)
│   │   └── label_mapping.py          # 7-class label space + mappings
│   │
│   ├── models/                       # Neural network architectures
│   │   ├── __init__.py
│   │   ├── backbone_factory.py       # timm backbone instantiation
│   │   ├── fer_model.py              # Backbone + Head combined model
│   │   └── heads.py                  # Classification head (pool → dropout → linear)
│   │
│   ├── training/                     # Training loop components
│   │   ├── __init__.py
│   │   ├── trainer.py                # Multi-phase training orchestrator
│   │   ├── losses.py                 # Weighted CE, Focal Loss, factory
│   │   ├── metrics.py                # MetricTracker (sklearn-based)
│   │   ├── callbacks.py              # EarlyStopping, ModelCheckpointer
│   │   └── optimizer.py              # Optimizer + scheduler creation
│   │
│   ├── evaluation/                   # Post-training evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Test set evaluation runner
│   │   ├── calibration.py            # Temperature scaling, ECE
│   │   └── visualization.py          # Confusion matrix, ROC, reliability diagrams
│   │
│   ├── export/                       # ONNX deployment
│   │   ├── __init__.py
│   │   ├── onnx_exporter.py          # torch.onnx.export + simplification
│   │   ├── onnx_verifier.py          # Numerical comparison PT vs ONNX
│   │   └── benchmarker.py            # Latency + file size benchmarking
│   │
│   ├── inference/                    # End-to-end inference
│   │   ├── __init__.py
│   │   ├── predictor.py              # ONNX Runtime inference wrapper
│   │   └── demo.py                   # Gradio demo definition
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── logging.py                # Console + file logging setup
│       ├── seeding.py                # seed_everything() for reproducibility
│       ├── device.py                 # CPU/CUDA/MPS detection
│       └── io.py                     # File I/O helpers
│
├── scripts/                          # CLI entry points
│   ├── train.py                      # Launch training
│   ├── evaluate.py                   # Run evaluation on test set
│   ├── export_onnx.py                # Export to ONNX + verify + benchmark
│   ├── demo.py                       # Launch Gradio demo
│   └── preprocess_dataset.py         # Offline dataset preprocessing
│
├── tests/                            # Unit and integration tests
│   ├── test_label_mapping.py
│   ├── test_transforms.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_pipeline.py             # Integration test (tiny data)
│
├── notebooks/                        # Jupyter notebooks for EDA
│   └── 01_data_exploration.ipynb
│
├── experiments/                      # Auto-generated experiment outputs (gitignored)
│   └── {backbone}_{dataset}_{timestamp}/
│       ├── config.yaml               # Frozen experiment config
│       ├── checkpoints/              # Model checkpoints
│       ├── logs/                     # TensorBoard event files
│       ├── results/                  # Evaluation outputs, plots, metrics JSON
│       └── exports/                  # ONNX files
│
├── data/                             # Dataset storage (gitignored)
│   ├── raw/                          # Downloaded raw datasets
│   │   ├── rafdb/
│   │   ├── fer2013/
│   │   └── affectnet/
│   └── processed/                    # Preprocessed 224×224 crops + metadata CSV
│       ├── rafdb/
│       ├── fer2013/
│       └── affectnet/
│
├── docs/                             # Documentation
│   ├── setup.md                      # Installation guide
│   ├── experiments.md                # How to run experiments
│   └── architecture.md              # Architecture overview
│
├── pyproject.toml                    # Project metadata + dependencies
├── requirements.txt                  # Pinned dependencies (generated)
├── .gitignore                        # Ignore data/, experiments/, .venv/, etc.
└── README.md                         # Project overview and quickstart
```

### Directory Purpose Summary

| Directory | Purpose |
|-----------|---------|
| `configs/` | Static YAML files. One base config + per-backbone overrides. Keeps experiments reproducible without code changes. |
| `src/fer/` | Core Python package. Modular design: each subpackage can be tested independently. |
| `scripts/` | Thin CLI wrappers — parse args, load config, call library code. Separates executable logic from reusable library. |
| `tests/` | Critical for catching label mapping bugs, shape mismatches, and metric computation errors. |
| `notebooks/` | EDA scratchpad: visualize bounding boxes, sample augmentations, analyze class distributions. Not production code. |
| `experiments/` | Output directory. Each run creates an isolated timestamped folder with all artifacts. Gitignored. |
| `data/` | Separated into `raw/` (untouched downloads) and `processed/` (standardized 224×224 crops). Gitignored. |
| `docs/` | Setup, experiment running, and architecture documentation. |

---

## 3. Configuration System

### 3.1 Design Decision

| Approach | Verdict | Reasoning |
|----------|---------|-----------|
| **Plain YAML + dataclasses** | ✅ Chosen | Simple, type-safe, no magic, easily serialized for reproducibility |
| Hydra | ❌ Rejected | Too complex for 3-week scope; steep learning curve, config directory proliferation |
| argparse only | ❌ Rejected | Doesn't scale for nested configs; hard to save/reproduce state |
| OmegaConf standalone | ❌ Rejected | Viable but adds dependency without clear benefit over plain dataclasses |

### 3.2 Configuration Structure

The config is divided into nested dataclasses:

- **`ExperimentConfig`** — name, seed, output directory
- **`DataConfig`** — dataset name, paths, batch size, num_workers, face detection toggle, input size
- **`AugmentationConfig`** — flip probability, rotation range, color jitter, random erasing, MixUp alpha
- **`ModelConfig`** — backbone name (timm), pretrained flag, num_classes, dropout rate
- **`TrainingConfig`** — phases (epochs, LR per phase), optimizer type, weight decay, loss type, class weights, early stopping patience, mixed precision toggle
- **`ExportConfig`** — ONNX opset version, dynamic axes, simplify flag

### 3.3 Example Base Config (`configs/base.yaml`)

```yaml
experiment:
  name: "mobilenetv3_rafdb"
  seed: 42
  output_dir: "./experiments"

data:
  dataset: "rafdb"
  data_dir: "./data"
  input_size: 224
  batch_size: 64
  num_workers: 4
  pin_memory: true
  bypass_face_detection: true    # RAF-DB is pre-aligned
  balance_classes: true

augmentation:
  horizontal_flip_prob: 0.5
  rotation_limit: 10
  rotation_prob: 0.3
  random_resized_crop_scale: [0.92, 1.0]
  color_jitter_prob: 0.25
  color_jitter_brightness: 0.15
  color_jitter_contrast: 0.15
  color_jitter_hue: 0.02
  random_erasing_prob: 0.1
  random_erasing_scale: [0.02, 0.10]
  mixup_alpha: 0.0              # 0.0 = disabled; set 0.2 for pre-training phase

model:
  backbone: "mobilenetv3_large_100"
  pretrained: true
  num_classes: 7
  dropout: 0.2

training:
  phases:
    - name: "head_warmup"
      epochs: 5
      lr: 0.001
      freeze_backbone: true
    - name: "full_training"
      epochs: 20
      lr: 0.0001
      freeze_backbone: false
    - name: "fine_tune"
      epochs: 10
      lr: 0.00003
      freeze_backbone: false
  optimizer: "adamw"
  weight_decay: 0.0001
  loss: "weighted_ce"           # or "focal"
  focal_gamma: 2.0
  focal_alpha: null              # auto-compute from class weights
  scheduler: "reduce_on_plateau"
  scheduler_factor: 0.5
  scheduler_patience: 4
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001
  mixed_precision: true

export:
  opset_version: 17
  simplify: true
  dynamic_batch: true
```

### 3.4 Config Workflow

1. **Load Base**: Script loads `configs/base.yaml`
2. **Merge Overrides**: If `--config configs/mobilenetv3.yaml` is provided, its values recursively override the base config
3. **CLI Overrides**: Additional `--key=value` arguments override specific fields (e.g., `--training.phases.0.epochs=3`)
4. **Parse & Validate**: Merged dict is unpacked into `ProjectConfig` dataclass. Type errors raise immediately.
5. **Save**: Before training starts, the final resolved config is serialized to `experiments/{run_name}/config.yaml` for exact reproducibility.

### 3.5 Per-Backbone Overrides

Each backbone YAML only contains fields that differ from base:

```yaml
# configs/efficientnet_b0.yaml
model:
  backbone: "efficientnet_b0"
```

```yaml
# configs/mobilevit_xs.yaml
model:
  backbone: "mobilevit_xs"
```

---

## 4. Dataset Pipeline

### 4.1 Data Flow Diagram

```
Raw Image (Any resolution, Grayscale or RGB)
       │
       ▼
┌──────────────────────┐
│ 1. RGB Conversion    │  (Grayscale → 3-channel RGB if needed)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐     bypass_face_detection = True
│ 2. Face Detection    │ ─────────────────────────────────┐
│ (MediaPipe/RetinaFace)│                                  │
└──────────────────────┘                                  │
       │ (bbox + landmarks)                               │
       ▼                                                  │
┌──────────────────────┐                                  │
│ 3. Alignment         │                                  │
│ (Rotate based on     │                                  │
│  eye landmarks)      │                                  │
└──────────────────────┘                                  │
       │                                                  │
       ▼                                                  │
┌──────────────────────┐                                  │
│ 4. Cropping          │ ◄────────────────────────────────┘
│ (Extract face bbox)  │   (pre-aligned datasets skip 2-3)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 5. Resize to 224×224 │
└──────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ SAVE to disk: data/processed/{dataset}/{split}/      │
│ UPDATE metadata CSV                                   │
└──────────────────────────────────────────────────────┘
       │
       ▼ (During training, via DataLoader)
┌──────────────────────┐
│ 6. Augmentation      │  (albumentations; TRAIN split only)
│ • HorizontalFlip(0.5)│
│ • Rotation(±10°)     │
│ • RandomResizedCrop  │
│ • ColorJitter        │
│ • RandomErasing      │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 7. Normalization     │  (ImageNet mean/std from timm)
└──────────────────────┘
       │
       ▼
   Model Input Tensor
   [B, 3, 224, 224]
```

### 4.2 Dataset Abstraction

**Base Interface (`BaseFERDataset`)** — abstract class inheriting `torch.utils.data.Dataset`:
- `load_metadata()` → reads standardized metadata CSV, returns DataFrame
- `__len__()` → number of samples
- `__getitem__(idx)` → loads preprocessed image, applies transforms, returns `(tensor, label_int)`
- `get_class_weights()` → returns tensor of inverse-frequency class weights for loss function
- `get_labels()` → returns all labels for sampler construction

**Per-Dataset Implementations:**

| Dataset | Adapter Class | Raw Format | Label Mapping | Official Splits |
|---------|--------------|------------|---------------|-----------------|
| RAF-DB | `RAFDBDataset` | Aligned images + txt annotations | 1-7 → 0-6 (surprise=1→5, fear=2→2, etc.) | Train/Test official |
| FER2013 | `FER2013Dataset` | CSV with pixel columns | 0-6 already aligned | Train/PublicTest/PrivateTest |
| AffectNet | `AffectNetDataset` | Images + CSV | 0-7 → drop contempt(7) | Train/Val official |

### 4.3 Unified Label Mapping

| Index | Emotion | Notes |
|-------|---------|-------|
| 0 | anger | |
| 1 | disgust | Minority class — monitor closely |
| 2 | fear | Minority class — monitor closely |
| 3 | happiness | Majority class in most datasets |
| 4 | sadness | |
| 5 | surprise | |
| 6 | neutral | Often second-largest class |

> [!IMPORTANT]
> Each dataset adapter must map its native labels to this canonical space. RAF-DB uses a different numeric ordering than FER2013 — this is a critical source of bugs if not handled centrally.

### 4.4 Metadata Schema

Stored at `data/processed/{dataset}/metadata.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | str | Relative path to preprocessed 224×224 image |
| `original_label` | str/int | Label from source dataset |
| `mapped_label` | int | Canonical 0-6 label |
| `dataset_source` | str | "rafdb", "fer2013", "affectnet" |
| `split` | str | "train", "val", "test" |
| `subject_id` | str/null | For identity-aware splitting (CK+, JAFFE) |
| `image_hash` | str | MD5 hash for near-duplicate detection |
| `is_grayscale_origin` | bool | Whether source was grayscale (converted to RGB) |

### 4.5 Split Handling

- **RAF-DB**: Official train/test split preserved. Val set: stratified 10% from training set (fixed seed).
- **FER2013**: Official train/PublicTest(val)/PrivateTest(test) preserved.
- **AffectNet**: Official train/val split preserved. Drop contempt class.
- **Near-duplicates**: MD5 hash comparison across datasets prevents cross-dataset leakage when merging data.

### 4.6 DataLoader Configuration

- `num_workers`: configurable (default 4)
- `pin_memory`: True (for GPU transfer)
- Balanced sampling: `WeightedRandomSampler` with inverse-frequency weights, applied to train loader only
- Val/Test loaders: no sampling, no augmentation

### 4.7 Extensibility

To add a new dataset:
1. Create `src/fer/data/newdataset_dataset.py` inheriting `BaseFERDataset`
2. Implement `load_metadata()` with the dataset's specific parsing logic
3. Define label mapping in `label_mapping.py`
4. Add preprocessing script entry to `preprocess_dataset.py`
5. No changes needed to training loop, model, or evaluation code

---

## 5. Training Pipeline

### 5.1 Trainer Module

The `Trainer` class orchestrates the complete training workflow:

**Responsibilities:**
- Execute the multi-phase epoch loop
- Manage optimizer step, gradient scaling (AMP), and scheduler updates
- Trigger `MetricTracker` after each batch/epoch
- Interface with `Checkpointer` and `EarlyStopping` callbacks
- Log to TensorBoard and console

**Multi-Phase Training:**

| Phase | Name | Backbone | LR | Epochs | Purpose |
|-------|------|----------|----|--------|---------|
| 1 | Head Warmup | Frozen | 1e-3 | 3-5 | Stabilize the new classification head on FER features |
| 2 | Full Training | Unfrozen | 1e-4 | 15-25 | Full representation learning on primary dataset |
| 3 | Fine-Tune | Unfrozen | 1e-5 to 3e-5 | 8-15 | Low-LR fine-tuning on target benchmark (RAF-DB) |

**Phase Transitions:** Automatic, based on epoch count from config. At each transition:
- Optimizer and scheduler are re-initialized with new LR
- Early stopping patience is reset
- Backbone layers are unfrozen (Phase 1→2)

> [!NOTE]
> **Trade-off with literature review**: The literature suggests AffectNet→RAF-DB sequential training across phases. For simplicity, initial implementation treats phases as LR/freeze stages within a single dataset. Multi-dataset sequential training can be added by chaining separate training runs using the checkpoint resume mechanism.

### 5.2 Validation Loop

Runs after every training epoch:
1. Set model to eval mode, disable gradients
2. Iterate validation DataLoader, accumulate logits + labels in `MetricTracker`
3. Compute all metrics (accuracy, F1s, confusion matrix, ROC-AUC)
4. Log to TensorBoard
5. Return metrics dict → Checkpointer, EarlyStopping, Scheduler

### 5.3 Metrics Module

`MetricTracker` class:
- **Accumulation**: Stores raw logits and ground truth labels across all batches in an epoch
- **Epoch Computation** (via scikit-learn):
  - Accuracy
  - Macro-F1 (**primary model selection metric**)
  - Weighted-F1
  - Per-class precision, recall, F1
  - Confusion matrix (normalized, row-wise)
  - ROC-AUC (one-vs-rest)
- Returns a flat dict: `{"accuracy": 0.85, "macro_f1": 0.78, ...}`

### 5.4 Loss Functions

| Loss | Use Case | Configuration |
|------|----------|---------------|
| **Weighted Cross Entropy** | Primary — handles class imbalance | Class weights auto-computed from training distribution |
| **Focal Loss** | Secondary — if WCE fails on disgust/fear | Configurable γ (default 2.0), α (auto from class weights) |

Selected via config: `training.loss: "weighted_ce"` or `training.loss: "focal"`

### 5.5 Checkpointing

**What is saved in each checkpoint:**
- Model `state_dict`
- Optimizer `state_dict`
- Scheduler `state_dict`
- Current epoch number
- Best validation metric (macro-F1)
- Full experiment config
- Random states (Python, NumPy, PyTorch, CUDA)

**Naming convention:**
- `best_model.pt` — highest macro-F1
- `last_model.pt` — most recent epoch
- `checkpoint_epoch_{N}.pt` — periodic saves (configurable interval)

**Top-K**: Keep the K best checkpoints by macro-F1 to manage disk space.

**Resume**: Load checkpoint → restore all states → continue from exact epoch.

### 5.6 Early Stopping

- **Monitor**: Validation macro-F1
- **Patience**: Configurable (default 8-10 epochs)
- **Min delta**: Configurable minimum improvement threshold
- **Phase interaction**: Patience resets at each phase transition (prevents premature stopping when backbone is newly unfrozen)

### 5.7 Mixed Precision

- `torch.amp.autocast` for FP16 forward passes
- `torch.amp.GradScaler` to prevent gradient underflow
- Safe in FP16: convolutions, linear layers, attention
- Always FP32: loss computation, softmax, batch norm
- Disableable via `training.mixed_precision: false` for debugging

### 5.8 Logging

| Channel | Content | Tool |
|---------|---------|------|
| TensorBoard | Scalars (loss, LR, all metrics), images (confusion matrix), per-epoch | `torch.utils.tensorboard.SummaryWriter` |
| Console | tqdm progress bar, epoch summary line | tqdm + print |
| File | Full training log with timestamps | Python `logging` module, saved to experiment dir |

### 5.9 Separation of Concerns

```
Trainer ─── orchestrates the loop
  ├── MetricTracker ─── computes metrics
  ├── Loss module ─── provides loss function
  ├── Checkpointer ─── manages checkpoint saves/loads
  ├── EarlyStopping ─── monitors for convergence
  ├── Optimizer ─── parameter updates
  └── Scheduler ─── learning rate adjustments
```

---

## 6. Experiment Management

### 6.1 Experiment Naming

**Convention**: `{backbone}_{dataset}_{YYYYMMDD}_{HHMM}`

**Examples:**
- `mobilenetv3_rafdb_20260724_0930`
- `efficientnet_b0_rafdb_20260725_1400`
- `mobilevit_xs_rafdb_20260726_1100`

Auto-generated from config values + current timestamp.

### 6.2 Experiment Directory Structure

```
experiments/
  mobilenetv3_rafdb_20260724_0930/
    config.yaml          # Frozen copy of resolved experiment config
    checkpoints/
      best_model.pt
      last_model.pt
    logs/
      events.out.tfevents.*   # TensorBoard event files
    results/
      metrics.json            # All evaluation metrics
      confusion_matrix.png
      reliability_diagram.png
      per_class_f1.png
      roc_curves.png
      classification_report.txt
    exports/
      model.onnx              # Exported ONNX model
      benchmark_report.json   # Latency and size comparison
```

### 6.3 Random Seed Handling

`seed_everything(seed)` sets:
- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- DataLoader worker seeds via `worker_init_fn`

**CUDA determinism:**
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- Trade-off: ~10-15% slower training, but fully reproducible results

### 6.4 TensorBoard Integration

- All experiments log to `experiments/{name}/logs/`
- Launch: `tensorboard --logdir experiments/`
- Compare backbones side-by-side in TensorBoard UI
- Confusion matrices logged as images at epoch end

### 6.5 Experiment Comparison

A standalone comparison utility:
1. Scans `experiments/*/results/metrics.json`
2. Generates comparison CSV/markdown table
3. Columns: backbone, params, model size, macro-F1, accuracy, ECE, PyTorch latency, ONNX latency, ONNX size

---

## 7. ONNX Deployment Pipeline

### 7.1 Export Process

1. Load best PyTorch checkpoint
2. Set model to eval mode
3. Include softmax in the model graph (so ONNX outputs probabilities directly)
4. `torch.onnx.export()` with:
   - Dummy input: `torch.randn(1, 3, 224, 224)`
   - Dynamic axes: `{'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`
   - Opset version: **17** (broad compatibility, modern operator support)
5. Run `onnx-simplifier` to optimize the computation graph
6. Validate with `onnx.checker.check_model()`
7. Verify file size < 100MB

> [!IMPORTANT]
> **Softmax handling**: The softmax layer is included inside the ONNX graph so that the exported model directly outputs calibrated probabilities. Temperature scaling (if applied) should be baked in before export.

### 7.2 Export Verification

- Run inference on 10+ sample images with both PyTorch and ONNX Runtime
- Compare probability distributions
- **Tolerance**: max absolute difference < 1e-5
- Log pass/fail status and max observed difference

### 7.3 Latency Benchmarking

| Metric | Methodology |
|--------|-------------|
| PyTorch CPU | 100 warm-up, 1000 timed iterations, batch=1 |
| ONNX Runtime CPU | Same |
| ONNX Runtime GPU | Same (if available) |

Report: mean, std, p95 latency in milliseconds.

### 7.4 File Size Comparison

| Artifact | Expected Size |
|----------|--------------|
| MobileNetV3 `.pt` | ~22 MB |
| MobileNetV3 `.onnx` | ~22 MB |
| EfficientNet-B0 `.pt` | ~21 MB |
| EfficientNet-B0 `.onnx` | ~21 MB |
| MobileViT-XS `.pt` | ~9 MB |
| MobileViT-XS `.onnx` | ~9 MB |

All well under the 100MB constraint.

### 7.5 ONNX Inference Pipeline

1. Load ONNX model via `onnxruntime.InferenceSession`
2. Preprocessing: same resize + normalize as training (NumPy/OpenCV-based, no PyTorch dependency)
3. Post-processing: label index → emotion name mapping
4. Face detection integration: raw image → MediaPipe face crop → preprocess → ONNX inference → probabilities

### 7.6 Deployment Validation

End-to-end test:
```
Raw image → Face detection → Crop → Resize → Normalize → ONNX inference → Emotion probabilities
```
Verify output matches PyTorch pipeline on the same raw image.

---

## 8. Evaluation Pipeline

### 8.1 Metrics Hierarchy

| Level | Metric | Purpose |
|-------|--------|---------|
| **Primary** | Macro-F1 | Model selection — fair across imbalanced classes |
| **Secondary** | Accuracy | Literature comparison |
| **Secondary** | Weighted-F1 | Operational quality assessment |
| **Per-class** | Precision, Recall, F1 | Identify weak emotion classes (disgust, fear) |
| **Ranking** | ROC-AUC (one-vs-rest) | Threshold-independent discrimination |
| **Calibration** | ECE, Reliability diagram | Probability meaningfulness — **required by task spec** |

### 8.2 Confusion Matrix

- Row-normalized (true labels on rows) heatmap
- Saved as PNG + logged to TensorBoard
- **Common FER confusions to watch:**
  - anger ↔ disgust (similar facial muscle activation)
  - fear ↔ surprise (similar eye widening)
  - sadness ↔ neutral (subtle differences)

### 8.3 Calibration Assessment

The task spec explicitly requires **probability outputs**. Raw softmax probabilities from neural networks are typically overconfident.

**Workflow:**
1. Train model → save best checkpoint
2. Compute ECE on validation set (15 bins)
3. Learn temperature parameter T on validation logits using L-BFGS
4. Recompute ECE after temperature scaling
5. Generate before/after reliability diagrams
6. Bake temperature into model before ONNX export

**Target**: ECE < 0.10 after calibration.

### 8.4 Backbone Comparison Table

| Backbone | Params | Size (MB) | Macro-F1 | Accuracy | ECE (post-cal) | Latency PT (ms) | Latency ONNX (ms) | ONNX Size (MB) |
|----------|--------|-----------|----------|----------|----------------|-----------------|-------------------|-----------------|
| MobileNetV3-Large | 5.5M | ~22 | — | — | — | — | — | — |
| EfficientNet-B0 | 5.3M | ~21 | — | — | — | — | — | — |
| MobileViT-XS | 2.3M | ~9 | — | — | — | — | — | — |

**Fairness**: All backbones trained with identical data splits, augmentation pipeline, and training protocol.

### 8.5 Model Selection Criteria

Decision hierarchy (in order):
1. ✅ ONNX file < 100MB
2. 📊 Highest macro-F1 on RAF-DB test set
3. 📐 ECE < 0.10 after temperature scaling
4. ⚡ Reasonable inference latency for CPU deployment

### 8.6 Auto-Generated Evaluation Report

Each experiment produces:
- `results/metrics.json` — all metrics as structured JSON
- `results/confusion_matrix.png` — normalized heatmap
- `results/reliability_diagram.png` — before/after calibration
- `results/per_class_f1.png` — bar chart of per-class F1
- `results/roc_curves.png` — one-vs-rest ROC curves
- `results/classification_report.txt` — sklearn classification report

---

## 9. Development Roadmap

### 3-Week Milestone Plan

| ID | Milestone | Deliverable | Dependencies | Days | Testable Output |
|:--:|-----------|-------------|:------------:|:----:|-----------------|
| **1.1** | Project Scaffold & Config | Directory structure, config parser, utils | None | 1-2 | `pip install -e .` works; config loads without error |
| **1.2** | RAF-DB Adapter & Preprocessing | RAF-DB dataset class, transforms, face detection | 1.1 | 2-3 | DataLoader outputs `[B, 3, 224, 224]` tensors with correct labels |
| **1.3** | Model Factory & Baseline Loop | timm integration (MobileNetV3), basic train/val loop | 1.1, 1.2 | 3-5 | 1-epoch training completes; loss decreases |
| **1.4** | Basic Eval & End-to-End Run | Accuracy/F1 tracking, TensorBoard logging | 1.3 | 5-7 | Full training run completes; metrics visible in TensorBoard |
| **2.1** | Multi-backbone Support | EfficientNet-B0 and MobileViT-XS added | 1.3 | 8 | All 3 backbones train successfully |
| **2.2** | Augmentation & Class Balance | Advanced transforms, weighted CE, balanced sampler | 1.2 | 8-9 | Augmented samples visualized; class distribution verified |
| **2.3** | FER2013 Adapter | FER2013 dataset class and loader | 1.2 | 10 | FER2013 DataLoader outputs correct shapes and labels |
| **2.4** | Multi-phase Training & Tuning | Phase transitions, LR scheduling, hyperparameter experiments | 1.4, 2.2 | 10-12 | Validation metrics improve over baseline |
| **2.5** | Comprehensive Evaluation & Calibration | ECE, temperature scaling, confusion matrix, reliability diagrams | 1.4 | 12-14 | Auto-generated evaluation report (JSON + PNGs) |
| **3.1** | ONNX Export & Benchmarking | Export script, numerical validation, latency test | 2.5 | 15-16 | `.onnx` file created; tolerance test passes; size < 100MB |
| **3.2** | Gradio Demo | Web UI for live inference | 3.1 | 16-17 | Browser-based interactive demo at localhost |
| **3.3** | AffectNet Adapter (Optional) | AffectNet dataset support | 2.3 | 18 | DataLoader functions correctly |
| **3.4** | Final Comparison Report | Final metrics for all backbones | 2.5, 3.1 | 18-19 | Completed backbone comparison table |
| **3.5** | Documentation & Presentation | README, architecture doc, presentation | All | 20-21 | Polished repository; presentation slides |

### Critical Path

```
1.1 → 1.2 → 1.3 → 1.4 → 2.4 → 2.5 → 3.1 → 3.2
                                              ↓
                                         3.4 → 3.5
```

> [!WARNING]
> **Risk mitigation**: Test ONNX export early (during Milestone 1.3 or 1.4 as a quick sanity check) rather than waiting until Week 3. This prevents discovering ONNX incompatibilities late.

---

## 10. Implementation Tickets

### Phase 1: Project Foundation

---

#### T-001: Project Scaffold

| Field | Detail |
|-------|--------|
| **Objective** | Establish directory structure and standard project files |
| **Files to Create** | `pyproject.toml`, `.gitignore`, `README.md`, all `__init__.py` files per repository structure |
| **Dependencies** | None |
| **Expected Outputs** | Installable Python package; all directories exist |
| **Acceptance Criteria** | `pip install -e .` succeeds; `import fer` works; `.gitignore` covers `data/`, `experiments/`, `__pycache__`, `.venv/` |
| **Validation** | `pip install -e . && python -c "import fer; print('OK')"` |

---

#### T-002: Configuration System

| Field | Detail |
|-------|--------|
| **Objective** | Implement type-safe YAML config system with dataclasses |
| **Files to Create** | `src/fer/config.py`, `configs/base.yaml` |
| **Dependencies** | T-001 |
| **Expected Outputs** | Dataclasses: `ExperimentConfig`, `DataConfig`, `AugmentationConfig`, `ModelConfig`, `TrainingConfig`, `ExportConfig`, `ProjectConfig`. Function: `load_config(path) → ProjectConfig` |
| **Acceptance Criteria** | Loads `base.yaml`, validates types, errors on missing required fields, supports recursive override merging |
| **Validation** | `python -c "from fer.config import load_config; cfg = load_config('configs/base.yaml'); print(cfg.model.backbone)"` |

---

#### T-003: Utility Modules

| Field | Detail |
|-------|--------|
| **Objective** | Create logging, seeding, device management, and I/O utilities |
| **Files to Create** | `src/fer/utils/logging.py`, `src/fer/utils/seeding.py`, `src/fer/utils/device.py`, `src/fer/utils/io.py` |
| **Dependencies** | T-001 |
| **Expected Outputs** | `setup_logger(name, log_dir)`, `seed_everything(seed)`, `get_device()`, `ensure_dir(path)`, `save_json(data, path)`, `load_json(path)` |
| **Acceptance Criteria** | Logger outputs to console + file; seed function covers random/numpy/torch/cuda; device detects CPU/CUDA/MPS |
| **Validation** | `python -c "from fer.utils.seeding import seed_everything; seed_everything(42); print('OK')"` |

---

#### T-004: Label Mapping Module

| Field | Detail |
|-------|--------|
| **Objective** | Define canonical 7-class label space and mapping utilities |
| **Files to Create** | `src/fer/data/label_mapping.py` |
| **Dependencies** | T-001 |
| **Expected Outputs** | Constants: `EMOTION_LABELS`, `LABEL_TO_IDX`, `IDX_TO_LABEL`, `NUM_CLASSES`. Functions: `map_label(dataset_name, original_label) → int`. Per-dataset mapping dicts for RAF-DB, FER2013, AffectNet. |
| **Acceptance Criteria** | `LABEL_TO_IDX['happiness'] == 3`; RAF-DB label 4 maps to canonical "happiness" (idx 3); validation asserts raise on unknown labels |
| **Validation** | `python -c "from fer.data.label_mapping import LABEL_TO_IDX; assert LABEL_TO_IDX['happiness'] == 3; print('OK')"` |

---

#### T-005: Base Dataset Abstract Class

| Field | Detail |
|-------|--------|
| **Objective** | Create abstract base class for all FER datasets |
| **Files to Create** | `src/fer/data/base_dataset.py` |
| **Dependencies** | T-002, T-004 |
| **Expected Outputs** | `BaseFERDataset(torch.utils.data.Dataset, ABC)` with: abstract `load_metadata()`, concrete `__len__`, `__getitem__` (loads image, applies transform, returns tensor+label), `get_class_weights()`, `get_labels()` |
| **Acceptance Criteria** | Cannot be instantiated directly (raises TypeError); subclasses must implement `load_metadata()` |
| **Validation** | `python -c "from fer.data.base_dataset import BaseFERDataset; BaseFERDataset()"` should raise TypeError |

---

### Phase 2: Data Pipeline

---

#### T-006: Face Detection Module

| Field | Detail |
|-------|--------|
| **Objective** | Implement configurable face detection wrapper |
| **Files to Create** | `src/fer/data/face_detector.py` |
| **Dependencies** | T-003 |
| **Expected Outputs** | `FaceDetector` class with `detect(image) → list[FaceDetection]` and `detect_and_crop(image) → cropped_face`. Supports bypass mode (returns original image). Backend: MediaPipe (lightweight) or RetinaFace (higher accuracy). |
| **Acceptance Criteria** | Detects face in sample image; returns cropped face. Bypass mode returns original. Falls back to full image if no face detected. |
| **Validation** | Feed sample image → verify cropped output is a face region. Feed non-face image → verify fallback. |

---

#### T-007: Preprocessing Pipeline

| Field | Detail |
|-------|--------|
| **Objective** | Chain face detection → alignment → crop → resize → save |
| **Files to Create** | `src/fer/data/preprocessing.py`, `scripts/preprocess_dataset.py` |
| **Dependencies** | T-006 |
| **Expected Outputs** | `Preprocessor` class and CLI script that processes a raw dataset directory into standardized 224×224 RGB images + metadata CSV |
| **Acceptance Criteria** | Output images are exactly (224, 224, 3) RGB. Metadata CSV matches schema. Grayscale sources converted to 3-channel. |
| **Validation** | Process 10 sample images → check output dimensions and CSV columns |

---

#### T-008: Augmentation Transforms

| Field | Detail |
|-------|--------|
| **Objective** | Define albumentations transform pipelines |
| **Files to Create** | `src/fer/data/transforms.py` |
| **Dependencies** | T-007 |
| **Expected Outputs** | `get_train_transforms(config) → A.Compose`, `get_val_transforms(config) → A.Compose`. Train: HorizontalFlip(0.5), ShiftScaleRotate(±10°, p=0.3), RandomResizedCrop(scale=0.92-1.0), ColorJitter(p=0.25), CoarseDropout/RandomErasing(p=0.1), Normalize(ImageNet). Val: Resize(224) + Normalize(ImageNet) only. |
| **Acceptance Criteria** | Train transforms produce visible augmentation variety; val transforms preserve image. Both output tensors of correct shape. |
| **Validation** | Apply train transform to same image 5 times → visually verify variation. Check output shape [3, 224, 224]. |

---

#### T-009: RAF-DB Dataset Adapter

| Field | Detail |
|-------|--------|
| **Objective** | Implement RAF-DB dataset class |
| **Files to Create** | `src/fer/data/rafdb_dataset.py` |
| **Dependencies** | T-005, T-008 |
| **Expected Outputs** | `RAFDBDataset(BaseFERDataset)` that parses RAF-DB annotations, maps labels to canonical 0-6, creates val split from training data |
| **Acceptance Criteria** | Correctly maps RAF-DB labels (1-7) to canonical (0-6). Respects official train/test split. 10% stratified val from train. |
| **Validation** | `dataset = RAFDBDataset(...); img, label = dataset[0]; assert img.shape == (3, 224, 224); assert 0 <= label <= 6` |

---

#### T-010: DataLoader Factory

| Field | Detail |
|-------|--------|
| **Objective** | Create DataLoader construction utility |
| **Files to Create** | `src/fer/data/__init__.py` (add `create_dataloaders` function) |
| **Dependencies** | T-009 |
| **Expected Outputs** | `create_dataloaders(config) → (train_loader, val_loader, test_loader)`. Configurable batch_size, num_workers, pin_memory. Optional WeightedRandomSampler for balanced training. |
| **Acceptance Criteria** | Returns functional DataLoaders. Balanced sampling produces approximately equal class representation per batch. |
| **Validation** | Iterate 1 batch → verify shape `[B, 3, 224, 224]` and label range [0, 6] |

---

### Phase 3: Model Architecture

---

#### T-011: Backbone Factory

| Field | Detail |
|-------|--------|
| **Objective** | Integrate timm for backbone instantiation |
| **Files to Create** | `src/fer/models/backbone_factory.py` |
| **Dependencies** | T-002 |
| **Expected Outputs** | `create_backbone(name, pretrained) → (nn.Module, feature_dim)`. Supports: `mobilenetv3_large_100`, `efficientnet_b0`, `mobilevit_xs`. Removes original classifier head. |
| **Acceptance Criteria** | All 3 backbones load successfully with pretrained weights. Returns feature dimension (e.g., 960 for MobileNetV3). Output is feature map, not classification logits. |
| **Validation** | `backbone, dim = create_backbone("mobilenetv3_large_100", True); out = backbone(torch.randn(1,3,224,224)); print(out.shape, dim)` |

---

#### T-012: Classification Head

| Field | Detail |
|-------|--------|
| **Objective** | Build custom FER classification head |
| **Files to Create** | `src/fer/models/heads.py` |
| **Dependencies** | None |
| **Expected Outputs** | `FERHead(nn.Module)`: AdaptiveAvgPool2d(1) → Flatten → Dropout(configurable) → Linear(feature_dim, num_classes) |
| **Acceptance Criteria** | Accepts any feature_dim and num_classes. Output shape: `[B, num_classes]`. Dropout rate configurable. |
| **Validation** | `head = FERHead(960, 7); out = head(torch.randn(2, 960, 7, 7)); assert out.shape == (2, 7)` |

---

#### T-013: FER Model

| Field | Detail |
|-------|--------|
| **Objective** | Combine backbone + head into complete model |
| **Files to Create** | `src/fer/models/fer_model.py` |
| **Dependencies** | T-011, T-012 |
| **Expected Outputs** | `FERModel(nn.Module)` with: `forward(x) → logits`, `freeze_backbone()`, `unfreeze_backbone()`, `get_model_size_mb()`. Optional: `forward_with_softmax(x) → probabilities` for ONNX export. |
| **Acceptance Criteria** | Forward pass: input `[B, 3, 224, 224]` → output `[B, 7]`. Freeze/unfreeze correctly toggles `requires_grad`. Model size < 100MB for all backbones. |
| **Validation** | `model = FERModel("mobilenetv3_large_100"); out = model(torch.randn(1,3,224,224)); assert out.shape == (1,7); print(f"Size: {model.get_model_size_mb():.1f} MB")` |

---

### Phase 4: Training Core

---

#### T-014: Loss Functions

| Field | Detail |
|-------|--------|
| **Objective** | Implement training loss functions |
| **Files to Create** | `src/fer/training/losses.py` |
| **Dependencies** | T-002 |
| **Expected Outputs** | `WeightedCrossEntropyLoss(class_weights)`, `FocalLoss(alpha, gamma)`, `create_loss(config, class_weights) → nn.Module` |
| **Acceptance Criteria** | WCE correctly applies class weights. Focal Loss reduces weight of easy examples. Factory function selects based on config. |
| **Validation** | Pass dummy logits/targets → assert loss is scalar tensor with grad |

---

#### T-015: Metrics Module

| Field | Detail |
|-------|--------|
| **Objective** | Implement metric accumulation and computation |
| **Files to Create** | `src/fer/training/metrics.py` |
| **Dependencies** | T-004 |
| **Expected Outputs** | `MetricTracker` class: `update(logits, targets)`, `compute() → dict`, `reset()`. Computes: accuracy, macro_f1, weighted_f1, per_class_precision, per_class_recall, per_class_f1, confusion_matrix, roc_auc |
| **Acceptance Criteria** | Accumulates across batches. Returns correct metrics dict. Confusion matrix shape: `[7, 7]`. |
| **Validation** | Feed synthetic predictions → verify F1 matches manual calculation |

---

#### T-016: Callbacks

| Field | Detail |
|-------|--------|
| **Objective** | Implement EarlyStopping and ModelCheckpointer |
| **Files to Create** | `src/fer/training/callbacks.py` |
| **Dependencies** | T-003 |
| **Expected Outputs** | `EarlyStopping(patience, min_delta, monitor)` with `step(metric) → should_stop`, `reset()`. `ModelCheckpointer(save_dir, monitor, top_k)` with `step(metric, state_dict) → saved_path`. |
| **Acceptance Criteria** | EarlyStopping triggers after patience epochs of no improvement. Checkpointer saves on improvement, maintains top-K files. |
| **Validation** | Simulate 15 epochs of decreasing then stagnant metrics → assert EarlyStopping triggers at correct epoch |

---

#### T-017: Trainer

| Field | Detail |
|-------|--------|
| **Objective** | Core training engine |
| **Files to Create** | `src/fer/training/trainer.py`, `src/fer/training/optimizer.py` |
| **Dependencies** | T-013, T-014, T-015, T-016 |
| **Expected Outputs** | `Trainer` class: `train()` method executing multi-phase loop. Handles: epoch iteration, optimizer step, AMP, validation, TensorBoard logging, checkpointing, early stopping, phase transitions. `create_optimizer(model, config)` and `create_scheduler(optimizer, config)` utilities. |
| **Acceptance Criteria** | Completes 1 epoch without errors. TensorBoard events file created. Checkpoint saved. Console shows tqdm progress. |
| **Validation** | Initialize with small DataLoader + model → run 1 epoch → verify TensorBoard log dir has events |

---

#### T-018: Training Script

| Field | Detail |
|-------|--------|
| **Objective** | Main CLI entry point for training |
| **Files to Create** | `scripts/train.py` |
| **Dependencies** | T-010, T-017 |
| **Expected Outputs** | Script accepting `--config` argument. Loads config, creates experiment directory, initializes all components, starts training. |
| **Acceptance Criteria** | `python scripts/train.py --config configs/base.yaml` runs without error. Experiment directory created with config.yaml copy. |
| **Validation** | Run with `--config configs/base.yaml` for 1 epoch (override epochs=1) → verify experiment directory structure |

---

#### T-019: First End-to-End Training Run

| Field | Detail |
|-------|--------|
| **Objective** | Verify complete system cohesion on real data |
| **Files to Create** | `configs/mobilenetv3.yaml` |
| **Dependencies** | T-018 |
| **Expected Outputs** | Successful multi-epoch training of MobileNetV3 on RAF-DB. Decreasing loss, improving macro-F1. |
| **Acceptance Criteria** | Training completes. Loss decreases. Best model checkpoint saved. TensorBoard logs show learning curves. |
| **Validation** | Inspect TensorBoard → confirm loss goes down, F1 goes up. Check `best_model.pt` exists. |

---

### Phase 5: Evaluation & Calibration

---

#### T-020: Evaluator Module

| Field | Detail |
|-------|--------|
| **Objective** | Standalone test set evaluation |
| **Files to Create** | `src/fer/evaluation/evaluator.py` |
| **Dependencies** | T-013, T-015 |
| **Expected Outputs** | `Evaluator` class: `evaluate(model, test_loader) → (metrics_dict, all_logits, all_targets)`. Loads checkpoint, runs eval, returns all metrics + raw outputs for calibration. |
| **Acceptance Criteria** | Returns complete metrics dict. Raw logits/targets available for downstream calibration. |
| **Validation** | Evaluate saved checkpoint on test set → verify metrics dict contains all expected keys |

---

#### T-021: Calibration Module

| Field | Detail |
|-------|--------|
| **Objective** | Temperature scaling and ECE computation |
| **Files to Create** | `src/fer/evaluation/calibration.py` |
| **Dependencies** | T-020 |
| **Expected Outputs** | `compute_ece(probs, targets, n_bins=15) → float`, `TemperatureScaler` nn.Module that learns optimal T on validation logits using L-BFGS. `calibrate(logits, targets) → (temperature, calibrated_probs)`. |
| **Acceptance Criteria** | ECE computes correctly. Temperature scaling reduces ECE on validation set. |
| **Validation** | Pass uncalibrated logits → optimize temperature → assert ECE decreases |

---

#### T-022: Visualization Module

| Field | Detail |
|-------|--------|
| **Objective** | Plotting utilities for evaluation reports |
| **Files to Create** | `src/fer/evaluation/visualization.py` |
| **Dependencies** | T-020 |
| **Expected Outputs** | Functions: `plot_confusion_matrix(cm, labels, save_path)`, `plot_per_class_f1(f1_scores, labels, save_path)`, `plot_roc_curves(targets, probs, labels, save_path)`, `plot_reliability_diagram(probs, targets, save_path)`. All save PNG files. |
| **Acceptance Criteria** | All functions produce valid PNG images. Confusion matrix is normalized row-wise. Reliability diagram shows calibration line. |
| **Validation** | Call each function with mock data → verify PNG files are created and openable |

---

#### T-023: Evaluation Script

| Field | Detail |
|-------|--------|
| **Objective** | CLI entry point for evaluation |
| **Files to Create** | `scripts/evaluate.py` |
| **Dependencies** | T-020, T-021, T-022 |
| **Expected Outputs** | Script accepting `--checkpoint` and `--config`. Runs full evaluation + calibration + generates all plots and metrics JSON. |
| **Acceptance Criteria** | `python scripts/evaluate.py --checkpoint best_model.pt --config base.yaml` produces complete results directory |
| **Validation** | Run on trained checkpoint → verify `results/` contains `metrics.json` + all PNG plots |

---

### Phase 6: Additional Datasets

---

#### T-024: FER2013 Dataset Adapter

| Field | Detail |
|-------|--------|
| **Objective** | Add FER2013 support |
| **Files to Create** | `src/fer/data/fer2013_dataset.py` |
| **Dependencies** | T-005 |
| **Expected Outputs** | `FER2013Dataset(BaseFERDataset)` parsing CSV pixel columns, mapping labels, using official splits (Training/PublicTest/PrivateTest) |
| **Acceptance Criteria** | Handles 48×48 grayscale → 224×224 RGB conversion. Label mapping correct. Official splits preserved. |
| **Validation** | `dataset = FER2013Dataset(..., split="train"); img, label = dataset[0]; assert img.shape == (3, 224, 224)` |

---

#### T-025: AffectNet Dataset Adapter

| Field | Detail |
|-------|--------|
| **Objective** | Add AffectNet-7 support |
| **Files to Create** | `src/fer/data/affectnet_dataset.py` |
| **Dependencies** | T-005 |
| **Expected Outputs** | `AffectNetDataset(BaseFERDataset)` parsing AffectNet directory structure. Drops contempt class (label 7). Uses official train/val split. |
| **Acceptance Criteria** | Only 7 classes loaded. Official split preserved. Labels correctly mapped. |
| **Validation** | `dataset = AffectNetDataset(..., split="train"); assert set(dataset.get_labels()).issubset({0,1,2,3,4,5,6})` |

---

### Phase 7: ONNX Pipeline

---

#### T-026: ONNX Exporter

| Field | Detail |
|-------|--------|
| **Objective** | Export PyTorch model to ONNX |
| **Files to Create** | `src/fer/export/onnx_exporter.py` |
| **Dependencies** | T-013 |
| **Expected Outputs** | `export_to_onnx(model, save_path, config)` using `torch.onnx.export`, opset 17, dynamic batch axis. Includes softmax in graph. Runs onnx-simplifier. Validates with `onnx.checker`. |
| **Acceptance Criteria** | Produces valid `.onnx` file. File size < 100MB. onnx.checker passes. |
| **Validation** | Export MobileNetV3 → verify `.onnx` file exists and checker passes |

---

#### T-027: ONNX Verifier

| Field | Detail |
|-------|--------|
| **Objective** | Numerical comparison PyTorch vs ONNX Runtime |
| **Files to Create** | `src/fer/export/onnx_verifier.py` |
| **Dependencies** | T-026 |
| **Expected Outputs** | `verify_onnx(pytorch_model, onnx_path, num_samples=10) → (passed: bool, max_diff: float)`. Compares outputs on random inputs. |
| **Acceptance Criteria** | Max absolute difference < 1e-5. Tests with multiple input samples. |
| **Validation** | Run verification → assert `passed == True` and `max_diff < 1e-5` |

---

#### T-028: Benchmarker

| Field | Detail |
|-------|--------|
| **Objective** | Latency and file size benchmarking |
| **Files to Create** | `src/fer/export/benchmarker.py` |
| **Dependencies** | T-027 |
| **Expected Outputs** | `benchmark(pytorch_model, onnx_path) → BenchmarkReport`. Measures: PyTorch CPU latency, ONNX CPU latency, file sizes. 100 warmup + 1000 iterations. Reports mean/std/p95. |
| **Acceptance Criteria** | Returns structured benchmark report. ONNX size < 100MB verified. |
| **Validation** | Run benchmark → verify report contains latency and size metrics |

---

#### T-029: Export Script

| Field | Detail |
|-------|--------|
| **Objective** | CLI for complete export pipeline |
| **Files to Create** | `scripts/export_onnx.py` |
| **Dependencies** | T-026, T-027, T-028 |
| **Expected Outputs** | Script accepting `--checkpoint` and `--config`. Runs: export → verify → benchmark. Saves all outputs to experiment's `exports/` directory. |
| **Acceptance Criteria** | `python scripts/export_onnx.py --checkpoint best_model.pt` produces `.onnx` + `benchmark_report.json` |
| **Validation** | Run on trained checkpoint → verify all export artifacts exist |

---

### Phase 8: Inference & Demo

---

#### T-030: Predictor Module

| Field | Detail |
|-------|--------|
| **Objective** | ONNX Runtime inference wrapper |
| **Files to Create** | `src/fer/inference/predictor.py` |
| **Dependencies** | T-007, T-026 |
| **Expected Outputs** | `FERPredictor` class: loads ONNX model, implements `predict(image) → dict[str, float]` returning emotion probabilities. Includes preprocessing (resize, normalize using NumPy/OpenCV). Optional face detection integration. |
| **Acceptance Criteria** | Accepts raw image (path or NumPy array). Returns dict like `{"happiness": 0.85, "neutral": 0.10, ...}`. Probabilities sum to ~1.0. |
| **Validation** | Predict on sample image → verify output is valid probability distribution |

---

#### T-031: Gradio Demo

| Field | Detail |
|-------|--------|
| **Objective** | Interactive web UI for emotion recognition |
| **Files to Create** | `src/fer/inference/demo.py` |
| **Dependencies** | T-030 |
| **Expected Outputs** | Gradio Blocks interface: image upload or webcam input → bar chart of emotion probabilities + input image displayed. |
| **Acceptance Criteria** | App launches at localhost:7860. Processes uploaded images correctly. Bar chart shows all 7 emotions with probabilities. |
| **Validation** | Launch demo → upload sample image → verify bar chart renders correctly |

---

#### T-032: Demo Script

| Field | Detail |
|-------|--------|
| **Objective** | CLI entry point for Gradio demo |
| **Files to Create** | `scripts/demo.py` |
| **Dependencies** | T-031 |
| **Expected Outputs** | Script accepting `--model` (ONNX path). Launches Gradio app. |
| **Acceptance Criteria** | `python scripts/demo.py --model model.onnx` starts the server |
| **Validation** | Run command → access localhost:7860 in browser |

---

### Phase 9: Experiment Management & Comparison

---

#### T-033: Experiment Comparison Utilities

| Field | Detail |
|-------|--------|
| **Objective** | Compare results across multiple experiments |
| **Files to Create** | `src/fer/utils/compare.py` |
| **Dependencies** | T-023 |
| **Expected Outputs** | `compare_experiments(experiment_dirs) → DataFrame/markdown`. Parses `metrics.json` from each experiment, generates comparison table. |
| **Acceptance Criteria** | Aggregates: backbone, macro-F1, accuracy, ECE, latency, model size |
| **Validation** | Run on 2+ experiment directories → verify formatted table output |

---

#### T-034: Multi-Backbone Experiment Configs

| Field | Detail |
|-------|--------|
| **Objective** | Define per-backbone YAML configs |
| **Files to Create** | `configs/efficientnet_b0.yaml`, `configs/mobilevit_xs.yaml` |
| **Dependencies** | T-002, T-011 |
| **Expected Outputs** | Override configs that only change `model.backbone` from base |
| **Acceptance Criteria** | Both configs load without errors. Training runs with each backbone. |
| **Validation** | `load_config("configs/efficientnet_b0.yaml")` succeeds |

---

#### T-035: Backbone Comparison Report

| Field | Detail |
|-------|--------|
| **Objective** | Automate final comparison report generation |
| **Files to Create** | `scripts/generate_report.py` |
| **Dependencies** | T-033, T-034 |
| **Expected Outputs** | Script that scans experiment directories, generates markdown comparison report with tables and key findings |
| **Acceptance Criteria** | Report includes: comparison table, best model recommendation, per-backbone metrics |
| **Validation** | Run after completing experiments for all 3 backbones → verify report is generated |

---

### Phase 10: Testing & Documentation

---

#### T-036: Unit Tests

| Field | Detail |
|-------|--------|
| **Objective** | Core component unit tests |
| **Files to Create** | `tests/test_label_mapping.py`, `tests/test_transforms.py`, `tests/test_model.py`, `tests/test_metrics.py` |
| **Dependencies** | All source modules |
| **Expected Outputs** | pytest suite covering: label mapping correctness, transform output shapes, model forward pass shapes, metric computation accuracy |
| **Acceptance Criteria** | All tests pass. Critical paths covered: label mapping edge cases, model output shapes, metric correctness. |
| **Validation** | `pytest tests/ -v` |

---

#### T-037: Integration Tests

| Field | Detail |
|-------|--------|
| **Objective** | End-to-end pipeline test on tiny data |
| **Files to Create** | `tests/test_pipeline.py` |
| **Dependencies** | T-036 |
| **Expected Outputs** | Test that: creates tiny synthetic dataset → trains 1 epoch → evaluates → exports ONNX → verifies numerics |
| **Acceptance Criteria** | Test passes without exceptions. Covers full pipeline. |
| **Validation** | `pytest tests/test_pipeline.py -v` |

---

#### T-038: Documentation

| Field | Detail |
|-------|--------|
| **Objective** | Finalize all project documentation |
| **Files to Create** | `README.md` (update), `docs/setup.md`, `docs/experiments.md`, `docs/architecture.md` |
| **Dependencies** | T-001 |
| **Expected Outputs** | Comprehensive guides: installation, dataset preparation, running training, running experiments, evaluating, exporting, demo. |
| **Acceptance Criteria** | A new developer can set up the repo and start a training run using only the documentation |
| **Validation** | Follow README instructions from scratch → verify everything works |

---

## 11. Potential Risks

| # | Risk | Impact | Likelihood | Prevention Strategy |
|---|------|--------|------------|---------------------|
| 1 | **Data Leakage** — same subject in train and test | High (invalidates results) | Medium | Use subject-independent splits; track `subject_id` in metadata; never break official splits |
| 2 | **Incorrect Label Mapping** — different datasets use different numeric IDs | High (silent model degradation) | High | Centralized `label_mapping.py`; validation asserts in `__getitem__`; unit tests for every dataset's mapping |
| 3 | **Preprocessing Inconsistency** — different normalization between datasets or train/test | Medium (degrades transfer learning) | High | Single preprocessing pipeline; config-driven normalization matching timm pretrained weights exactly |
| 4 | **Augmentation on Val/Test** — accidentally applying train augmentations to evaluation | High (invalid metrics) | Low | Strict separation: `get_train_transforms()` vs `get_val_transforms()` factory functions |
| 5 | **ONNX Incompatibility** — PyTorch ops not supported in ONNX | High (blocks deployment) | Low | Use standard timm backbones (well-tested for ONNX); test export in Week 1, not Week 3 |
| 6 | **Reproducibility Issues** — different results across runs | Low (frustrating) | Medium | `seed_everything()` covering all sources; deterministic CUDA mode; log seeds in config |
| 7 | **Experiment Overwrite** — accidentally overwriting previous experiments | Medium (lost work) | Medium | Timestamped unique experiment names; config snapshots saved in experiment dir |
| 8 | **Model Size > 100MB** — ONNX model exceeds constraint | High (fails requirement) | Very Low | All chosen backbones are 9-22MB FP32; automated size check in export pipeline |
| 9 | **Class Imbalance Collapse** — model ignores disgust/fear | High (poor real-world performance) | High | Weighted CE + balanced sampler; monitor per-class metrics and confusion matrix |
| 10 | **Calibration Neglect** — overconfident softmax probabilities | Medium (misleading probability outputs) | Medium | ECE measurement; temperature scaling; reliability diagrams; bake calibration into ONNX |
| 11 | **Face Detection Failure** — detector misses face or crops incorrectly | Medium (missing inferences) | Medium | Fallback to full image; manual inspection of preprocessed sample grid; log detection failures |
| 12 | **Numerical Drift** — ONNX outputs differ from PyTorch | Low (unexpected production behavior) | Low | Automated numerical comparison test with tolerance < 1e-5 during export |

---

## 12. Recommended Technology Stack

| Category | Tool | Version | Why This Choice |
|----------|------|---------|-----------------|
| **Language** | Python | 3.10-3.11 | Best PyTorch compatibility; modern features (dataclasses, f-strings, match statements) |
| **Deep Learning** | PyTorch | ≥2.1 | Imperative execution for easy debugging; torch.compile support; improved ONNX export; native AMP |
| **Vision** | torchvision | Match PyTorch | Basic image utilities; part of PyTorch ecosystem |
| **Model Zoo** | timm | ≥0.9.16 | De facto standard for vision backbones. All 3 architectures available under unified API. Reduces boilerplate dramatically. |
| **Augmentation** | albumentations | ≥1.3.1 | C++/OpenCV backend = faster than torchvision. Richer transforms. Reproducible random state. *Chosen over torchvision transforms for speed and flexibility.* |
| **Image I/O** | Pillow, OpenCV | Latest stable | Standard image loading. OpenCV required for albumentations and face detection preprocessing. |
| **Face Detection** | mediapipe | Latest stable | Lightweight, fast CPU inference. Good enough for offline preprocessing. *Alternative: insightface/RetinaFace for higher accuracy, but heavier dependency.* |
| **ONNX** | onnx + onnxruntime + onnx-simplifier | Latest stable | Industry standard. onnx-simplifier resolves complex PyTorch graph nodes. |
| **Metrics** | scikit-learn | ≥1.3 | Trusted implementation: macro/weighted F1, ROC-AUC, confusion matrix, classification report |
| **Visualization** | matplotlib + seaborn | Latest stable | Confusion matrix heatmaps, reliability diagrams, bar charts, ROC curves |
| **Experiment Tracking** | tensorboard | ≥2.14 | Zero setup, offline, local. *Chosen over W&B: no API key, no internet needed, simpler for 3-week project.* |
| **Demo** | gradio | ≥4.0 | Interactive ML demo in <50 lines. Webcam + upload + output visualization built-in. |
| **Config** | PyYAML | ≥6.0 | Simple YAML parsing. Paired with dataclasses for type safety. *Chosen over Hydra: far simpler, no magic.* |
| **Data** | pandas | ≥2.0 | Metadata CSV handling, label distribution analysis |
| **Progress** | tqdm | Latest | Lightweight training progress bars |
| **Testing** | pytest | ≥7.0 | Minimal boilerplate; excellent fixture support; standard for Python projects |
| **Formatting** | black + isort | Latest | Consistent code style without debates |
| **Linting** | ruff | Latest | Blazing fast; replaces flake8 + pylint + isort in one tool |
| **Dependencies** | pip + pyproject.toml | — | Modern Python packaging. *No conda: simpler, faster resolution.* |
| **Version Control** | git | — | Standard |

### Why NOT Chosen

| Tool | Reason for Rejection |
|------|---------------------|
| TensorFlow/Keras | Task spec allows both; PyTorch chosen for timm ecosystem and research community alignment |
| Hydra | Too complex for 3-week scope; steep learning curve; config directory proliferation |
| Weights & Biases | Requires API key + internet; overkill for single-developer 3-week project |
| torchvision transforms (for augmentation) | Slower than albumentations; fewer transforms available |
| conda | Slower resolution; unnecessary for this project; pip + venv is sufficient |
| Lightning/Catalyst | Training framework abstractions add complexity; vanilla PyTorch is more debuggable for research |

---

## User Review Required

> [!IMPORTANT]
> **This implementation blueprint is ready for your review.** Please confirm:
> 1. Does the repository structure match your expectations?
> 2. Are you comfortable with the 38-ticket breakdown and phasing?
> 3. Any specific datasets you want prioritized differently?
> 4. Should the demo include webcam support or image upload only?
> 5. Any preferences on the face detection library (MediaPipe vs RetinaFace)?

## Open Questions

> [!NOTE]
> These are low-priority questions that won't block implementation but would improve the design:
> 1. **Dataset access**: Do you already have RAF-DB and AffectNet downloaded, or should the preprocessing script include download helpers?
> 2. **GPU availability**: What GPU will be used for training? This affects batch size defaults and mixed precision recommendations.
> 3. **Temperature scaling in ONNX**: Should the learned temperature be baked into the ONNX model, or kept as a separate post-processing step?
