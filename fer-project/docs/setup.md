# Setup & Installation Guide

## Prerequisites

- Python 3.10 or 3.11
- PyTorch 2.1+
- CUDA GPU (optional, Apple Silicon MPS and CPU fully supported)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd fer-project

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Dataset Setup

Place raw datasets under `./data/raw/`:

```
data/
└── raw/
    ├── rafdb/
    │   ├── EmoLabel/
    │   │   └── list_patition_label.txt
    │   └── Image/
    │       └── aligned/
    ├── fer2013/
    │   └── fer2013.csv
    └── affectnet/
        ├── labels/
        │   ├── training.csv
        │   └── validation.csv
        ├── train_set/images/
        └── val_set/images/
```

## Preprocessing

Run offline preprocessing for a dataset:

```bash
# Preprocess RAF-DB
python scripts/preprocess_dataset.py --dataset rafdb --data-dir ./data --bypass-face-detection

# Preprocess FER2013
python scripts/preprocess_dataset.py --dataset fer2013 --data-dir ./data --bypass-face-detection

# Preprocess AffectNet
python scripts/preprocess_dataset.py --dataset affectnet --data-dir ./data
```

## Verification

Run the full test suite to verify your setup:

```bash
pytest tests/ -v
```
