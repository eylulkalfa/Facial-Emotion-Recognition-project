"""File I/O helper utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Creates directory and all parent directories if they don't exist.

    Args:
        path: Path string or Path object.

    Returns:
        Path object of the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to JSON file with formatting.

    Args:
        data: Dict to save.
        path: File path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file into a dictionary.

    Args:
        path: File path.

    Returns:
        Loaded dictionary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Return file size in megabytes.

    Args:
        path: File path.

    Returns:
        File size in MB as float.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return os.path.getsize(p) / (1024 * 1024)
