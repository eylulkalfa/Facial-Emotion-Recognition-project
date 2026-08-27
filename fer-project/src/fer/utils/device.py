"""Device management utilities for PyTorch (CUDA, MPS, CPU)."""

import torch


def get_device() -> torch.device:
    """Detect available compute device (CUDA -> MPS -> CPU).

    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Selected device: {device}")
    return device


def get_device_info() -> dict:
    """Get metadata about hardware devices.

    Returns:
        Dict with device name and availability info.
    """
    cuda_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    if cuda_avail:
        device_str = "cuda"
        cuda_name = torch.cuda.get_device_name(0)
        cuda_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif mps_avail:
        device_str = "mps"
        cuda_name = None
        cuda_mem = None
    else:
        device_str = "cpu"
        cuda_name = None
        cuda_mem = None

    return {
        "device": device_str,
        "cuda_available": cuda_avail,
        "mps_available": mps_avail,
        "cuda_device_name": cuda_name,
        "cuda_memory_gb": cuda_mem,
    }
