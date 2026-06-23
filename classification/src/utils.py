"""
Utility helpers for checkpoint passing and device selection.
"""

from pathlib import Path

import torch


def get_device(preferred: str = "auto") -> str:
    """Select best available device."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_to_bytes(path: Path) -> bytes:
    """Read a checkpoint file into bytes for Metaflow artifact passing."""
    return path.read_bytes()


def bytes_to_checkpoint(data: bytes, path: Path):
    """Write checkpoint bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
