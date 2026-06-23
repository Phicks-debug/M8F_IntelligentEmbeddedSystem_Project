"""Convenience CLI for mushroom species + edibility prediction.

Run from the repository root:
    .venv/bin/python classification/predict.py path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

CLASSIFICATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from src.inference import main


if __name__ == "__main__":
    main()
