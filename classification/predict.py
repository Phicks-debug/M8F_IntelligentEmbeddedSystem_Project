"""Convenience CLI for mushroom species + edibility prediction.

Run from the repository root:
    .venv/bin/python classification/predict.py path/to/image.jpg

    expects a classification model at: M8F_Intelligent.../exported_model/....
    
    
    example output:
    "image": "data/processed/classification_data/test/Pleurotus_ostreatus/inat_44_0.jpg",
  "top_prediction": {
    "class_name": "Pleurotus_ostreatus",
    "display_name": "Pleurotus ostreatus",
    "confidence": 0.9999902248382568,
    "edibility_category": "edible",
    "edibility_label": "commonly edible",
    "edibility_note": "Oyster mushroom; widely cultivated and eaten."
    
    """

from __future__ import annotations

import sys
from pathlib import Path

CLASSIFICATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from src.inference import main


if __name__ == "__main__":
    main()
