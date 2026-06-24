"""Convenience CLI for detection + species classification.

Run from the repository root:
    python detection_classification/full_predict.py path/to/image.jpg
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
CLASSIFICATION_ROOT = PROJECT_ROOT / "classification"
FULL_PIPELINE_PATH = PIPELINE_ROOT / "src" / "full_pipeline.py"

sys.path.insert(0, str(CLASSIFICATION_ROOT))

spec = importlib.util.spec_from_file_location(
    "detection_classification_full_pipeline",
    FULL_PIPELINE_PATH,
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load full pipeline module from {FULL_PIPELINE_PATH}")
full_pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full_pipeline
spec.loader.exec_module(full_pipeline)
main = full_pipeline.main


if __name__ == "__main__":
    main()
