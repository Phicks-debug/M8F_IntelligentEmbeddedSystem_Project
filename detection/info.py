from pathlib import Path

from ultralytics import YOLO

# Download with: yolo download yolo26n-cls.pt
SCRIPT_DIR = Path(__file__).resolve().parent

model = YOLO(SCRIPT_DIR / "yolo26n-cls.pt")
model.info()