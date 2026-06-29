from pathlib import Path

from ultralytics import YOLO
import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model = YOLO("yolo26n-cls.pt")

    data_dir = SCRIPT_DIR / "dataset"
    model.train(
        data=str(data_dir),
        epochs=50,
        imgsz=224,
        patience=10,
    )

    model.val()

    metrics = model.val(data=str(data_dir / "test"))
    print(metrics)


if __name__ == "__main__":
    main()