from ultralytics import YOLO
import torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


    model = YOLO("yolo11n-cls.pt")

    model.train(
        data=r"C:\Users\jules\Documents\prog\M8F_IntelligentEmbeddedSystem_Project\detection\dataset",
        epochs=50,
        imgsz=224,
        patience=10
    )

    model.val()

    metrics = model.val(data="dataset/test")
    print(metrics)


if __name__ == "__main__":
    main()