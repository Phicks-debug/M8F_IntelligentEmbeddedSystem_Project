import argparse
import json
from pathlib import Path

from orchestration.classify import class_names, classify
from orchestration.detect import crop, detect
from orchestration.edibility import SAFETY_WARNING


def run(image_path, detector_path, classifier_path, data_dir, threshold, padding, device, topk):
    detections = detect(image_path, detector_path, threshold)
    if not detections:
        return {
            "image": str(image_path),
            "detection": None,
            "classification": None,
            "message": "No mushroom detection passed the confidence threshold.",
            "safety_warning": SAFETY_WARNING,
        }

    best = detections[0]
    mushroom_crop = crop(image_path, best, padding)
    predictions = classify(
        mushroom_crop,
        classifier_path,
        class_names(data_dir),
        device=device,
        topk=topk,
    )
    return {
        "image": str(image_path),
        "detection": best,
        "num_detections": len(detections),
        "classification": predictions,
        "safety_warning": SAFETY_WARNING,
    }


def format_output(result):
    if result["detection"] is None:
        return "\n".join(
            [
                "Detection: no mushroom found",
                result["message"],
                result["safety_warning"],
            ]
        )

    detection = result["detection"]
    best = result["classification"][0]
    return "\n".join(
        [
            f"Detection: mushroom ({detection['confidence'] * 100:.2f}%)",
            f"Species: {best['name']} ({best['confidence'] * 100:.2f}%)",
            f"Edibility: {best['edibility']}",
            f"Note: {best['note']}",
            result["safety_warning"],
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run detection, species classification, and edibility mapping."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", type=Path, default=Path("exported_models/detection.onnx"))
    parser.add_argument("--classifier", type=Path, default=Path("exported_models/mobilenetv4.pt"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(
        args.image,
        args.detector,
        args.classifier,
        args.data_dir,
        args.threshold,
        args.padding,
        args.device,
        args.topk,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
