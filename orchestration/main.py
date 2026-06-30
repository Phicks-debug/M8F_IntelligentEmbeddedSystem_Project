import argparse
import json
from pathlib import Path

from orchestration.classify import class_names, classify
from orchestration.detect import crop, detect, save_boxes
from orchestration.edibility import SAFETY_WARNING


def run(
    image_path,
    detector_path,
    classifier_path,
    data_dir,
    threshold,
    padding,
    device,
    topk,
    save_detection=None,
):
    detections, best_candidate = detect(
        image_path,
        detector_path,
        threshold,
        return_best=True,
    )
    if save_detection:
        save_boxes(image_path, detections, save_detection)

    if not detections:
        return {
            "image": str(image_path),
            "detection_image": str(save_detection) if save_detection else None,
            "detection": None,
            "best_detection": best_candidate,
            "threshold": threshold,
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
        "detection_image": str(save_detection) if save_detection else None,
        "detection": best,
        "best_detection": best,
        "threshold": threshold,
        "num_detections": len(detections),
        "classification": predictions,
        "safety_warning": SAFETY_WARNING,
    }


def format_output(result):
    if result["detection"] is None:
        best_detection = result["best_detection"]
        lines = [
            "Detection: no mushroom found",
            f"Best detector guess: {best_detection['confidence'] * 100:.2f}% "
            f"(threshold: {result['threshold'] * 100:.2f}%)"
            if best_detection
            else "Best detector guess: none",
            result["message"],
            f"Detection image: {result['detection_image']}"
            if result["detection_image"]
            else "",
            result["safety_warning"],
        ]
        return "\n".join(
            line
            for line in lines
            if line
        )

    detection = result["detection"]
    best = result["classification"][0]
    lines = [
        f"Detection: mushroom ({detection['confidence'] * 100:.2f}%)",
        f"Detection image: {result['detection_image']}"
        if result["detection_image"]
        else "",
        f"Species: {best['name']} ({best['confidence'] * 100:.2f}%)",
        f"Edibility: {best['edibility']}",
        f"Note: {best['note']}",
        result["safety_warning"],
    ]
    return "\n".join(
        line
        for line in lines
        if line
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
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save-detection", type=Path)
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
        args.save_detection,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
