import argparse
import json
from pathlib import Path

from orchestration.classify import class_names
from orchestration.detect import crop, save_boxes
from orchestration.edibility import SAFETY_WARNING
from orchestration.hailo.classify import classify_image
from orchestration.hailo.detect import detect
from orchestration.hailo.runtime import HailoDevice, HailoModel
from orchestration.main import format_output


def run(
    image_path,
    detector,
    classifier,
    data_dir,
    threshold,
    padding,
    topk,
    save_detection=None,
):
    detections, best_candidate = detect(
        image_path,
        detector,
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
    predictions = classify_image(
        mushroom_crop,
        classifier,
        class_names(data_dir),
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


def main():
    parser = argparse.ArgumentParser(
        description="Run detection and classification with Hailo HEF models."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", type=Path, default=Path("exported_models/detection.hef"))
    parser.add_argument("--classifier", type=Path, default=Path("exported_models/mobilenetv4.hef"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save-detection", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with HailoDevice() as device:
        with HailoModel(args.detector, device) as detector:
            with HailoModel(args.classifier, device) as classifier:
                result = run(
                    args.image,
                    detector,
                    classifier,
                    args.data_dir,
                    args.threshold,
                    args.padding,
                    args.topk,
                    args.save_detection,
                )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
