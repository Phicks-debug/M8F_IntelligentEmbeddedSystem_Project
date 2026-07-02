import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from orchestration.detect import letterbox, save_boxes
from orchestration.hailo.runtime import HailoModel


def detection_rows(output):
    output = np.asarray(output)
    if output.ndim == 3:
        output = output[0]
    return output.reshape(-1, 6)


def detect_image(image, detector, threshold=0.60, return_best=False):
    width, height = image.size
    boxed, scale, pad_x, pad_y = letterbox(image.convert("RGB"))

    batch = np.asarray(boxed, dtype=np.float32) / 255.0
    batch = batch[None]
    output = detector.infer(batch)

    results = []
    best_candidate = None
    for row in detection_rows(output):
        x1, y1, x2, y2, confidence, class_id = row.tolist()
        box = [
            max(0, min(width, (x1 - pad_x) / scale)),
            max(0, min(height, (y1 - pad_y) / scale)),
            max(0, min(width, (x2 - pad_x) / scale)),
            max(0, min(height, (y2 - pad_y) / scale)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        candidate = {
            "label": "mushroom",
            "confidence": float(confidence),
            "class_id": int(class_id),
            "box": box,
        }
        if best_candidate is None or candidate["confidence"] > best_candidate["confidence"]:
            best_candidate = candidate
        if confidence >= threshold:
            results.append(candidate)

    results = sorted(results, key=lambda item: item["confidence"], reverse=True)
    if return_best:
        return results, best_candidate
    return results


def detect(image_path, detector, threshold=0.60, return_best=False):
    image = Image.open(image_path).convert("RGB")
    return detect_image(image, detector, threshold, return_best)


def main():
    parser = argparse.ArgumentParser(description="Run mushroom detection with Hailo HEF.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exported_models/detection.hef"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    with HailoModel(args.model) as detector:
        detections, best_candidate = detect(
            args.image,
            detector,
            args.threshold,
            return_best=True,
        )

    if args.save:
        save_boxes(args.image, detections, args.save)
    print(
        json.dumps(
            {
                "detections": detections,
                "best_detection": best_candidate,
                "threshold": args.threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
