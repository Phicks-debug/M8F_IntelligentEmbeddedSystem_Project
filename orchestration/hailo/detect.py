import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from orchestration.detect import letterbox, save_boxes
from orchestration.hailo.runtime import HailoModel


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0
    return intersection / union


def non_max_suppression(candidates, iou_threshold=0.45):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        if all(box_iou(candidate[:4], kept_item[:4]) < iou_threshold for kept_item in kept):
            kept.append(candidate)
    return kept


def decode_yolo_output(output, image_size=640):
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    if output.ndim == 2 and output.shape[0] in (5, 6):
        output = output.T

    if output.ndim != 2:
        raise ValueError(f"Unsupported detection output shape: {list(output.shape)}")

    if output.shape[1] == 6:
        return output

    if output.shape[1] != 5:
        raise ValueError(f"Unsupported detection output shape: {list(output.shape)}")

    x_center = output[:, 0]
    y_center = output[:, 1]
    width = output[:, 2]
    height = output[:, 3]
    confidence = output[:, 4]

    if confidence.size and (confidence.min() < 0 or confidence.max() > 1):
        confidence = sigmoid(confidence)

    if max(x_center.max(), y_center.max(), width.max(), height.max()) <= 1.5:
        x_center = x_center * image_size
        y_center = y_center * image_size
        width = width * image_size
        height = height * image_size

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    class_id = np.zeros_like(confidence)
    rows = np.stack([x1, y1, x2, y2, confidence, class_id], axis=1)
    return non_max_suppression(rows.tolist())


def detection_rows(output):
    if isinstance(output, dict):
        arrays = [np.asarray(value) for value in output.values()]
        candidates = [array for array in arrays if array.size and 5 in array.shape]
        if not candidates:
            shapes = {
                name: list(np.asarray(value).shape)
                for name, value in output.items()
            }
            raise ValueError(f"Could not find detection output with 5 or 6 values: {shapes}")
        output = max(candidates, key=lambda array: array.size)

    return decode_yolo_output(output)


def detect_image(image, detector, threshold=0.60, return_best=False):
    width, height = image.size
    boxed, scale, pad_x, pad_y = letterbox(image.convert("RGB"))

    batch = np.asarray(boxed, dtype=np.float32) / 255.0
    batch = batch[None]
    output = detector.infer(batch)

    results = []
    best_candidate = None
    for row in detection_rows(output):
        x1, y1, x2, y2, confidence, class_id = row
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
