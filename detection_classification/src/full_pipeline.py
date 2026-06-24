"""
Full mushroom image pipeline: detect first, then classify the detected crop.

The detector is an ONNX YOLO model with one class: ``mushroom``. It finds a
bounding box, crops that region from the original image, and passes the crop to
the existing species classifier.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.edibility import SAFETY_WARNING
from src.inference import (
    format_prediction,
    load_class_names,
    load_torchscript_model,
    predict_pil_image,
)


@dataclass(frozen=True)
class Detection:
    """One detection box in original image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str = "mushroom"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "box": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
        }


def _letterbox(
    image: Image.Image,
    size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, int, int]:
    """Resize image to a square with YOLO-style padding."""

    width, height = image.size
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), color)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _preprocess_detection_image(
    image: Image.Image,
    input_size: int = 640,
) -> tuple[np.ndarray, float, int, int]:
    """Convert a PIL image into detector input tensor."""

    boxed, scale, pad_x, pad_y = _letterbox(image.convert("RGB"), input_size)
    arr = np.asarray(boxed, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return arr, scale, pad_x, pad_y


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def run_detection_onnx(
    image_path: Path,
    detector_path: Path,
    *,
    confidence_threshold: float = 0.25,
    input_size: int = 640,
) -> list[Detection]:
    """Run the ONNX detector and return boxes in original image coordinates."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX detection requires onnxruntime. Install classification requirements "
            "or run: pip install onnxruntime"
        ) from exc

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    batch, scale, pad_x, pad_y = _preprocess_detection_image(image, input_size)

    session = ort.InferenceSession(
        str(detector_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: batch})[0]

    detections: list[Detection] = []
    for row in output[0]:
        x1, y1, x2, y2, confidence, class_id = row.tolist()
        if confidence < confidence_threshold:
            continue

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2), width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidence),
                class_id=int(class_id),
            )
        )

    detections.sort(key=lambda det: det.confidence, reverse=True)
    return detections


def crop_detection(
    image_path: Path,
    detection: Detection,
    *,
    padding: float = 0.08,
) -> Image.Image:
    """Crop a detection box from the original image with optional padding."""

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box_width = detection.x2 - detection.x1
    box_height = detection.y2 - detection.y1
    pad = padding * max(box_width, box_height)
    x1, y1, x2, y2 = _clip_box(
        (
            detection.x1 - pad,
            detection.y1 - pad,
            detection.x2 + pad,
            detection.y2 + pad,
        ),
        width,
        height,
    )
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def run_full_pipeline(
    image_path: Path,
    *,
    detector_path: Path,
    classifier_path: Path,
    data_dir: Path,
    detection_threshold: float = 0.25,
    crop_padding: float = 0.08,
    image_size: int = 224,
    device: str = "cpu",
    topk: int = 3,
) -> dict[str, Any]:
    """Detect mushroom, crop best box, classify species, and map edibility."""

    detections = run_detection_onnx(
        image_path,
        detector_path,
        confidence_threshold=detection_threshold,
    )
    if not detections:
        return {
            "image": str(image_path),
            "detection": None,
            "classification": None,
            "decision": "no_mushroom_detected",
            "message": "No mushroom detection passed the confidence threshold.",
            "safety_warning": SAFETY_WARNING,
        }

    best = detections[0]
    crop = crop_detection(image_path, best, padding=crop_padding)
    class_names = load_class_names(data_dir)
    classifier = load_torchscript_model(classifier_path, device)
    classification = predict_pil_image(
        crop,
        classifier,
        class_names,
        image_label=f"{image_path} [crop from detection box]",
        image_size=image_size,
        device=device,
        topk=topk,
    )

    return {
        "image": str(image_path),
        "detection": best.to_dict(),
        "num_detections": len(detections),
        "classification": classification,
        "decision": "classified",
        "safety_warning": SAFETY_WARNING,
    }


def format_full_pipeline_result(result: dict[str, Any]) -> str:
    """Return readable CLI output for the full pipeline."""

    lines = [
        "Full mushroom pipeline result",
        "=" * 29,
        f"Image: {result['image']}",
    ]

    detection = result.get("detection")
    if detection is None:
        lines.extend(
            [
                "Detection: no mushroom found",
                "",
                result["message"],
                "",
                f"Safety warning: {result['safety_warning']}",
            ]
        )
        return "\n".join(lines)

    box = detection["box"]
    lines.extend(
        [
            (
                f"Detection: {detection['label']} "
                f"({detection['confidence'] * 100:.2f}%)"
            ),
            (
                "Detection box: "
                f"({box['x1']:.1f}, {box['y1']:.1f}) -> "
                f"({box['x2']:.1f}, {box['y2']:.1f})"
            ),
            f"Detections above threshold: {result['num_detections']}",
            "",
            format_prediction(result["classification"]),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect a mushroom, classify species, and report edibility."
    )
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("exported_models/detection.onnx"),
        help="Path to ONNX mushroom detector.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("exported_models/mobilenetv4.pt"),
        help="Path to TorchScript species classifier.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
        help="ImageFolder dataset root used to recover species class order.",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the human-readable summary.",
    )
    args = parser.parse_args()

    result = run_full_pipeline(
        args.image,
        detector_path=args.detector,
        classifier_path=args.classifier,
        data_dir=args.data_dir,
        detection_threshold=args.detection_threshold,
        crop_padding=args.crop_padding,
        image_size=args.image_size,
        device=args.device,
        topk=args.topk,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_full_pipeline_result(result))


if __name__ == "__main__":
    main()
