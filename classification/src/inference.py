"""
Single-image inference for the mushroom classification model.

Example:
    python -m src.inference path/to/image.jpg \
        --model exported_models/mobilenetv4.pt \
        --data-dir data/processed/classification_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import datasets

from src.core import build_transforms
from src.edibility import SAFETY_WARNING, get_edibility, validate_edibility_map


def load_class_names(data_dir: Path) -> list[str]:
    """Read class names in the same order used by torchvision ImageFolder."""

    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training split directory: {train_dir}")
    return datasets.ImageFolder(train_dir).classes


def load_torchscript_model(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Load a TorchScript exported model."""

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


@torch.inference_mode()
def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    *,
    image_size: int = 224,
    device: str = "cpu",
    topk: int = 3,
) -> dict[str, Any]:
    """Predict species and edibility for one image."""

    validate_edibility_map(class_names)

    transform = build_transforms(image_size, is_train=False)
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)

    model = model.to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, len(class_names))
    confidences, indices = probs.topk(k)
    predictions = []
    for confidence, index in zip(confidences.tolist(), indices.tolist()):
        class_name = class_names[index]
        edibility = get_edibility(class_name)
        predictions.append(
            {
                "class_name": class_name,
                "display_name": class_name.replace("_", " "),
                "confidence": confidence,
                "edibility_category": edibility.category,
                "edibility_label": edibility.label,
                "edibility_note": edibility.note,
            }
        )

    result = {
        "image": str(image_path),
        "top_prediction": predictions[0],
        "topk": predictions,
        "safety_warning": SAFETY_WARNING,
    }
    return result


def format_prediction(result: dict[str, Any], *, min_candidate_confidence: float = 0.01) -> str:
    """Return a readable CLI summary with confidence percentages."""

    top = result["top_prediction"]
    lines = [
        "Mushroom classification result",
        "=" * 30,
        f"Image: {result['image']}",
        f"Predicted species: {top['display_name']}",
        f"Confidence: {top['confidence'] * 100:.2f}%",
        f"Edibility: {top['edibility_label']}",
        f"Note: {top['edibility_note']}",
    ]

    other_predictions = result["topk"][1:]
    visible_others = [
        pred
        for pred in other_predictions
        if pred["confidence"] >= min_candidate_confidence
    ]
    hidden_count = len(other_predictions) - len(visible_others)

    if visible_others or hidden_count:
        lines.extend(["", "Other candidates:"])
        for pred in visible_others:
            lines.append(
                f"- {pred['display_name']}: {pred['confidence'] * 100:.2f}% "
                f"({pred['edibility_label']})"
            )
        if hidden_count:
            plural = "candidate has" if hidden_count == 1 else "candidates have"
            lines.append(
                f"- {hidden_count} other {plural} less than "
                f"{min_candidate_confidence * 100:.0f}% confidence."
            )

    lines.extend(["", f"Safety warning: {result['safety_warning']}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mushroom species and edibility.")
    parser.add_argument("image", type=Path, help="Path to an input mushroom image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("exported_models/mobilenetv4.pt"),
        help="Path to TorchScript .pt model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
        help="ImageFolder dataset root used to recover class order.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the human-readable summary.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.data_dir)
    model = load_torchscript_model(args.model, args.device)
    result = predict_image(
        args.image,
        model,
        class_names,
        image_size=args.image_size,
        device=args.device,
        topk=args.topk,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_prediction(result))


if __name__ == "__main__":
    main()
