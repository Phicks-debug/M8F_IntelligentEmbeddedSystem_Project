import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from orchestration.edibility import SAFETY_WARNING, edibility_for

IMAGE_SIZE = 224
RESIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def class_names(data_dir):
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training split: {train_dir}")
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def preprocess(image):
    image = image.convert("RGB")
    width, height = image.size
    scale = RESIZE / min(width, height)
    image = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BILINEAR,
    )

    width, height = image.size
    left = (width - IMAGE_SIZE) // 2
    top = (height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - IMAGENET_MEAN) / IMAGENET_STD


def classify(image, model_path, names, device="cpu", topk=3):
    array = preprocess(image)
    batch = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]

    results = []
    for confidence, index in zip(*probabilities.topk(min(topk, len(names)))):
        class_name = names[int(index)]
        edible = edibility_for(class_name)
        results.append(
            {
                "class_name": class_name,
                "name": class_name.replace("_", " "),
                "confidence": float(confidence),
                "edibility": edible["label"],
                "note": edible["note"],
            }
        )
    return results


def format_results(results):
    best = results[0]
    lines = [
        f"Predicted species: {best['name']}",
        f"Confidence: {best['confidence'] * 100:.2f}%",
        f"Edibility: {best['edibility']}",
        f"Note: {best['note']}",
    ]

    hidden = 0
    other_lines = []
    for result in results[1:]:
        if result["confidence"] < 0.01:
            hidden += 1
        else:
            other_lines.append(f"- {result['name']}: {result['confidence'] * 100:.2f}%")

    if other_lines or hidden:
        lines.append("")
        lines.append("Other candidates:")
        lines.extend(other_lines)
        if hidden:
            lines.append(f"- {hidden} other candidates have less than 1% confidence.")

    lines.append("")
    lines.append(f"Safety warning: {SAFETY_WARNING}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run mushroom species classification only.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exported_models/mobilenetv4.pt"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image)
    results = classify(
        image,
        args.model,
        class_names(args.data_dir),
        device=args.device,
        topk=args.topk,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
