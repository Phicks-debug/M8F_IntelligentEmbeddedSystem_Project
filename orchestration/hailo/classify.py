import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from orchestration.classify import class_names, format_results, preprocess
from orchestration.edibility import edibility_for
from orchestration.hailo.runtime import HailoModel


def softmax(values):
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


def classify_image(image, classifier, names, topk=3):
    batch = preprocess(image)[None].astype(np.float32)
    logits = np.asarray(classifier.infer(batch)).reshape(-1)
    probabilities = softmax(logits)
    indexes = np.argsort(probabilities)[::-1][: min(topk, len(names))]

    results = []
    for index in indexes:
        class_name = names[int(index)]
        edible = edibility_for(class_name)
        results.append(
            {
                "class_name": class_name,
                "name": class_name.replace("_", " "),
                "confidence": float(probabilities[index]),
                "edibility": edible["label"],
                "note": edible["note"],
            }
        )
    return results


def classify(image_path, classifier, names, topk=3):
    image = Image.open(image_path).convert("RGB")
    return classify_image(image, classifier, names, topk)


def main():
    parser = argparse.ArgumentParser(
        description="Run mushroom species classification with Hailo HEF."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exported_models/mobilenetv4.hef"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with HailoModel(args.model) as classifier:
        results = classify(
            args.image,
            classifier,
            class_names(args.data_dir),
            args.topk,
        )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
