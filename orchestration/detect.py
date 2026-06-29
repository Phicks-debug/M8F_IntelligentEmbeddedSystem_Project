import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def letterbox(image, size=640):
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (round(width * scale), round(height * scale))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_size[0]) // 2
    pad_y = (size - new_size[1]) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def detect(image_path, model_path, threshold=0.60):
    import onnxruntime as ort

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxed, scale, pad_x, pad_y = letterbox(image)

    batch = np.asarray(boxed, dtype=np.float32) / 255.0
    batch = batch.transpose(2, 0, 1)[None]

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = np.asarray(session.run(None, {input_name: batch})[0])

    results = []
    for row in output[0]:
        x1, y1, x2, y2, confidence, class_id = row.tolist()
        if confidence < threshold:
            continue

        box = [
            max(0, min(width, (x1 - pad_x) / scale)),
            max(0, min(height, (y1 - pad_y) / scale)),
            max(0, min(width, (x2 - pad_x) / scale)),
            max(0, min(height, (y2 - pad_y) / scale)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        results.append(
            {
                "label": "mushroom",
                "confidence": float(confidence),
                "class_id": int(class_id),
                "box": box,
            }
        )

    return sorted(results, key=lambda item: item["confidence"], reverse=True)


def crop(image_path, detection, padding=0.08):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = detection["box"]
    pad = max(x2 - x1, y2 - y1) * padding
    box = (
        int(max(0, x1 - pad)),
        int(max(0, y1 - pad)),
        int(min(width, x2 + pad)),
        int(min(height, y2 + pad)),
    )
    return image.crop(box)


def save_boxes(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, y1 + 4), label, fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run mushroom detection only.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exported_models/detection.onnx"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    detections = detect(args.image, args.model, args.threshold)
    if args.save:
        save_boxes(args.image, detections, args.save)
    print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
