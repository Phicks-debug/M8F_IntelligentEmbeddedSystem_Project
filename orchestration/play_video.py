import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from orchestration.classify import class_names, classify, preprocess
from orchestration.detect import letterbox
from orchestration.edibility import edibility_for


def resize_frame(image, width):
    if not width:
        return image

    current_width, current_height = image.size
    scale = width / current_width
    height = round(current_height * scale)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def image_to_bgr(image):
    import cv2

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def crop_image(image, detection, padding=0.08):
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


class OnnxDetector:
    def __init__(self, model_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image, threshold=0.60, return_best=False):
        width, height = image.size
        boxed, scale, pad_x, pad_y = letterbox(image.convert("RGB"))

        batch = np.asarray(boxed, dtype=np.float32) / 255.0
        batch = batch.transpose(2, 0, 1)[None]
        output = np.asarray(self.session.run(None, {self.input_name: batch})[0])

        results = []
        best_candidate = None
        for row in output[0]:
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


class OnnxClassifier:
    def __init__(self, model_path, names):
        import onnxruntime as ort

        self.names = names
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, image, topk=3):
        array = preprocess(image)
        batch = array.transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0][0]
        probabilities = softmax(logits)
        indices = probabilities.argsort()[::-1][: min(topk, len(self.names))]

        results = []
        for index in indices:
            class_name = self.names[int(index)]
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


class TorchScriptClassifier:
    def __init__(self, model_path, names, device):
        self.model_path = model_path
        self.names = names
        self.device = device

    def classify(self, image, topk=3):
        return classify(
            image,
            self.model_path,
            self.names,
            device=self.device,
            topk=topk,
        )


def softmax(values):
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def load_classifier(model_path, names, device):
    if model_path.suffix.lower() == ".onnx":
        return OnnxClassifier(model_path, names)
    return TorchScriptClassifier(model_path, names, device)


def run_inference(image, detector, classifier, args):
    detections, best_detection = detector.detect(
        image,
        args.threshold,
        return_best=True,
    )

    predictions = None
    if detections:
        mushroom_crop = crop_image(image, detections[0], args.padding)
        predictions = classifier.classify(mushroom_crop, topk=args.topk)

    return detections, best_detection, predictions


def draw_preview(image, detections, best_detection, predictions, threshold):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    shown_detections = detections
    if not shown_detections and best_detection:
        shown_detections = [best_detection]

    for detection in shown_detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        color = "red" if detection in detections else "orange"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1 + 4, y1 + 4), label, fill=color)

    if detections and predictions:
        detection = detections[0]
        best = predictions[0]
        lines = [
            f"Detection: mushroom {detection['confidence'] * 100:.1f}%",
            f"Species: {best['name']} {best['confidence'] * 100:.1f}%",
            f"Edibility: {best['edibility']}",
        ]
    elif best_detection:
        lines = [
            "Detection: no mushroom",
            f"Best guess: {best_detection['confidence'] * 100:.1f}%",
            f"Threshold: {threshold * 100:.1f}%",
        ]
    else:
        lines = ["Detection: no mushroom"]

    y = 16
    for line in lines:
        draw.rectangle((12, y - 4, 560, y + 22), fill="black")
        draw.text((20, y), line, fill="white")
        y += 30

    return image


def format_live_result(frame_number, detections, best_detection, predictions, threshold):
    if not detections:
        if best_detection:
            return (
                f"Frame {frame_number}: no mushroom "
                f"(best {best_detection['confidence'] * 100:.2f}%, "
                f"threshold {threshold * 100:.2f}%)"
            )
        return f"Frame {frame_number}: no mushroom"

    detection = detections[0]
    best = predictions[0]
    return (
        f"Frame {frame_number}: mushroom {detection['confidence'] * 100:.2f}% | "
        f"{best['name']} {best['confidence'] * 100:.2f}% | "
        f"{best['edibility']}"
    )


def open_video_writer(output_path, fps, frame_size):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), codec, fps, frame_size)


def play_video(args):
    import cv2

    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {args.video}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = max(1, round(1000 / fps))
    names = class_names(args.data_dir)
    detector = OnnxDetector(args.detector)
    classifier = load_classifier(args.classifier, names, args.device)
    writer = None
    last_print = 0.0
    frame_number = 0
    detections = []
    best_detection = None
    predictions = None

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).convert("RGB")
            image = resize_frame(image, args.width)

            if frame_number % args.detect_every == 0:
                detections, best_detection, predictions = run_inference(
                    image,
                    detector,
                    classifier,
                    args,
                )

            preview = draw_preview(
                image,
                detections,
                best_detection,
                predictions,
                args.threshold,
            )
            preview_frame = image_to_bgr(preview)

            if args.save_video:
                if writer is None:
                    height, width = preview_frame.shape[:2]
                    writer = open_video_writer(args.save_video, fps, (width, height))
                writer.write(preview_frame)

            if args.display:
                cv2.imshow(args.window_name, preview_frame)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_print >= args.print_seconds:
                last_print = now
                print(
                    format_live_result(
                        frame_number,
                        detections,
                        best_detection,
                        predictions,
                        args.threshold,
                    )
                )

            if args.json:
                print(
                    json.dumps(
                        {
                            "frame": frame_number,
                            "detections": detections,
                            "best_detection": best_detection,
                            "classification": predictions,
                        }
                    )
                )

            frame_number += 1
    except KeyboardInterrupt:
        print("Stopped video playback inference.")
    finally:
        video.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Play a prerecorded video with local ONNX detection/classification overlays."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--detector", type=Path, default=Path("exported_models/detection.onnx"))
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("exported_models/mobilenetv4_int8.onnx"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--print-seconds", type=float, default=1.0)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--window-name", default="Mushroom video inference")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.detect_every = max(1, args.detect_every)
    play_video(args)


if __name__ == "__main__":
    main()
