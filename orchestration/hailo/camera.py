import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from orchestration.classify import class_names
from orchestration.hailo.classify import classify_image
from orchestration.hailo.detect import detect_image
from orchestration.hailo.runtime import HailoDevice, HailoModel


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


def draw_preview(image, detections, best_detection, predictions, threshold):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, y1 + 4), label, fill="red")

    lines = []
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
        draw.rectangle((12, y - 4, 520, y + 22), fill="black")
        draw.text((20, y), line, fill="white")
        y += 30

    return image


def show_preview(image, window_name):
    import cv2
    import numpy as np

    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def open_picamera(width, height):
    from picamera2 import Picamera2  # type: ignore

    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1)
    return camera


def read_picamera(camera):
    return Image.fromarray(camera.capture_array()).convert("RGB")


def open_opencv_camera(index, width, height):
    import cv2

    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not camera.isOpened():
        raise ValueError(f"Could not open camera index {index}")
    return camera


def read_opencv_camera(camera):
    import cv2

    ok, frame = camera.read()
    if not ok:
        raise ValueError("Could not read camera frame")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame).convert("RGB")


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


def run_camera(args):
    names = class_names(args.data_dir)
    if args.backend == "picamera2":
        camera = open_picamera(args.width, args.height)
        read_frame = read_picamera
    else:
        camera = open_opencv_camera(args.camera_index, args.width, args.height)
        read_frame = read_opencv_camera

    last_print = 0.0
    frame_number = 0

    try:
        with HailoDevice() as device:
            with HailoModel(args.detector, device) as detector:
                with HailoModel(args.classifier, device) as classifier:
                    while args.frames <= 0 or frame_number < args.frames:
                        image = read_frame(camera)
                        detections, best_detection = detect_image(
                            image,
                            detector,
                            args.threshold,
                            return_best=True,
                        )

                        predictions = None
                        if detections:
                            mushroom_crop = crop_image(image, detections[0], args.padding)
                            predictions = classify_image(
                                mushroom_crop,
                                classifier,
                                names,
                                topk=args.topk,
                            )

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

                        preview = None
                        if args.display or args.save_latest:
                            preview = draw_preview(
                                image,
                                detections,
                                best_detection,
                                predictions,
                                args.threshold,
                            )

                        if args.display and not show_preview(preview, args.window_name):
                            break

                        if args.save_latest:
                            if preview is None:
                                preview = draw_preview(
                                    image,
                                    detections,
                                    best_detection,
                                    predictions,
                                    args.threshold,
                                )
                            args.save_latest.parent.mkdir(parents=True, exist_ok=True)
                            preview.save(args.save_latest)

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
        print("Stopped camera inference.")
    finally:
        if args.display:
            import cv2

            cv2.destroyAllWindows()
        if args.backend == "picamera2":
            camera.stop()
        else:
            camera.release()


def main():
    parser = argparse.ArgumentParser(
        description="Run live Raspberry Pi camera inference with Hailo HEF models."
    )
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
    parser.add_argument("--backend", choices=("picamera2", "opencv"), default="picamera2")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--print-seconds", type=float, default=1.0)
    parser.add_argument("--save-latest", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--window-name", default="Mushroom detection")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_camera(args)


if __name__ == "__main__":
    main()
