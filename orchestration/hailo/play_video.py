import argparse
import json
import time
from pathlib import Path

from PIL import Image

from orchestration.classify import class_names
from orchestration.hailo.camera import crop_image, draw_preview, format_live_result
from orchestration.hailo.classify import classify_image
from orchestration.hailo.detect import detect_image
from orchestration.hailo.runtime import HailoDevice, HailoModel


def resize_frame(image, width):
    if not width:
        return image

    current_width, current_height = image.size
    scale = width / current_width
    height = round(current_height * scale)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def image_to_bgr(image):
    import cv2
    import numpy as np

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run_inference(image, detector, classifier, names, args, frame_number):
    detections, best_detection = detect_image(
        image,
        detector,
        args.threshold,
        return_best=True,
        input_scale=args.detector_input_scale,
        debug=args.debug_detector and frame_number == 0,
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

    return detections, best_detection, predictions


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
    writer = None
    last_print = 0.0
    frame_number = 0
    detections = []
    best_detection = None
    predictions = None

    try:
        with HailoDevice() as device:
            with HailoModel(args.detector, device) as detector:
                with HailoModel(args.classifier, device) as classifier:
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
                                names,
                                args,
                                frame_number,
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
                                writer = open_video_writer(
                                    args.save_video,
                                    fps,
                                    (width, height),
                                )
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
        description="Play a prerecorded video with live Hailo detection/classification overlays."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--detector", type=Path, default=Path("exported_models/detection.hef"))
    parser.add_argument("--classifier", type=Path, default=Path("exported_models/mobilenetv4.hef"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--detector-input-scale", type=float, default=1 / 255)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--print-seconds", type=float, default=1.0)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--window-name", default="Mushroom video inference")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug-detector", action="store_true")
    args = parser.parse_args()

    args.detect_every = max(1, args.detect_every)
    play_video(args)


if __name__ == "__main__":
    main()
