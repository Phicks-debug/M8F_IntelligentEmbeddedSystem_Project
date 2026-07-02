import argparse
import json
from pathlib import Path

from PIL import Image

from orchestration.hailo.detect import detect_image
from orchestration.hailo.main import run
from orchestration.hailo.runtime import HailoModel
from orchestration.video import format_video_output, frame_score, save_frame


def best_video_frame(video_path, detector, threshold, sample_seconds, output_path):
    import cv2

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * sample_seconds))
    best = None
    frame_index = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        detections, best_candidate = detect_image(
            image,
            detector,
            threshold,
            return_best=True,
        )
        score = frame_score(detections, best_candidate)
        if best is None or score > best["score"]:
            best = {
                "frame": image.copy(),
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "score": score,
                "detections": detections,
                "best_detection": best_candidate,
            }

        frame_index += 1

    video.release()
    if best is None:
        raise ValueError(f"No frames found in video: {video_path}")

    save_frame(best["frame"], output_path)
    best["image_path"] = str(output_path)
    best.pop("frame")
    return best


def run_video(
    video_path,
    detector,
    classifier,
    data_dir,
    threshold,
    padding,
    topk,
    sample_seconds,
    output_dir,
):
    stem = video_path.stem
    best_frame_path = output_dir / f"{stem}_best_frame.jpg"
    detection_path = output_dir / f"{stem}_detection.jpg"

    best = best_video_frame(
        video_path,
        detector,
        threshold,
        sample_seconds,
        best_frame_path,
    )
    result = run(
        best_frame_path,
        detector,
        classifier,
        data_dir,
        threshold,
        padding,
        topk,
        detection_path,
    )
    result["video"] = str(video_path)
    result["best_frame"] = best
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pick the best video frame and run Hailo HEF inference."
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
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with HailoModel(args.detector) as detector:
        with HailoModel(args.classifier) as classifier:
            result = run_video(
                args.video,
                detector,
                classifier,
                args.data_dir,
                args.threshold,
                args.padding,
                args.topk,
                args.sample_seconds,
                args.output_dir,
            )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_video_output(result))


if __name__ == "__main__":
    main()
