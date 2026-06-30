import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from orchestration.main import format_output, run
from orchestration.detect import detect


def frame_score(detections, best_candidate):
    if detections:
        return detections[0]["confidence"]
    if best_candidate:
        return best_candidate["confidence"]
    return 0.0


def save_frame(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output_path)
    return output_path


def best_video_frame(video_path, detector_path, threshold, sample_seconds, output_path):
    import cv2

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * sample_seconds))
    best = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        frame_index = 0

        while True:
            ok, frame = video.read()
            if not ok:
                break

            if frame_index % step != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            temp_path = temp_dir / f"frame_{frame_index}.jpg"
            image.save(temp_path)

            detections, best_candidate = detect(
                temp_path,
                detector_path,
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
    detector_path,
    classifier_path,
    data_dir,
    threshold,
    padding,
    device,
    topk,
    sample_seconds,
    output_dir,
):
    stem = video_path.stem
    best_frame_path = output_dir / f"{stem}_best_frame.jpg"
    detection_path = output_dir / f"{stem}_detection.jpg"

    best = best_video_frame(
        video_path,
        detector_path,
        threshold,
        sample_seconds,
        best_frame_path,
    )
    result = run(
        best_frame_path,
        detector_path,
        classifier_path,
        data_dir,
        threshold,
        padding,
        device,
        topk,
        detection_path,
    )
    result["video"] = str(video_path)
    result["best_frame"] = best
    return result


def format_video_output(result):
    best = result["best_frame"]
    lines = [
        f"Video: {result['video']}",
        f"Best frame: {best['frame_index']} at {best['time_seconds']:.2f}s",
        f"Saved best frame: {best['image_path']}",
        "",
        format_output(result),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Pick the best mushroom frame from a video and run the image pipeline."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("exported_models/detection.onnx"),
    )
    parser.add_argument("--classifier", type=Path, default=Path("exported_models/mobilenetv4.pt"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/classification_data"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_video(
        args.video,
        args.detector,
        args.classifier,
        args.data_dir,
        args.threshold,
        args.padding,
        args.device,
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
