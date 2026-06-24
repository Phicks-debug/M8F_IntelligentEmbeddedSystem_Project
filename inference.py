"""Unified inference CLI for the mushroom classification model.

Selects the runtime backend from the model file extension:

    .pt          TorchScript (PyTorch)
    .onnx        ONNX Runtime
    .hex / .hef  Hailo (HailoRT, runs on the Hailo accelerator)

The script prints the model's raw output (logits) only. It does not map
indices to class names or edibility; that is the caller's responsibility.

Example:
    python inference.py path/to/image.jpg --model exported_models/mobilenetv4.onnx
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224  # size fed to the model, after center crop
RESIZE = 256  # shorter side is scaled to this before cropping

# ImageNet normalization stats the model was trained with (per RGB channel).
# They are standard ImageNet statistics, duplicate from the training pipeline
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICES = ["cpu", "cuda", "mps"]


def preprocess(image_path: Path, normalize: bool = True) -> np.ndarray:
    """Resize (shorter side), center-crop, and optionally normalize an image.

    Returns an HWC float32 array. When normalize is False the array holds
    raw [0, 1] pixels (Hailo models typically normalize on-chip).
    """
    image = Image.open(image_path).convert("RGB")

    width, height = image.size
    scale = RESIZE / min(width, height)
    image = image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
    )

    width, height = image.size
    left = (width - IMAGE_SIZE) // 2
    top = (height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    if normalize:
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array


def run_torchscript(model_path: Path, image_path: Path, device: str) -> np.ndarray:
    """Run a TorchScript (.pt) model."""
    import torch

    array = preprocess(image_path)  # HWC, normalized
    batch = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    with torch.inference_mode():
        output = model(batch)
    return output.cpu().numpy()


def run_onnx(model_path: Path, image_path: Path, device: str) -> np.ndarray:
    """Run an ONNX (.onnx) model."""
    import onnxruntime as ort

    array = preprocess(image_path)  # HWC, normalized
    batch = array.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # NCHW

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: batch})[0]
    return np.asarray(output)


def run_hailo(model_path: Path, image_path: Path, device: str) -> np.ndarray:
    """Run a Hailo (.hex/.hef) model on the Hailo accelerator."""
    from hailo_platform import (  # type: ignore[import-not-found]
        HEF,
        ConfigureParams,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    array = preprocess(image_path, normalize=False)  # HWC, [0, 1]
    batch = (array * 255.0).round().astype(np.uint8)[np.newaxis]  # NHWC uint8

    hef = HEF(str(model_path))
    input_name = hef.get_input_vstream_infos()[0].name
    output_name = hef.get_output_vstream_infos()[0].name

    with VDevice() as target:
        params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, params)[0]
        input_params = InputVStreamParams.make(network_group)
        output_params = OutputVStreamParams.make(network_group)
        with (
            network_group.activate(),
            InferVStreams(network_group, input_params, output_params) as pipeline,
        ):
            output = pipeline.infer({input_name: batch})[output_name]
    return np.asarray(output)


# Dispatch table: model extension to backend runner.
RUNNERS = {
    ".pt": run_torchscript,
    ".onnx": run_onnx,
    ".hex": run_hailo,
    ".hef": run_hailo,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run .pt, .onnx, or .hex/.hef model.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument("--model", type=Path, required=True, help="Model file path.")
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    args = parser.parse_args()

    runner = RUNNERS.get(args.model.suffix.lower())
    if runner is None:
        parser.error(
            f"Unsupported model type '{args.model.suffix}'. "
            f"Supported: {', '.join(sorted(RUNNERS))}."
        )

    output = runner(args.model, args.image, args.device)
    np.set_printoptions(precision=6, suppress=True)
    print(np.asarray(output).ravel())


if __name__ == "__main__":
    main()
