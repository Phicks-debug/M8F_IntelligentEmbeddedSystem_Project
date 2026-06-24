"""Build the Hailo calibration + evaluation arrays for the FP32 ONNX model.

Runs on ANY machine that can run the training stack (macOS/MPS is fine) — it
does NOT need the Hailo SDK. It only prepares the .npy files the Hailo host
consumes during quantization (`optimize`) and accuracy testing.

It reuses the EXACT eval preprocessing from `classification.src.core`
(Resize(256) -> CenterCrop(224) -> /255 -> ImageNet normalize), so the
distribution Hailo calibrates on is identical to what the network was trained
and validated on. Mismatched preprocessing here is the #1 cause of a large
post-quantization accuracy drop.

IMPORTANT — this is NOT the lab's [0,1] calibration. The lab's
`capture_calib_dataset.py` emits NHWC float32 in [0,1], which is only correct
for a model whose ONNX input is [0,1]. `mobilenetv4.onnx` has NO preprocessing
in the graph: its input is ImageNet-NORMALIZED (~[-2.6, 2.6]). So calibration
here is normalized too, and on-device inference must apply the SAME normalize
before feeding the HEF (see classification/hailo/eval_on_pi.py).

Outputs (into classification/hailo/artifacts/):
  calib_nhwc.npy   float32 (N, 224, 224, 3)  -> Hailo `optimize` calibration set
  test_nhwc.npy    float32 (M, 224, 224, 3)  -> accuracy eval inputs
  test_labels.npy  int64   (M,)              -> accuracy eval ground-truth
  classes.json     index -> class name (ImageFolder order)

Layout note: arrays are saved NHWC because the Hailo Dataflow Compiler expects
calibration/inference data in NHWC, even though the ONNX input is NCHW. The
parser handles the transpose; the data you hand `optimize()` must be NHWC.

Usage:
    .venv/bin/python classification/hailo/build_calib.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from classification.src.core import get_dataloaders  # noqa: E402

CONFIG = Path("classification/configs/config.yaml")
OUT_DIR = Path("classification/hailo/artifacts")


def _collect(loader, max_batches: int) -> tuple[np.ndarray, np.ndarray]:
    """Drain up to `max_batches` from a loader into NHWC float32 + labels."""
    imgs, labels = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        # x is NCHW, already normalized float32. Hailo wants NHWC.
        imgs.append(x.permute(0, 2, 3, 1).contiguous().cpu().numpy())
        labels.append(y.cpu().numpy())
    return (
        np.concatenate(imgs).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    hailo_cal = cfg["export"]["hailo"]["calibration"]
    split = hailo_cal.get("split", "val")
    batch_size = int(hailo_cal.get("batch_size", 8))
    batches = int(hailo_cal.get("batches", 32))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.get("seed", 0)))

    train, val, test, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=batch_size,
        workers=0,
        mixup=False,
    )
    split_loader = {"train": train, "val": val, "test": test}[split]

    # Calibration set: from the configured split (default val), `batches` worth.
    calib, _ = _collect(split_loader, batches)
    np.save(OUT_DIR / "calib_nhwc.npy", calib)

    # Eval set: the FULL test split, so the quantized accuracy is comparable to
    # the FP32 / onnxruntime-INT8 numbers in exported_models/onnx_validation.json.
    test_x, test_y = _collect(test, max_batches=10**9)
    np.save(OUT_DIR / "test_nhwc.npy", test_x)
    np.save(OUT_DIR / "test_labels.npy", test_y)

    classes = getattr(test.dataset, "classes", None)
    if classes is None:
        raise AttributeError(
            f"Dataset {type(test.dataset).__name__!r} has no .classes attribute; "
            "expected torchvision.datasets.ImageFolder."
        )
    (OUT_DIR / "classes.json").write_text(
        json.dumps({i: c for i, c in enumerate(classes)}, indent=2)
    )

    print(f"calib_nhwc.npy   {calib.shape}  ({split} split)")
    print(f"test_nhwc.npy    {test_x.shape}")
    print(f"test_labels.npy  {test_y.shape}  ({len(classes)} classes)")
    print(f"value range      [{calib.min():.3f}, {calib.max():.3f}] (normalized)")
    print(f"written to       {OUT_DIR}")


if __name__ == "__main__":
    main()
