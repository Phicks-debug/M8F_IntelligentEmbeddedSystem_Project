"""FP32 baseline accuracy on the SAME arrays the Hailo retest uses.

Runs anywhere onnxruntime is installed (macOS is fine) — no Hailo SDK needed.
It feeds classification/hailo/artifacts/test_nhwc.npy through the FP32 ONNX and
reports top-1 accuracy. This is the reference number the quantized Hailo model
(emulator on the workstation, then the HEF on the Pi) must be compared against:

    FP32 (here)        -> baseline
    Hailo emulator     -> quantize_and_eval.py on the x86 host
    Hailo HEF on-device-> eval_on_pi.py on the Pi

If all three feed identical preprocessing, any accuracy gap is pure
quantization error — which is exactly what we want to measure.

Usage:
    .venv/bin/python classification/hailo/eval_fp32_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ART = Path("classification/hailo/artifacts")
ONNX = Path("exported_models/mobilenetv4.onnx")


def main() -> None:
    test_x = np.load(ART / "test_nhwc.npy")  # (M, 224, 224, 3) normalized
    test_y = np.load(ART / "test_labels.npy")
    classes = json.loads((ART / "classes.json").read_text())

    # ONNX input is NCHW; our arrays are NHWC -> transpose back.
    nchw = np.transpose(test_x, (0, 3, 1, 2)).astype(np.float32)

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    preds = []
    for i in range(0, len(nchw), 64):
        logits = np.asarray(sess.run(None, {in_name: nchw[i : i + 64]})[0])
        preds.append(logits.argmax(1))
    preds = np.concatenate(preds)

    acc = float((preds == test_y).mean())
    print(f"FP32 ONNX top-1 accuracy: {acc:.4f}  (n={len(test_y)})")

    # Per-class accuracy — useful to spot a class that quantization will later
    # collapse (the classic INT8 failure mode).
    print("\nper-class:")
    for idx, name in classes.items():
        mask = test_y == int(idx)
        if mask.any():
            print(f"  {int(idx)} {name:<24} {float((preds[mask] == test_y[mask]).mean()):.3f}"
                  f"  (n={int(mask.sum())})")


if __name__ == "__main__":
    main()
