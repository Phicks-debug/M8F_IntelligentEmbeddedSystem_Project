"""Quantize the FP32 ONNX with Hailo DFC, measure emulated accuracy, compile HEF.

RUN THIS ON THE x86 LINUX WORKSTATION that has Hailo Dataflow Compiler 5.3.0
(the `exportyolov26n` conda env from the lab guide). It does NOT run on macOS.

    conda activate exportyolov26n
    python3 classification/hailo/quantize_and_eval.py \
        --onnx exported_models/mobilenetv4.onnx \
        --calib classification/hailo/artifacts/calib_nhwc.npy \
        --test  classification/hailo/artifacts/test_nhwc.npy \
        --labels classification/hailo/artifacts/test_labels.npy \
        --hef   exported_models/mobilenetv4_mushroom_hailo10h.hef

This is the "retest" on the compiler side. It:
  1. translate_onnx_model  -> parse FP32 ONNX to Hailo HAR
  2. optimize(calib)       -> Hailo's OWN INT8 post-training quantization,
                              calibrated on YOUR normalized NHWC calib set
  3. infer (SDK_QUANTIZED) -> emulate the quantized net on the test set and
                              print top-1 accuracy (compare to FP32 = 0.8920)
  4. compile()             -> write the .hef for the Hailo-10H runtime

The calib/test arrays are ImageNet-normalized NHWC (built by build_calib.py) —
they are NOT the lab's [0,1] capture, because this ONNX has no preprocessing in
the graph. Feed the same normalized arrays at inference on the Pi.

Note on the ONNX: mobilenetv4.onnx stores weights externally in
mobilenetv4.onnx.data — keep both files together when copying to the host.
"""

import argparse
from pathlib import Path

import numpy as np

# Hailo SDK — only importable inside the DFC conda env.
from hailo_sdk_client import ClientRunner  # type: ignore
from hailo_sdk_client.exposed_definitions import InferenceContext  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--calib", required=True, help="NHWC float32 .npy (normalized)")
    p.add_argument("--test", required=True, help="NHWC float32 .npy (normalized)")
    p.add_argument("--labels", required=True, help="int64 .npy ground-truth")
    p.add_argument("--hef", default=None, help="output .hef path (skip compile if omitted)")
    p.add_argument("--hw-arch", default="hailo10h")
    p.add_argument("--model-name", default="mobilenetv4_mushroom")
    p.add_argument("--har", default=None, help="optional path to save the quantized HAR")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calib = np.load(args.calib).astype(np.float32)
    test = np.load(args.test).astype(np.float32)
    labels = np.load(args.labels)
    print(f"calib {calib.shape}  test {test.shape}  arch {args.hw_arch}")

    runner = ClientRunner(hw_arch=args.hw_arch)

    # 1. Parse. The ONNX input is 'input' (NCHW in the graph); the parser maps
    #    it to Hailo's NHWC. Let the parser auto-detect start/end nodes for a
    #    plain classifier — no head ops to cut (unlike the YOLO path in the guide).
    runner.translate_onnx_model(args.onnx, args.model_name)

    # 2. Hailo's own INT8 PTQ, calibrated on the normalized NHWC set.
    print("optimize(): Hailo INT8 calibration ...")
    runner.optimize(calib)

    if args.har:
        runner.save_har(args.har)
        print(f"saved quantized HAR -> {args.har}")

    # 3. Emulated quantized accuracy — the on-host preview of on-device accuracy.
    print("infer(SDK_QUANTIZED): emulating quantized net on test set ...")
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        logits = runner.infer(ctx, test)
    logits = np.asarray(logits).reshape(len(test), -1)
    preds = logits.argmax(1)
    acc = float((preds == labels).mean())
    print(f"\nHailo QUANTIZED (emulated) top-1: {acc:.4f}  (n={len(labels)})")
    print("compare to FP32 ONNX baseline 0.8920 — a drop > ~0.05 means revisit")
    print("calibration (more batches, more representative frames).")

    # 4. Compile to HEF.
    if args.hef:
        print("compile(): building HEF ...")
        hef = runner.compile()
        Path(args.hef).write_bytes(hef)
        print(f"wrote HEF -> {args.hef}")
        print(f"verify with:  hailortcli parse-hef {args.hef} | head")


if __name__ == "__main__":
    main()
