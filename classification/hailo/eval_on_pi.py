"""On-device Hailo-10H accuracy retest — run on the Raspberry Pi 5 + HAT+.

Needs HailoRT (the `hailo_platform` Python package) installed on the Pi, the
compiled .hef, and the test arrays from build_calib.py copied over:

    scp exported_models/mobilenetv4_mushroom_hailo10h.hef pi@<pi>:~/mushroom/
    scp classification/hailo/artifacts/test_nhwc.npy       pi@<pi>:~/mushroom/
    scp classification/hailo/artifacts/test_labels.npy     pi@<pi>:~/mushroom/

    # On the Pi
    python3 eval_on_pi.py \
        --hef mushroom/mobilenetv4_mushroom_hailo10h.hef \
        --test mushroom/test_nhwc.npy \
        --labels mushroom/test_labels.npy

This measures REAL silicon accuracy and compares it to the FP32 baseline
(0.8920). The test arrays are ImageNet-normalized NHWC — the HEF was compiled
WITHOUT input normalization, so we feed normalized floats here too. For live
camera inference, apply the identical Resize(256)->CenterCrop(224)->/255->
ImageNet-normalize before infer() (see _preprocess_frame at the bottom).

HailoRT API note: the classic InferVStreams path below is stable across 4.x.
Interface is PCIe for the HAT+. If your HailoRT differs, the InferModel API is
the modern alternative — the preprocessing/accuracy logic is unchanged.
"""

import argparse

import numpy as np
from hailo_platform import (  # type: ignore
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hef", required=True)
    p.add_argument("--test", required=True, help="NHWC float32 .npy (normalized)")
    p.add_argument("--labels", required=True, help="int64 .npy ground-truth")
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test = np.load(args.test).astype(np.float32)  # (M, 224, 224, 3) NHWC normalized
    labels = np.load(args.labels)

    hef = HEF(args.hef)
    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, cfg)[0]
        ng_params = network_group.create_params()

        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        # FLOAT32 in: hand HailoRT our normalized floats; it quantizes on host
        # to match the calibration. FLOAT32 out: dequantized logits.
        in_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        preds = []
        with network_group.activate(ng_params):
            with InferVStreams(network_group, in_params, out_params) as pipeline:
                for i in range(0, len(test), args.batch):
                    batch = test[i : i + args.batch]
                    out = pipeline.infer({in_info.name: batch})
                    logits = np.asarray(out[out_info.name]).reshape(len(batch), -1)
                    preds.append(logits.argmax(1))
        preds = np.concatenate(preds)

    acc = float((preds == labels).mean())
    print(f"Hailo-10H ON-DEVICE top-1: {acc:.4f}  (n={len(labels)})")
    print("FP32 baseline 0.8920 — gap is the true on-silicon quantization cost.")


if __name__ == "__main__":
    main()
