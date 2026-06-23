# Classification Pipeline

Training pipeline for the Ray-Ban classification model. It trains a teacher, fine-tunes a MobileNetV4 student, distills, calibrates INT8 quantization, benchmarks, reports, and exports deployment artifacts.

## Data Format

Use an ImageFolder layout:

```text
data/processed/classification_data/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Set the path and class count in `classification/configs/config.yaml`:

```yaml
data:
  dir: data/processed/classification_data
  num_classes: 8
```

## Setup

Use the project `.venv` with `uv`:

```bash
uv pip install -r classification/requirements.txt
```

Run from the repository root.

## Local Run

Full pipeline:

```bash
.venv/bin/python classification/src/main.py run
```

Smoke test:

```bash
.venv/bin/python classification/src/main.py run --quick-test
```

Single stage:

```bash
.venv/bin/python classification/src/main.py run --stage finetune
.venv/bin/python classification/src/main.py run --stage distill
.venv/bin/python classification/src/main.py run --stage quantize
.venv/bin/python classification/src/main.py run --stage benchmark
.venv/bin/python classification/src/main.py run --stage report
.venv/bin/python classification/src/main.py run --stage export
```

Stages: `all`, `teacher`, `finetune`, `distill`, `quantize`, `benchmark`, `report`, `export`.

## Cloud GPU Run

Metaflow can run the same flow on a remote backend:

```bash
.venv/bin/python classification/src/main.py run --with batch
.venv/bin/python classification/src/main.py run --with kubernetes
.venv/bin/python classification/src/main.py run --with argo
```

The GPU steps already request GPU resources in `src/main.py`.

## Local, S3, or R2 Storage

`data.dir`, `paths.checkpoint_dir`, and `paths.export_dir` can be local paths or object-store URIs. Remote data is staged to `paths.local_cache_dir`; outputs are synced back after stages.

S3 example:

```yaml
data:
  dir: s3://my-bucket/rayban/classification_data

paths:
  checkpoint_dir: s3://my-bucket/rayban/checkpoints
  export_dir: s3://my-bucket/rayban/exported_models
  local_cache_dir: .cache/classification
```

Cloudflare R2 example:

```yaml
data:
  dir: r2://my-bucket/rayban/classification_data

paths:
  checkpoint_dir: r2://my-bucket/rayban/checkpoints
  export_dir: r2://my-bucket/rayban/exported_models
  storage_options:
    client_kwargs:
      endpoint_url: https://<account-id>.r2.cloudflarestorage.com
```

Use normal S3/R2 credentials in the environment.

## Outputs

```text
checkpoints/
  teacher_for_distill.pth
  mobilenetv4_finetuned.pth
  mobilenetv4_distilled.pth
  mobilenetv4_quantized.pth
  *_history.json
  *_training_curves.png
  *_confusion_matrix.png
  *_confidence_analysis.png
  benchmark_comparison.json
  benchmark_comparison.png

exported_models/
  mobilenetv4.pt
  mobilenetv4.pt2
  mobilenetv4.onnx
  mobilenetv4_int8.onnx
```

Use `exported_models/mobilenetv4_int8.onnx` as the Ray-Ban NPU deployment model. It is a calibrated QDQ INT8 ONNX model. `mobilenetv4.onnx` is only the FP32 intermediate used to build the INT8 export.

The `quantize` stage also saves a calibrated PyTorch INT8 checkpoint at `checkpoints/mobilenetv4_quantized.pth`.

Reruns use fixed output filenames. Training checkpoints update when a new best validation score is saved; quantize and export outputs overwrite their previous files.

## Checks

```bash
.venv/bin/python -m ruff check classification/src
uvx pyright --pythonpath .venv/bin/python
.venv/bin/python classification/src/main.py check
```
