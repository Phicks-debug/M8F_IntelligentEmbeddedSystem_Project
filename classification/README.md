# Classification

End-to-end classifier for the 8-class mushroom dataset. Trains a large EfficientNet-B3 **teacher**, transfers the knowledge to a small MobileNetV4 **student**, then quantizes and exports the student for edge deployment.

The pipeline is wired with **Metaflow** (orchestration + retry), **Hydra/OmegaConf** (configs), and **MLflow** (experiment tracking).

## Layout

```text
classification/
├── configs/           # Hydra YAML configs (config, stage.*, model.*)
├── src/
│   ├── core.py        # Models, transforms, dataloaders, training loop
│   ├── pipeline.py    # Stage runners: teacher / finetune / distill / quantize / benchmark / export
│   ├── tracking.py    # MLflow helpers
│   ├── utils.py       # Device + checkpoint utilities
│   └── main.py        # Metaflow FlowSpec entrypoint
├── requirements.txt
└── Dockerfile
```

## Requirements

Runtime deps (`requirements.txt`):

| Package | Purpose |
| --- | --- |
| `torch>=2.0`, `torchvision>=0.15` | CUDA/MPS training & v2 transforms |
| `timm>=1.0` | MobileNetV4 student with pretrained weights |
| `metaflow>=2.12` | Pipeline orchestration (steps, retry, resources) |
| `hydra-core>=1.3`, `omegaconf>=2.3` | Layered YAML configuration |
| `mlflow>=2.14` | Experiment tracking + artifact logging |
| `torchao>=0.3` | INT8 weight-only quantization (preferred) |
| `tqdm`, `numpy` | Logging, math |

Optional but recommended: `fvcore` (FLOPs benchmark), `executorch` (edge export).

## Quick Start

1. **Preprocess data** with `data-preprocessing-classification.ipynb` at the project root. It copies images into `data/processed/classification_data/{train,val,test}/<species>/*.jpg` and writes the aligned `metadata.csv`.
2. **Install deps**:

   ```bash
   pip install -r classification/requirements.txt
   ```

3. **Run the full pipeline** locally:

   ```bash
   python classification/src/main.py run
   ```

4. **Smoke test** (1 epoch per stage):

   ```bash
   python classification/src/main.py run --quick-test
   ```

## Usage

Run a single stage instead of the full chain:

```bash
python classification/src/main.py run --stage finetune
python classification/src/main.py run --stage distill
python classification/src/main.py run --stage quantize
python classification/src/main.py run --stage benchmark
python classification/src/main.py run --stage export
```

Available stages: `all`, `teacher`, `finetune`, `distill`, `quantize`, `benchmark`, `export`.

Metaflow compute backends (any of these works):

```bash
python classification/src/main.py run --with kubernetes
python classification/src/main.py run --with argo
python classification/src/main.py run --with batch   # AWS Batch
```

## Configuration

All training knobs are under `configs/` and composed by Hydra at runtime:

- `configs/config.yaml` — defaults (data, finetune, distill, paths, device)
- `configs/stage/{finetune,distill,quantize}.yaml` — per-stage overrides
- `configs/model/{efficientnet_b3,mobilenetv4}.yaml` — model definitions

To change defaults, edit `configs/config.yaml` or add a YAML under `configs/stage/` (e.g., copy `stage/finetune.yaml` and override fields — Hydra picks up files in that group automatically).

## Docker

```bash
docker build -f classification/Dockerfile -t mushroom-cls .
docker run --rm -v $(pwd)/data:/workspace/data mushroom-cls          # full pipeline
docker run --rm -e STAGE=finetune mushroom-cls                       # one stage
```

The container runs as a non-root user and is set up for `@resources(cpu=4, memory=…, gpu=1)` Metaflow steps.

## Outputs

| Path | Content |
| --- | --- |
| `checkpoints/*.pth` | Trained teacher / student / distilled / quantized weights |
| `exported_models/*.pt`, `*.pt2` | TorchScript & `torch.export` artifacts |
| `mlflow.db` | Local SQLite tracking backend |
| `pipeline_summary.json` | Final metrics + export paths |
