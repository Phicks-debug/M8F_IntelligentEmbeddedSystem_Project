"""
Mushroom Classification Training Flow

Usage:
  python src/main.py run
  python src/main.py run --stage finetune
  python src/main.py run --with batch      # AWS Batch
  python src/main.py run --with kubernetes # Kubernetes
"""

import sys
from pathlib import Path
from typing import Any, Sized, cast

from metaflow import resources, retry  # type: ignore
from metaflow.decorators import step
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_ROOT = REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))


class TrainingFlow(FlowSpec):
    """End-to-end mushroom classification training pipeline."""

    cfg: dict[str, Any]
    teacher_ckpt: Path | None
    finetuned_ckpt: Path | None
    distilled_ckpt: Path | None
    quantized_ckpt: Path | None
    metrics: dict[str, Any]
    exported: dict[str, Any]

    stage = Parameter(
        "stage",
        help="Pipeline stage to run (all, teacher, finetune, distill, quantize, benchmark, report, export)",
        default="all",
    )
    config_name = Parameter(
        "config",
        help="Hydra config name (relative to configs/)",
        default="config",
    )
    quick_test = Parameter(
        "quick-test",
        help="Run a 1-epoch smoke test per stage",
        default=False,
    )

    @step
    def start(self):
        """Initialize Hydra config and validate data."""
        import hydra
        from omegaconf import OmegaConf

        from src.utils import prepare_runtime_paths

        with hydra.initialize_config_dir(
            config_dir=str(CLASSIFICATION_ROOT / "configs"), version_base=None
        ):
            cfg = hydra.compose(config_name=str(self.config_name))

        self.cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        prepare_runtime_paths(self.cfg)

        if self.quick_test:
            self.cfg["finetune"]["epochs"] = 1
            self.cfg["distill"]["epochs"] = 1
            print("*** QUICK TEST MODE (1 epoch per stage) ***")

        print("Configuration:")
        print(OmegaConf.to_yaml(OmegaConf.create(self.cfg)))

        data_dir = Path(self.cfg["data"]["dir"])
        for split in ("train", "val", "test"):
            assert (data_dir / split).exists(), (
                f"Missing data split: {data_dir / split}"
            )
        print(f"Data validated: {data_dir}")

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Validate dataset paths, class count, and split sizes."""
        from src.core import get_dataloaders

        train, val, test, _ = get_dataloaders(
            data_dir=Path(self.cfg["data"]["dir"]),
            image_size=self.cfg["data"]["image_size"],
            num_classes=self.cfg["data"]["num_classes"],
            batch_size=self.cfg["finetune"]["batch_size"],
            workers=self.cfg["data"]["workers"],
            mixup=False,
        )
        from torchvision.datasets import ImageFolder

        train_folder = cast(ImageFolder, train.dataset)
        val_ds = cast(Sized, val.dataset)
        test_ds = cast(Sized, test.dataset)
        self.class_names = train_folder.classes
        print(
            f"Dataset sizes: train={len(train_folder)}, val={len(val_ds)}, test={len(test_ds)}"
        )
        print(f"Classes: {self.class_names}")
        self.next(self.train_teacher)

    @resources(cpu=4, memory=24000, gpu=1)
    @retry(times=2)
    @step
    def train_teacher(self):
        """Fine-tune EfficientNet-B3 teacher."""
        if self.stage not in ("all", "teacher"):
            print(f"Skipping teacher training (stage={self.stage})")
            self.teacher_ckpt = None
            self.next(self.finetune)
            return

        from src.pipeline import run_train_teacher
        from src.tracking import start_run

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification",
            run_name="train_teacher",
            tracking_uri=tracking_uri,
        ):
            self.teacher_ckpt = run_train_teacher(self.cfg)

        print(f"Teacher checkpoint: {self.teacher_ckpt}")
        self.next(self.finetune)

    @resources(cpu=4, memory=16000, gpu=1)
    @retry(times=2)
    @step
    def finetune(self):
        """Fine-tune MobileNetV4 student."""
        if self.stage not in ("all", "finetune"):
            print(f"Skipping finetune (stage={self.stage})")
            self.finetuned_ckpt = None
            self.next(self.distill)
            return

        from src.pipeline import run_finetune
        from src.tracking import start_run

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="finetune", tracking_uri=tracking_uri
        ):
            self.finetuned_ckpt = run_finetune(self.cfg)

        print(f"Fine-tuned checkpoint: {self.finetuned_ckpt}")
        self.next(self.distill)

    @resources(cpu=4, memory=24000, gpu=1)
    @retry(times=2)
    @step
    def distill(self):
        """Distill knowledge from EfficientNet-B3 teacher."""
        if self.stage not in ("all", "distill"):
            print(f"Skipping distill (stage={self.stage})")
            self.distilled_ckpt = None
            self.next(self.quantize)
            return

        ckpt = Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_finetuned.pth"
        _finetuned_ckpt = getattr(self, "finetuned_ckpt", None)
        if _finetuned_ckpt is not None:
            ckpt = Path(_finetuned_ckpt)

        from src.pipeline import run_distill
        from src.tracking import start_run

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="distill", tracking_uri=tracking_uri
        ):
            self.distilled_ckpt = run_distill(self.cfg, ckpt)

        print(f"Distilled checkpoint: {self.distilled_ckpt}")
        self.next(self.quantize)

    @resources(cpu=4, memory=16000)
    @retry(times=2)
    @step
    def quantize(self):
        """Quantize student model to int8."""
        if self.stage not in ("all", "quantize"):
            print(f"Skipping quantize (stage={self.stage})")
            self.quantized_ckpt = None
            self.next(self.benchmark)
            return

        ckpt = Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_distilled.pth"
        _distilled_ckpt = getattr(self, "distilled_ckpt", None)
        if _distilled_ckpt is not None:
            ckpt = Path(_distilled_ckpt)
        elif not ckpt.exists():
            ckpt = (
                Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_finetuned.pth"
            )

        from src.pipeline import run_quantize
        from src.tracking import start_run

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="quantize", tracking_uri=tracking_uri
        ):
            self.quantized_ckpt = run_quantize(self.cfg, ckpt)

        print(f"Quantized checkpoint: {self.quantized_ckpt}")
        self.next(self.benchmark)

    @resources(cpu=4, memory=16000)
    @step
    def benchmark(self):
        """Evaluate accuracy, latency, model size, FLOPs."""
        if self.stage not in ("all", "benchmark"):
            print(f"Skipping benchmark (stage={self.stage})")
            self.metrics = {}
            self.next(self.export)
            return

        ckpt = Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_quantized.pth"
        _quantized_ckpt = getattr(self, "quantized_ckpt", None)
        if _quantized_ckpt is not None:
            ckpt = Path(_quantized_ckpt)
        elif not ckpt.exists():
            ckpt = (
                Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_distilled.pth"
            )
            if not ckpt.exists():
                ckpt = (
                    Path(self.cfg["paths"]["checkpoint_dir"])
                    / "mobilenetv4_finetuned.pth"
                )

        from src.pipeline import run_benchmark
        from src.tracking import start_run

        class_names = getattr(self, "class_names", None)
        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="benchmark", tracking_uri=tracking_uri
        ):
            self.metrics = run_benchmark(self.cfg, ckpt, class_names=class_names)

        print(f"Benchmark metrics: {self.metrics}")
        self.next(self.export)

    @resources(cpu=4, memory=16000)
    @step
    def export(self):
        """Export TorchScript, torch.export, and ONNX artifacts."""
        if self.stage not in ("all", "export"):
            print(f"Skipping export (stage={self.stage})")
            self.exported = {}
            self.next(self.report)
            return

        ckpt = Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_quantized.pth"
        _quantized_ckpt = getattr(self, "quantized_ckpt", None)
        if _quantized_ckpt is not None:
            ckpt = Path(_quantized_ckpt)
        elif not ckpt.exists():
            ckpt = (
                Path(self.cfg["paths"]["checkpoint_dir"]) / "mobilenetv4_distilled.pth"
            )
            if not ckpt.exists():
                ckpt = (
                    Path(self.cfg["paths"]["checkpoint_dir"])
                    / "mobilenetv4_finetuned.pth"
                )

        from src.pipeline import run_export
        from src.tracking import log_model_summary, start_run
        from src.utils import sync_runtime_outputs

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="export", tracking_uri=tracking_uri
        ):
            self.exported = run_export(self.cfg, ckpt)
            summary = {
                "model": self.cfg["finetune"]["arch"],
                "teacher": self.cfg["distill"]["teacher"],
                "num_classes": self.cfg["data"]["num_classes"],
                "image_size": self.cfg["data"]["image_size"],
                "metrics": self.metrics,
                "exported": self.exported,
            }
            log_model_summary(
                summary,
                Path(self.cfg["paths"]["checkpoint_dir"]) / "pipeline_summary.json",
            )
            sync_runtime_outputs(self.cfg, "checkpoint_dir", "export_dir")

        print(f"Exported artifacts: {self.exported}")
        self.next(self.report)

    @resources(cpu=4, memory=16000)
    @step
    def report(self):
        """Generate multi-model benchmark comparison and final analysis."""
        if self.stage not in ("all", "benchmark", "report"):
            print(f"Skipping report (stage={self.stage})")
            self.next(self.end)
            return

        class_names = getattr(self, "class_names", None)
        if class_names is None:
            print("  [WARN] class_names not available; skipping comparison report.")
            self.next(self.end)
            return

        from src.pipeline import run_full_benchmark_comparison
        from src.tracking import start_run

        tracking_uri = self.cfg.get("paths", {}).get("mlflow_tracking_uri")
        with start_run(
            "mushroom_classification", run_name="report", tracking_uri=tracking_uri
        ):
            self.comparison_results = run_full_benchmark_comparison(
                self.cfg, class_names
            )

        self.next(self.end)

    @step
    def end(self):
        """Pipeline complete."""
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        if self.metrics:
            print("Metrics:")
            for k, v in self.metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if self.exported:
            print("Exported:")
            for k, v in self.exported.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    TrainingFlow()
