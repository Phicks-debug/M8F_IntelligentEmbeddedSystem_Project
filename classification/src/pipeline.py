"""
Pipeline stage runners for mushroom classification.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.core import (
    _live_mem_snapshot,
    benchmark_latency,
    build_scheduler,
    class_weights_from_counts,
    count_flops,
    count_params,
    evaluate,
    get_dataloaders,
    load_checkpoint,
    make_student,
    make_teacher,
    model_size_mb,
    quantization_snr_db,
    save_checkpoint,
    save_resume_state,
    train_one_epoch,
    try_load_resume_state,
)
from src.reporting import (
    evaluate_with_predictions,
    plot_benchmark_comparison,
    plot_confidence_analysis,
    plot_confusion_matrix,
    plot_training_curves,
    print_benchmark_table,
    save_history,
)
from src.tracking import log_artifact, log_metrics, log_params
from src.utils import get_device, sync_runtime_outputs

DEFAULT_QUANTIZE_BACKEND = "torchao_int8_static_activation_int8_weight"


def _last_ckpt_path(best_path: Path) -> Path:
    """Companion resume-state path for a given best ckpt.

    `mobilenetv4_finetuned.pth` -> `mobilenetv4_finetuned.last.pth`.
    Kept as a separate file so the best ckpt's filename (which downstream
    stages like `run_distill` / `run_quantize` look up) is unchanged.
    """
    return best_path.with_name(best_path.stem + ".last" + best_path.suffix)


def _try_resume(
    best_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> tuple[int, float, int]:
    """Load resume state if any. Returns ``(start_epoch, best_acc, stall)``.

    A fresh start returns ``(0, 0.0, 0)``. On a successful resume, prints
    a single line so the user can see where the loop will pick up.
    """
    last_path = _last_ckpt_path(best_path)
    rs = try_load_resume_state(last_path, model, optimizer, scheduler)
    if rs is None:
        return 0, 0.0, 0
    print(
        f"  Resumed from epoch {rs.epoch + 1} "
        f"(best_acc={rs.best_acc:.4f}, stall={rs.stall})"
    )
    return rs.epoch, rs.best_acc, rs.stall


def _run_probe(
    cfg: dict,
    cfg_key: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    crit: nn.Module,
    eval_crit: nn.Module,
    freeze_predicate: Callable[[str], bool],
    banner: str,
    log_metric_key: str,
    probe_epochs: int,
    device: str,
    mixed_precision: bool,
    lr_override: Optional[float] = None,
) -> tuple[float, dict]:
    """Head-only probe training shared by teacher and student stages.

    Freezes parameters that do not satisfy ``freeze_predicate``, builds an
    AdamW optimiser on the trainable subset, and runs a short probe.

    Returns ``(best_probe_val_acc, history_dict)``.
    """
    for name, param in model.named_parameters():
        if not freeze_predicate(name):
            param.requires_grad_(False)

    opt_probe = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_override if lr_override is not None else cfg[cfg_key]["lr"],
        weight_decay=cfg[cfg_key]["weight_decay"],
    )

    print(f"\n[{banner}]")
    history = {"epochs": [], "val_acc": []}
    best_probe = 0.0
    for epoch in range(1, probe_epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            crit,
            opt_probe,
            epoch,
            device,
            mixed_precision=mixed_precision,
        )
        _, val_acc = evaluate(
            model,
            val_loader,
            eval_crit,
            device,
            mixed_precision=mixed_precision,
        )
        print(f"  Probe epoch {epoch}: val_acc={val_acc:.4f}")
        history["epochs"].append(epoch)
        history["val_acc"].append(val_acc)
        if val_acc > best_probe:
            best_probe = val_acc
    print(f"  Best probe val_acc: {best_probe:.4f}")
    log_metrics({log_metric_key: best_probe})
    return best_probe, history


def _run_resumable_epoch_loop(
    cfg: dict,
    cfg_key: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    crit: nn.Module,
    eval_crit: nn.Module,
    ckpt_path: Path,
    mixed_precision: bool,
    log_metric_key: str,
    *,
    teacher: Optional[nn.Module] = None,
    distill_cfg: Optional[dict] = None,
    patience_override: Optional[int] = None,
    lr_override: Optional[float] = None,
) -> tuple[float, float, dict]:
    """Resume-aware training loop shared by teacher, student, and distillation.

    Builds optimizer + scheduler, restores from ``*.last.pth`` if present,
    runs the train/validate/early-stop loop, then loads the best checkpoint
    and evaluates on the test split.

    Returns ``(best_val_acc, test_acc, history_dict)``.
    """
    device = model_device(model)
    opt = optim.AdamW(
        model.parameters(),
        lr=lr_override if lr_override is not None else cfg[cfg_key]["lr"],
        weight_decay=cfg[cfg_key]["weight_decay"],
    )
    sched = build_scheduler(opt, cfg[cfg_key]["warmup"], cfg[cfg_key]["epochs"])
    last_ckpt_path = _last_ckpt_path(ckpt_path)
    start_epoch, best_acc, stall = _try_resume(ckpt_path, model, opt, sched)
    patience = (
        patience_override if patience_override is not None else cfg.get("patience", 10)
    )
    target_epochs = cfg[cfg_key]["epochs"]

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if start_epoch < target_epochs:
        for epoch in range(start_epoch + 1, target_epochs + 1):
            train_kwargs: dict = {}
            if teacher is not None and distill_cfg is not None:
                train_kwargs = {"teacher": teacher, "distill_cfg": distill_cfg}
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                crit,
                opt,
                epoch,
                device,
                mixed_precision=mixed_precision,
                **train_kwargs,
            )
            snap = _live_mem_snapshot(device)
            if snap:
                print(f"  {snap}")
            if sched is not None:
                sched.step()
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                eval_crit,
                device,
                mixed_precision=mixed_precision,
            )
            print(f"  Epoch {epoch}: val_acc={val_acc:.4f}")
            log_metrics({log_metric_key: val_acc}, step=epoch)

            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, ckpt_path, epoch=epoch, val_acc=val_acc)
                stall = 0
                save_resume_state(
                    last_ckpt_path, model, opt, sched, epoch, best_acc, stall
                )
            else:
                stall += 1
                save_resume_state(
                    last_ckpt_path, model, opt, sched, epoch, best_acc, stall
                )
                if stall >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break
    else:
        print(
            f"  Resume state already completed {start_epoch} of "
            f"{target_epochs} epochs; skipping training loop."
        )

    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model, device)
    _, test_acc = evaluate(
        model,
        test_loader,
        eval_crit,
        device,
        mixed_precision=mixed_precision,
    )
    return best_acc, test_acc, history


def model_device(model: nn.Module) -> str:
    """Return ``model.device.type`` as a string — the form used by the rest
    of the pipeline (`get_device(cfg.get("device", "auto"))` returns one).
    """
    return next(model.parameters()).device.type


def run_train_teacher(cfg: dict) -> Path:
    """Fine-tune EfficientNet-B3 teacher. Returns path to best checkpoint."""
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))
    mp = cfg["data"].get("mixed_precision", False)

    train_probe, val_probe, _, class_counts = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    train_full, val_full, test_full, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=True,
    )

    model = make_teacher(cfg["data"]["num_classes"]).to(device)
    print(f"Teacher params: {count_params(model):,}")

    log_params(
        {
            "teacher.arch": "efficientnet_b3",
            "teacher.epochs": cfg["finetune"]["epochs"],
            "teacher.lr": cfg["finetune"]["lr"],
            "teacher.batch_size": cfg["finetune"]["batch_size"],
        }
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "teacher_for_distill.pth"

    crit = nn.CrossEntropyLoss(
        weight=class_weights_from_counts(class_counts, device),
        label_smoothing=cfg["finetune"].get("smoothing", 0.0),
    )
    eval_crit = nn.CrossEntropyLoss()

    _, probe_history = _run_probe(
        cfg=cfg,
        cfg_key="finetune",
        model=model,
        train_loader=train_probe,
        val_loader=val_probe,
        crit=crit,
        eval_crit=eval_crit,
        freeze_predicate=lambda name: "classifier" in name,
        banner="TEACHER PROBE",
        log_metric_key="teacher_probe_val_acc",
        probe_epochs=cfg["finetune"].get("probe_epochs", 5),
        device=device,
        mixed_precision=mp,
    )

    print("\n[TEACHER FULL] Fine-tuning all parameters")
    for param in model.parameters():
        param.requires_grad_(True)

    best_acc, test_acc, full_history = _run_resumable_epoch_loop(
        cfg=cfg,
        cfg_key="finetune",
        model=model,
        train_loader=train_full,
        val_loader=val_full,
        test_loader=test_full,
        crit=crit,
        eval_crit=eval_crit,
        ckpt_path=ckpt_path,
        mixed_precision=mp,
        log_metric_key="teacher_val_acc",
    )
    print(f"  Teacher test_acc: {test_acc:.4f}")
    log_metrics({"teacher_test_acc": test_acc, "teacher_best_val_acc": best_acc})
    log_artifact(ckpt_path)

    history = {"probe": probe_history, "full": full_history, "test_acc": test_acc}
    history_path = ckpt_dir / "teacher_history.json"
    save_history(history, history_path)
    plot_training_curves(
        history,
        ckpt_dir / "teacher_training_curves.png",
        title="Teacher (EfficientNet-B3) Training Progress",
    )
    sync_runtime_outputs(cfg, "checkpoint_dir")

    return ckpt_path


def run_finetune(cfg: dict) -> Path:
    """Fine-tune MobileNetV4 student. Returns path to best checkpoint."""
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))
    mp = cfg["data"].get("mixed_precision", False)

    light_aug = cfg["finetune"].get("light_augmentation", True)
    student_lr = cfg["finetune"].get("student_lr", cfg["finetune"]["lr"])
    student_patience = cfg["finetune"].get("student_patience", cfg.get("patience", 10))
    cfg["finetune"]["lr"] = student_lr

    train_probe, val_probe, _, class_counts = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    train_full, val_full, test_full, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=not light_aug,
        light_augmentation=light_aug,
    )

    model = make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"]).to(device)
    print(f"Student params: {count_params(model):,}")
    print(
        f"  student_lr={student_lr}, light_aug={light_aug}, patience={student_patience}"
    )

    log_params(
        {
            "finetune.arch": cfg["finetune"]["arch"],
            "finetune.epochs": cfg["finetune"]["epochs"],
            "finetune.lr": student_lr,
            "finetune.batch_size": cfg["finetune"]["batch_size"],
            "finetune.light_augmentation": light_aug,
            "finetune.student_patience": student_patience,
        }
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "mobilenetv4_finetuned.pth"

    class_weights = class_weights_from_counts(class_counts, device)
    crit = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["finetune"].get("smoothing", 0.0),
    )
    eval_crit = nn.CrossEntropyLoss()

    _, probe_history = _run_probe(
        cfg=cfg,
        cfg_key="finetune",
        model=model,
        train_loader=train_probe,
        val_loader=val_probe,
        crit=crit,
        eval_crit=eval_crit,
        freeze_predicate=lambda name: "head" in name or "classifier" in name,
        banner="PROBE",
        log_metric_key="probe_val_acc",
        probe_epochs=cfg["finetune"].get("probe_epochs", 3),
        device=device,
        mixed_precision=mp,
        lr_override=student_lr,
    )

    print("\n[FULL] Fine-tuning all parameters")
    for param in model.parameters():
        param.requires_grad_(True)

    best_acc, test_acc, full_history = _run_resumable_epoch_loop(
        cfg=cfg,
        cfg_key="finetune",
        model=model,
        train_loader=train_full,
        val_loader=val_full,
        test_loader=test_full,
        crit=crit,
        eval_crit=eval_crit,
        ckpt_path=ckpt_path,
        mixed_precision=mp,
        log_metric_key="finetune_val_acc",
        patience_override=student_patience,
        lr_override=student_lr,
    )
    print(f"  Student test_acc: {test_acc:.4f}")
    log_metrics({"finetune_test_acc": test_acc, "finetune_best_val_acc": best_acc})
    log_artifact(ckpt_path)

    history = {"probe": probe_history, "full": full_history, "test_acc": test_acc}
    history_path = ckpt_dir / "student_history.json"
    save_history(history, history_path)
    plot_training_curves(
        history,
        ckpt_dir / "student_training_curves.png",
        title="Student (MobileNetV4) Training Progress",
    )
    sync_runtime_outputs(cfg, "checkpoint_dir")

    return ckpt_path


def run_distill(cfg: dict, student_ckpt: Path) -> Path:
    """Distill teacher knowledge into student. Returns path to best checkpoint."""
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))
    mp = cfg["data"].get("mixed_precision", False)

    teacher = make_teacher(cfg["data"]["num_classes"]).to(device)
    teacher_ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "teacher_for_distill.pth"
    if teacher_ckpt_path.exists():
        load_checkpoint(teacher_ckpt_path, teacher, device)
        print(f"Loaded trained teacher from {teacher_ckpt_path}")
    else:
        print("Warning: no trained teacher found; using pretrained ImageNet teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"]).to(
        device
    )
    if student_ckpt.exists():
        load_checkpoint(student_ckpt, student, device)
        print(f"Loaded fine-tuned student from {student_ckpt}")

    train, val, test, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["distill"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )

    log_params(
        {
            "distill.teacher": cfg["distill"]["teacher"],
            "distill.student": cfg["distill"]["student"],
            "distill.epochs": cfg["distill"]["epochs"],
            "distill.lr": cfg["distill"]["lr"],
            "distill.T": cfg["distill"]["temperature"],
            "distill.alpha": cfg["distill"]["alpha"],
        }
    )

    crit = nn.CrossEntropyLoss(label_smoothing=cfg["finetune"].get("smoothing", 0.0))
    eval_crit = nn.CrossEntropyLoss()

    distill_cfg = {"T": cfg["distill"]["temperature"], "alpha": cfg["distill"]["alpha"]}

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "mobilenetv4_distilled.pth"

    best_acc, test_acc, history = _run_resumable_epoch_loop(
        cfg=cfg,
        cfg_key="distill",
        model=student,
        train_loader=train,
        val_loader=val,
        test_loader=test,
        crit=crit,
        eval_crit=eval_crit,
        ckpt_path=ckpt_path,
        mixed_precision=mp,
        log_metric_key="distill_val_acc",
        teacher=teacher,
        distill_cfg=distill_cfg,
    )
    print(f"  Distilled test_acc: {test_acc:.4f}")
    log_metrics({"distill_test_acc": test_acc, "distill_best_val_acc": best_acc})
    log_artifact(ckpt_path)

    # Save training history & plot
    history["test_acc"] = test_acc
    history_path = ckpt_dir / "distill_history.json"
    save_history(history, history_path)
    plot_training_curves(
        history,
        ckpt_dir / "distill_training_curves.png",
        title="Distilled Student Training Progress",
    )
    sync_runtime_outputs(cfg, "checkpoint_dir")

    return ckpt_path


def _quantize_backend(cfg: dict, checkpoint_path: Optional[Path] = None) -> str:
    if checkpoint_path is not None and "quantized" in checkpoint_path.stem:
        if checkpoint_path.exists():
            try:
                state = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                backend = state.get("backend")
                if isinstance(backend, str):
                    return backend
            except Exception as exc:
                print(
                    f"  [WARN] Could not read quantization backend from "
                    f"{checkpoint_path}: {exc}"
                )
        return cfg.get("quantize", {}).get("backend", DEFAULT_QUANTIZE_BACKEND)
    return ""


def _checkpoint_quantization_calibration(checkpoint_path: Path) -> Optional[dict]:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(
            f"  [WARN] Could not read calibration metadata from {checkpoint_path}: {exc}"
        )
        return None
    calibration = state.get("calibration")
    return calibration if isinstance(calibration, dict) else None


def _calibrate_linear_activation_ranges(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    max_batches: int,
    mixed_precision: bool = False,
) -> dict:
    """Collect per-linear input ranges used by static activation INT8."""
    from src.core import _amp_ctx

    stats: dict[str, dict[str, float | int]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs, _output) -> None:
            if not inputs:
                return
            act = inputs[0].detach().float()
            min_val = float(act.min().item())
            max_val = float(act.max().item())
            sample_count = int(act.numel())
            current = stats.setdefault(
                name,
                {
                    "min": min_val,
                    "max": max_val,
                    "ndim": int(act.ndim),
                    "samples": 0,
                },
            )
            current["min"] = min(float(current["min"]), min_val)
            current["max"] = max(float(current["max"]), max_val)
            current["ndim"] = max(int(current["ndim"]), int(act.ndim))
            current["samples"] = int(current["samples"]) + sample_count

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not hooks:
        raise RuntimeError(
            "Static quantization calibration found no nn.Linear modules."
        )

    model.eval()
    seen_batches = 0
    try:
        with torch.inference_mode():
            for images, _ in loader:
                if seen_batches >= max_batches:
                    break
                images = images.to(device)
                with _amp_ctx(device, mixed_precision):
                    model(images)
                seen_batches += 1
    finally:
        for hook in hooks:
            hook.remove()

    if seen_batches == 0:
        raise ValueError("Calibration loader produced 0 batches.")

    modules = {}
    for name, values in stats.items():
        min_val = float(values["min"])
        max_val = float(values["max"])
        max_abs = max(abs(min_val), abs(max_val))
        scale = max(max_abs / 127.0, 1e-8)
        modules[name] = {
            "min": min_val,
            "max": max_val,
            "scale": scale,
            "zero_point": None,
            "ndim": int(values["ndim"]),
            "samples": int(values["samples"]),
        }

    return {
        "method": "linear_input_minmax_symmetric",
        "batches": seen_batches,
        "modules": modules,
    }


def _apply_quantization_backend(
    model: nn.Module,
    backend: str,
    calibration: Optional[dict] = None,
) -> tuple[nn.Module, dict]:
    if backend == "torchao_int8_dynamic_activation_int8_weight":
        from torchao import quantization as torchao_quantization  # type: ignore

        quantize_fn = getattr(torchao_quantization, "quantize_")
        quantizer_fn = getattr(
            torchao_quantization, "int8_dynamic_activation_int8_weight"
        )
        quantize_fn(model, quantizer_fn())
        return model, {"method": "dynamic_activation_no_calibration"}
    if backend == "torchao_int8_static_activation_int8_weight":
        if calibration is None:
            raise ValueError(
                "Static INT8 quantization requires calibration metadata. "
                "Run calibration before quantize_ or load a calibrated checkpoint."
            )
        from torchao import quantization as torchao_quantization  # type: ignore
        from torchao.quantization.granularity import PerRow, PerTensor

        quantize_fn = getattr(torchao_quantization, "quantize_")
        config_cls = getattr(
            torchao_quantization, "Int8StaticActivationInt8WeightConfig"
        )
        modules = calibration.get("modules", {})
        quantized_modules = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or name not in modules:
                continue
            scale_shape = (1,) * int(modules[name].get("ndim", 2))
            scale = torch.full(
                scale_shape, float(modules[name]["scale"]), dtype=torch.float32
            )
            config = config_cls(
                act_quant_scale=scale,
                act_quant_zero_point=None,
                granularity=(PerTensor(), PerRow()),
            )
            quantize_fn(
                model,
                config,
                filter_fn=lambda _module, fqn, target=name: fqn == target,
            )
            quantized_modules += 1
        if quantized_modules == 0:
            raise RuntimeError(
                "Static INT8 quantization did not quantize any modules; "
                "check calibration module names."
            )
        metadata = dict(calibration)
        metadata["quantized_modules"] = quantized_modules
        return model, metadata
    if backend == "torchao_int8_weight_only":
        from torchao import quantization as torchao_quantization  # type: ignore

        quantize_fn = getattr(torchao_quantization, "quantize_")
        quantizer_fn = getattr(torchao_quantization, "int8_weight_only")
        quantize_fn(model, quantizer_fn())
        return model, {"method": "weight_only_no_activation_calibration"}
    if backend == "pytorch_dynamic_qint8":
        return (
            torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            ),
            {"method": "pytorch_dynamic_no_calibration"},
        )
    raise ValueError(
        f"Unknown quantize backend: {backend!r}. Use one of "
        "'torchao_int8_static_activation_int8_weight', "
        "'torchao_int8_dynamic_activation_int8_weight', "
        "'torchao_int8_weight_only', or 'pytorch_dynamic_qint8'."
    )


def _make_student_for_checkpoint(
    cfg: dict, checkpoint_path: Path, requested_device: str
) -> tuple[nn.Module, str]:
    backend = _quantize_backend(cfg, checkpoint_path)
    device = "cpu" if backend else requested_device
    model = make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"]).to(device)
    if backend:
        calibration = _checkpoint_quantization_calibration(checkpoint_path)
        model, _ = _apply_quantization_backend(model.eval(), backend, calibration)
        model = model.to(device)
    model.eval()
    load_checkpoint(checkpoint_path, model, device)
    return model, device


def _resolve_fp32_student_checkpoint(cfg: dict, checkpoint_path: Path) -> Path:
    """Return a non-quantized student checkpoint for export to standard ONNX."""
    if "quantized" not in checkpoint_path.stem:
        return checkpoint_path

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    for candidate in (
        ckpt_dir / "mobilenetv4_distilled.pth",
        ckpt_dir / "mobilenetv4_finetuned.pth",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "ONNX export needs a FP32 student checkpoint. Expected "
        f"{ckpt_dir / 'mobilenetv4_distilled.pth'} or "
        f"{ckpt_dir / 'mobilenetv4_finetuned.pth'}."
    )


def _export_calibrated_int8_onnx(
    cfg: dict,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
) -> bool:
    """Create a calibrated QDQ INT8 ONNX model for NPU deployment."""
    try:
        ort_quant = __import__(
            "onnxruntime.quantization",
            fromlist=[
                "CalibrationDataReader",
                "QuantFormat",
                "QuantType",
                "quantize_static",
            ],
        )
    except ImportError as exc:
        print(
            "INT8 ONNX export skipped: onnxruntime is not installed. "
            "Install classification/requirements.txt and rerun export."
        )
        print(f"  Import error: {exc}")
        return False

    cal_cfg = cfg.get("quantize", {}).get("calibration", {})
    cal_batch_size = cal_cfg.get("batch_size", 8)
    cal_batches = cal_cfg.get("batches", 32)
    cal_split = cal_cfg.get("split", "val")

    cal_train, cal_val, cal_test, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cal_batch_size,
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    calibration_loader = {
        "train": cal_train,
        "val": cal_val,
        "test": cal_test,
    }.get(cal_split)
    if calibration_loader is None:
        raise ValueError(
            f"Unknown quantize.calibration.split={cal_split!r}; "
            "use 'train', 'val', or 'test'."
        )

    class ImageCalibrationReader(ort_quant.CalibrationDataReader):
        def __init__(self):
            self._iterator = iter(calibration_loader)
            self._seen = 0

        def get_next(self):
            if self._seen >= cal_batches:
                return None
            try:
                images, _ = next(self._iterator)
            except StopIteration:
                return None
            self._seen += 1
            return {"input": images.cpu().numpy()}

    print(
        f"Calibrating INT8 ONNX on {cal_split} split "
        f"({cal_batches} batches, batch_size={cal_batch_size})"
    )
    ort_quant.quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=ImageCalibrationReader(),
        quant_format=ort_quant.QuantFormat.QDQ,
        activation_type=ort_quant.QuantType.QUInt8,
        weight_type=ort_quant.QuantType.QInt8,
        per_channel=True,
    )
    return True


def _snr_verdict(snr_db: float) -> str:
    if snr_db < 20.0:
        return "WARN: SNR below 20 dB"
    if snr_db >= 40.0:
        return "PASS: SNR >= 40 dB (excellent)"
    return "PASS: SNR in 20-40 dB band"


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _cross_entropy_np(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax_np(logits)
    picked = probs[np.arange(labels.shape[0]), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def _estimate_onnx_gops(path: Path) -> float:
    """Estimate ONNX Conv/Gemm/MatMul operations in GOPs.

    This is an approximation for deployment reporting. TOPS is hardware and
    runtime dependent, so the pipeline reports effective TOPS separately from
    ONNX Runtime latency.
    """
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        return 0.0

    try:
        model = shape_inference.infer_shapes(onnx.load(path))
    except Exception:
        model = onnx.load(path)

    shapes: dict[str, list[int]] = {}
    for value in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value else 1)
        shapes[value.name] = dims

    initializer_shapes = {
        init.name: list(init.dims) for init in model.graph.initializer
    }
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "DequantizeLinear" and node.input
    }

    def initializer_shape(name: str) -> list[int]:
        return initializer_shapes.get(
            name, initializer_shapes.get(aliases.get(name, "")) or []
        )

    ops = 0
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            out_shape = shapes.get(node.output[0], [])
            weight_shape = initializer_shape(node.input[1])
            if len(out_shape) >= 4 and len(weight_shape) >= 4:
                batch, out_ch, out_h, out_w = out_shape[:4]
                _, in_ch_per_group, k_h, k_w = weight_shape[:4]
                ops += batch * out_ch * out_h * out_w * in_ch_per_group * k_h * k_w
        elif node.op_type == "Gemm" and len(node.input) >= 2:
            out_shape = shapes.get(node.output[0], [])
            weight_shape = initializer_shape(node.input[1])
            if len(out_shape) >= 2 and len(weight_shape) >= 2:
                batch = out_shape[0]
                out_features = out_shape[-1]
                trans_b = 0
                for attr in node.attribute:
                    if attr.name == "transB":
                        trans_b = int(attr.i)
                        break
                in_features = weight_shape[1] if trans_b else weight_shape[0]
                ops += batch * out_features * in_features
        elif node.op_type == "MatMul" and len(node.input) >= 2:
            a_shape = shapes.get(node.input[0], initializer_shape(node.input[0]))
            b_shape = shapes.get(node.input[1], initializer_shape(node.input[1]))
            out_shape = shapes.get(node.output[0], [])
            if len(a_shape) >= 2 and len(b_shape) >= 2 and len(out_shape) >= 2:
                batch = int(np.prod(out_shape[:-1])) if len(out_shape) > 1 else 1
                ops += batch * b_shape[-2] * b_shape[-1]

    return ops / 1e9


def _run_onnx_session(session, images: torch.Tensor) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: images.cpu().numpy().astype(np.float32)})
    return outputs[0]


def _benchmark_onnx_model(
    path: Path,
    loader: DataLoader,
    *,
    max_latency_runs: int = 100,
) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total_loss, total_correct, n = 0.0, 0, 0
    for images, labels in loader:
        logits = _run_onnx_session(session, images)
        labels_np = labels.numpy()
        total_loss += _cross_entropy_np(logits, labels_np) * labels_np.shape[0]
        total_correct += int((logits.argmax(axis=1) == labels_np).sum())
        n += labels_np.shape[0]

    first_batch = next(iter(loader))[0][:1]
    for _ in range(10):
        _run_onnx_session(session, first_batch)
    start = time.perf_counter()
    for _ in range(max_latency_runs):
        _run_onnx_session(session, first_batch)
    latency_ms = ((time.perf_counter() - start) / max_latency_runs) * 1000.0

    size_mb = model_size_mb(path)
    gops = _estimate_onnx_gops(path)
    effective_tops = (gops / latency_ms) if latency_ms > 0 else 0.0
    return {
        "test_loss": total_loss / max(n, 1),
        "test_acc": total_correct / max(n, 1),
        "latency_ms": latency_ms,
        "size_mb": size_mb,
        "int8_gops": gops,
        "effective_tops": effective_tops,
    }


def _onnx_snr_db(
    fp32_path: Path,
    int8_path: Path,
    loader: DataLoader,
    *,
    max_batches: int,
) -> dict[str, float]:
    import onnxruntime as ort

    fp32_session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )
    sig_powers: list[float] = []
    noise_powers: list[float] = []
    sig_softmax: list[float] = []
    noise_softmax: list[float] = []
    seen = 0

    for images, _ in loader:
        if seen >= max_batches:
            break
        fp32_logits = _run_onnx_session(fp32_session, images)
        int8_logits = _run_onnx_session(int8_session, images)
        diff = fp32_logits - int8_logits
        sig_powers.append(float(np.mean(fp32_logits**2)))
        noise_powers.append(float(np.mean(diff**2)))
        fp32_probs = _softmax_np(fp32_logits)
        int8_probs = _softmax_np(int8_logits)
        sig_softmax.append(float(np.mean(fp32_probs**2)))
        noise_softmax.append(float(np.mean((fp32_probs - int8_probs) ** 2)))
        seen += 1

    if seen == 0:
        raise ValueError("ONNX SNR loader produced 0 batches")

    sig = sum(sig_powers) / seen
    noise = sum(noise_powers) / seen
    sig_s = sum(sig_softmax) / seen
    noise_s = sum(noise_softmax) / seen
    return {
        "snr_db_logit": 10.0 * math.log10(max(sig / max(noise, 1e-12), 1e-12)),
        "snr_db_softmax": 10.0 * math.log10(max(sig_s / max(noise_s, 1e-12), 1e-12)),
        "signal_power_logit": sig,
        "noise_power_logit": noise,
    }


def validate_onnx_exports(cfg: dict) -> list[dict]:
    """Validate exported FP32 and INT8 ONNX models on the test split."""
    export_dir = Path(cfg["paths"]["export_dir"])
    fp32_path = export_dir / "mobilenetv4.onnx"
    int8_path = export_dir / "mobilenetv4_int8.onnx"
    if not fp32_path.exists() or not int8_path.exists():
        missing = [str(p) for p in (fp32_path, int8_path) if not p.exists()]
        print(f"  [WARN] ONNX validation skipped; missing: {missing}")
        return []

    _, _, test_loader, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["benchmark"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    latency_runs = cfg.get("benchmark", {}).get("onnx_runs", 100)
    fp32_metrics = _benchmark_onnx_model(
        fp32_path, test_loader, max_latency_runs=latency_runs
    )
    int8_metrics = _benchmark_onnx_model(
        int8_path, test_loader, max_latency_runs=latency_runs
    )
    snr_cfg = cfg.get("quantize", {}).get("snr", {})
    snr = _onnx_snr_db(
        fp32_path,
        int8_path,
        test_loader,
        max_batches=snr_cfg.get("batches", 16),
    )
    verdict = _snr_verdict(snr["snr_db_logit"])
    quantize_cfg = cfg.get("quantize", {})
    snr_cfg = quantize_cfg.get("snr", {})
    min_snr_db = snr_cfg.get("min_db", 20.0)
    fail_below_min_snr = snr_cfg.get("fail_below_min", False)
    max_accuracy_drop = quantize_cfg.get("max_onnx_accuracy_drop", 0.05)
    accuracy_drop = fp32_metrics["test_acc"] - int8_metrics["test_acc"]
    print("ONNX validation:")
    print(
        f"  FP32 ONNX: acc={fp32_metrics['test_acc']:.4f} "
        f"loss={fp32_metrics['test_loss']:.4f} "
        f"latency={fp32_metrics['latency_ms']:.3f}ms "
        f"size={fp32_metrics['size_mb']:.2f}MB"
    )
    print(
        f"  INT8 ONNX: acc={int8_metrics['test_acc']:.4f} "
        f"loss={int8_metrics['test_loss']:.4f} "
        f"latency={int8_metrics['latency_ms']:.3f}ms "
        f"size={int8_metrics['size_mb']:.2f}MB "
        f"int8_gops={int8_metrics['int8_gops']:.3f} "
        f"effective_tops={int8_metrics['effective_tops']:.6f}"
    )
    print(
        f"  ONNX INT8 SNR(logit)={snr['snr_db_logit']:.2f} dB [{verdict}] "
        f"SNR(softmax)={snr['snr_db_softmax']:.2f} dB"
    )

    results = [
        {
            "name": "ONNX FP32",
            "params_m": 0.0,
            "test_acc": fp32_metrics["test_acc"],
            "test_loss": fp32_metrics["test_loss"],
            "size_mb": fp32_metrics["size_mb"],
            "gflops": fp32_metrics["int8_gops"],
            "latency_ms": fp32_metrics["latency_ms"],
            "effective_tops": 0.0,
        },
        {
            "name": "ONNX INT8 QDQ",
            "params_m": 0.0,
            "test_acc": int8_metrics["test_acc"],
            "test_loss": int8_metrics["test_loss"],
            "size_mb": int8_metrics["size_mb"],
            "gflops": 0.0,
            "int8_gops": int8_metrics["int8_gops"],
            "latency_ms": int8_metrics["latency_ms"],
            "effective_tops": int8_metrics["effective_tops"],
            "snr_db_logit": snr["snr_db_logit"],
            "snr_db_softmax": snr["snr_db_softmax"],
            "snr_verdict": verdict,
            "snr_min_db": min_snr_db,
            "snr_pass": snr["snr_db_logit"] >= min_snr_db,
            "snr_fail_below_min": fail_below_min_snr,
            "accuracy_drop": accuracy_drop,
            "max_accuracy_drop": max_accuracy_drop,
            "accuracy_pass": accuracy_drop <= max_accuracy_drop,
        },
    ]
    summary_path = export_dir / "onnx_validation.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ONNX validation saved: {summary_path}")
    log_artifact(summary_path)
    log_metrics(
        {
            "onnx_int8_test_acc": int8_metrics["test_acc"],
            "onnx_int8_latency_ms": int8_metrics["latency_ms"],
            "onnx_int8_effective_tops": int8_metrics["effective_tops"],
            "onnx_int8_snr_db_logit": snr["snr_db_logit"],
            "onnx_int8_snr_db_softmax": snr["snr_db_softmax"],
        }
    )
    return results


def _write_markdown_report(cfg: dict, results: list[dict]) -> Path:
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    export_dir = Path(cfg["paths"]["export_dir"])
    report_path = ckpt_dir / "classification_report.md"

    def rel(path: Path) -> str:
        return os.path.relpath(path, start=ckpt_dir)

    lines = [
        "# Classification Report",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Loss | Size MB | GFLOPs | INT8 GOPs | Lat ms | TOPS | SNR dB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {name} | {acc:.4f} | {loss:.4f} | {size:.2f} | {gflops:.3f} | "
            "{int8_gops:.3f} | {latency:.3f} | {tops:.6f} | {snr:.2f} |".format(
                name=row.get("name", ""),
                acc=row.get("test_acc", 0.0),
                loss=row.get("test_loss", 0.0),
                size=row.get("size_mb", 0.0),
                gflops=row.get("gflops", 0.0),
                int8_gops=row.get("int8_gops", 0.0),
                latency=row.get("latency_ms", 0.0),
                tops=row.get("effective_tops", 0.0),
                snr=row.get("snr_db_logit", 0.0),
            )
        )

    diagrams = [
        (
            "Benchmark comparison",
            ckpt_dir / "benchmark_comparison.png",
        ),
        (
            "Quantized confusion matrix",
            ckpt_dir / "mobilenetv4_quantized_confusion_matrix.png",
        ),
        (
            "Quantized confidence analysis",
            ckpt_dir / "mobilenetv4_quantized_confidence_analysis.png",
        ),
        (
            "Teacher training curves",
            ckpt_dir / "teacher_training_curves.png",
        ),
        (
            "Student training curves",
            ckpt_dir / "student_training_curves.png",
        ),
        (
            "Distillation training curves",
            ckpt_dir / "distill_training_curves.png",
        ),
    ]
    lines.extend(["", "## Diagrams", ""])
    for title, path in diagrams:
        if path.exists():
            lines.extend([f"### {title}", "", f"![{title}]({rel(path)})", ""])

    artifacts = [
        ckpt_dir / "benchmark_comparison.json",
        export_dir / "onnx_validation.json",
        ckpt_dir / "pipeline_summary.json",
        export_dir / "mobilenetv4_int8.onnx",
        export_dir / "mobilenetv4.onnx",
    ]
    lines.extend(["## Files", ""])
    for path in artifacts:
        if path.exists():
            label = path.name
            target = rel(path)
            lines.append(f"- [{label}]({target})")

    int8_rows = [row for row in results if row.get("name") == "ONNX INT8 QDQ"]
    if int8_rows:
        row = int8_rows[0]
        lines.extend(
            [
                "",
                "## Deployment Note",
                "",
                "`exported_models/mobilenetv4_int8.onnx` is the Ray-Ban NPU deployment candidate.",
                f"SNR verdict: `{row.get('snr_verdict', 'n/a')}`.",
                f"Accuracy drop: `{row.get('accuracy_drop', 0.0):.4f}` "
                f"(limit `{row.get('max_accuracy_drop', 0.0):.4f}`).",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown report saved: {report_path}")
    log_artifact(report_path)
    return report_path


def _require_onnx_validation_pass(results: list[dict]) -> None:
    for result in results:
        if result.get("name") != "ONNX INT8 QDQ":
            continue
        if not result.get("accuracy_pass", False):
            raise RuntimeError(
                "INT8 ONNX validation failed: "
                f"accuracy_drop={result.get('accuracy_drop', 0.0):.4f}, "
                f"allowed<={result.get('max_accuracy_drop', 0.05):.4f}. "
                "Do not deploy this Ray-Ban NPU artifact."
            )
        if result.get("snr_fail_below_min") and not result.get("snr_pass", False):
            raise RuntimeError(
                "INT8 ONNX validation failed: "
                f"SNR={result.get('snr_db_logit', 0.0):.2f} dB, "
                f"required>={result.get('snr_min_db', 20.0):.2f} dB. "
                "Do not deploy this Ray-Ban NPU artifact."
            )


def run_quantize(cfg: dict, checkpoint_path: Path) -> Path:
    """Quantize student for edge deployment.

    Default backend applies **FULL INT8** (weights + activations), which is the
    format the Qualcomm Hexagon NPU on Meta Ray-Ban's Snapdragon AR1 Gen 1
    compiles for. A weight-only backend is kept as a legacy fallback that
    is smaller but forces activations to FP32 (CPU/NPU offload path).

    Backends (`cfg["quantize"]["backend"]`):
      - ``torchao_int8_static_activation_int8_weight``: calibrated full INT8
        using validation batches for activation scales. (default)
      - ``torchao_int8_dynamic_activation_int8_weight``: full INT8, dynamic
        activation quant at runtime. (no calibration data needed)
      - ``torchao_int8_weight_only``: legacy weight-only INT8
      - ``pytorch_dynamic_qint8``: PyTorch dynamic fallback (no torchao)

    After quantization, computes SNR in dB between FP32 and INT8 logits on a
    small validation subset. Target band: **20-40 dB** (acceptable-good).
    """
    device = "cpu"

    # Two loaded students: one stays FP32 (SNR reference), one is quantized.
    fp32_model = (
        make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"])
        .to(device)
        .eval()
    )
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, fp32_model, device)
        print(f"Loaded FP32 baseline from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = (
        make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"])
        .to(device)
        .eval()
    )
    load_checkpoint(checkpoint_path, model, device)

    original_mb = model_size_mb(checkpoint_path)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_path = ckpt_dir / "mobilenetv4_quantized.pth"

    backend = cfg.get("quantize", {}).get("backend", DEFAULT_QUANTIZE_BACKEND)

    calibration = None
    if backend == "torchao_int8_static_activation_int8_weight":
        cal_cfg = cfg.get("quantize", {}).get("calibration", {})
        cal_batch_size = cal_cfg.get(
            "batch_size", cfg.get("quantize", {}).get("snr", {}).get("batch_size", 8)
        )
        cal_batches = cal_cfg.get("batches", 32)
        cal_split = cal_cfg.get("split", "val")
        cal_train, cal_val, cal_test, _ = get_dataloaders(
            data_dir=Path(cfg["data"]["dir"]),
            image_size=cfg["data"]["image_size"],
            num_classes=cfg["data"]["num_classes"],
            batch_size=cal_batch_size,
            workers=cfg["data"]["workers"],
            mixup=False,
        )
        calibration_loader = {
            "train": cal_train,
            "val": cal_val,
            "test": cal_test,
        }.get(cal_split)
        if calibration_loader is None:
            raise ValueError(
                f"Unknown quantize.calibration.split={cal_split!r}; "
                "use 'train', 'val', or 'test'."
            )
        print(
            f"Calibrating static INT8 activations on {cal_split} split "
            f"({cal_batches} batches, batch_size={cal_batch_size})"
        )
        calibration = _calibrate_linear_activation_ranges(
            model,
            calibration_loader,
            device,
            max_batches=cal_batches,
            mixed_precision=False,
        )
        print(
            "  Calibrated linear modules: "
            f"{len(calibration.get('modules', {}))}; "
            f"observed_batches={calibration['batches']}"
        )

    backend_used = backend
    try:
        model, quantization_metadata = _apply_quantization_backend(
            model, backend, calibration
        )
        save_checkpoint(
            model,
            out_path,
            quantized=True,
            backend=backend_used,
            calibration=quantization_metadata,
        )
        quantized_mb = model_size_mb(out_path)
        saved_mb = original_mb - quantized_mb
        saved_pct = (saved_mb / original_mb * 100) if original_mb else 0.0
        print(f"Quantized model saved to {out_path}")
        print(f"  Backend: {backend_used}")
        print(f"  Original size: {original_mb:.2f} MB")
        print(f"  Quantized size: {quantized_mb:.2f} MB")
        print(f"  Space saved: {saved_mb:.2f} MB ({saved_pct:.1f}%)")

        # --- Quantization SNR scoring ---
        snr_cfg = cfg.get("quantize", {}).get("snr", {})
        _, val_snr, _, _ = get_dataloaders(
            data_dir=Path(cfg["data"]["dir"]),
            image_size=cfg["data"]["image_size"],
            num_classes=cfg["data"]["num_classes"],
            batch_size=snr_cfg.get("batch_size", 8),
            workers=cfg["data"]["workers"],
            mixup=False,
        )
        snr = quantization_snr_db(
            fp32_model=fp32_model,
            int8_model=model,
            loader=val_snr,
            device=device,
            max_batches=snr_cfg.get("batches", 16),
        )
        snr_logit = snr["snr_db_logit"]
        # Target band: 20-40 dB (acceptable-good). <20 warn, >40 excellent.
        if snr_logit < 20.0:
            verdict = "WARN: SNR below 20 dB"
        elif snr_logit >= 40.0:
            verdict = "PASS: SNR >= 40 dB (excellent)"
        else:
            verdict = "PASS: SNR in 20-40 dB band"
        print(f"Quantization SNR (logit):    {snr_logit:.2f} dB [{verdict}]")
        print(f"Quantization SNR (softmax):  {snr['snr_db_softmax']:.2f} dB")
        print(f"  Signal power: {snr['signal_power_logit']:.6f}")
        print(f"  Noise power:  {snr['noise_power_logit']:.6f}")

        log_params({"quantization": backend_used})
        log_metrics(
            {
                "original_size_mb": original_mb,
                "quantized_size_mb": quantized_mb,
                "saved_pct": saved_pct,
                "snr_db_logit": snr_logit,
                "snr_db_softmax": snr["snr_db_softmax"],
                "calibration_batches": float(
                    calibration.get("batches", 0) if calibration else 0
                ),
                "calibrated_modules": float(
                    len(calibration.get("modules", {})) if calibration else 0
                ),
            }
        )
    except ImportError as e:
        print(f"torchao unavailable ({e}); falling back to PyTorch dynamic INT8")
        backend_used = "pytorch_dynamic_qint8"
        model, quantization_metadata = _apply_quantization_backend(model, backend_used)
        save_checkpoint(
            model,
            out_path,
            quantized=True,
            backend=backend_used,
            calibration=quantization_metadata,
        )
        print(f"Dynamically quantized model saved to {out_path}")
        log_params({"quantization": "pytorch_dynamic_qint8"})
        log_metrics(
            {
                "original_size_mb": original_mb,
                "quantized_size_mb": model_size_mb(out_path),
            }
        )

    log_artifact(out_path)
    sync_runtime_outputs(cfg, "checkpoint_dir")
    return out_path


def run_benchmark(
    cfg: dict, checkpoint_path: Path, *, class_names: Optional[list] = None
) -> dict:
    """Benchmark a checkpoint. Returns dict of metrics.

    Also generates confusion matrix and confidence-analysis plots when
    ``class_names`` is provided.
    """
    device = get_device(cfg.get("device", "auto"))
    mp = cfg["data"].get("mixed_precision", False)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, device = _make_student_for_checkpoint(cfg, checkpoint_path, device)
    mp = mp and device != "cpu"

    _, _, test_loader, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["benchmark"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )

    crit = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, crit, device, mixed_precision=mp)
    img_size = cfg["data"]["image_size"]
    latency = benchmark_latency(model, device, input_size=(1, 3, img_size, img_size))
    flops = count_flops(model, input_size=(1, 3, img_size, img_size))
    size = model_size_mb(checkpoint_path)

    metrics = {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "latency_ms": latency,
        "model_size_mb": size,
        "gflops": flops,
    }

    print("Benchmark results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Confusion matrix + confidence analysis ---
    if class_names is not None:
        ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        tag = checkpoint_path.stem  # e.g. mobilenetv4_quantized
        print(f"\n  Generating evaluation reports for {tag}...")
        _, _, y_true, y_pred, probs = evaluate_with_predictions(
            model, test_loader, crit, device, mixed_precision=mp
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            ckpt_dir / f"{tag}_confusion_matrix.png",
            normalize="true",
        )
        plot_confidence_analysis(
            y_true,
            y_pred,
            probs,
            class_names,
            ckpt_dir / f"{tag}_confidence_analysis.png",
        )
        sync_runtime_outputs(cfg, "checkpoint_dir")

    log_metrics(metrics)
    log_params({"benchmark.device": device, "benchmark.runs": 100})
    return metrics


def run_full_benchmark_comparison(cfg: dict, class_names: list[str]) -> list[dict]:
    """Evaluate all available checkpoints and produce a comparison report.

    Loads teacher, student-finetuned, student-distilled, and
    student-quantized checkpoints (whichever exist), evaluates each on the
    test set, and returns a list of result dicts suitable for
    ``plot_benchmark_comparison`` and ``print_benchmark_table``.

    Also generates ``benchmark_comparison.png`` and
    ``benchmark_comparison.txt`` in the checkpoint directory.
    """
    device = get_device(cfg.get("device", "auto"))
    mp = cfg["data"].get("mixed_precision", False)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    img_size = cfg["data"]["image_size"]

    _, _, test_loader, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=img_size,
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["benchmark"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    crit = nn.CrossEntropyLoss()

    print(f"  Report class order: {class_names}")
    entries: list[tuple[str, Path, str]] = [
        (
            "Teacher (EfficientNet-B3)",
            ckpt_dir / "teacher_for_distill.pth",
            "teacher",
        ),
        ("Student Fine-tuned", ckpt_dir / "mobilenetv4_finetuned.pth", "student"),
        ("Student Distilled", ckpt_dir / "mobilenetv4_distilled.pth", "student"),
        ("Student Quantized", ckpt_dir / "mobilenetv4_quantized.pth", "student"),
    ]

    results: list[dict] = []
    for name, path, kind in entries:
        if not path.exists():
            print(f"  [SKIP] {name}: checkpoint not found at {path}")
            continue
        print(f"\n  Benchmarking {name}...")
        eval_device = device
        eval_mp = mp
        if kind == "teacher":
            model = make_teacher(cfg["data"]["num_classes"]).to(device).eval()
            load_checkpoint(path, model, device)
        else:
            model, eval_device = _make_student_for_checkpoint(cfg, path, device)
            eval_mp = mp and eval_device != "cpu"
        test_loss, test_acc = evaluate(
            model, test_loader, crit, eval_device, mixed_precision=eval_mp
        )
        flops = count_flops(model, input_size=(1, 3, img_size, img_size))
        size = model_size_mb(path)
        params = count_params(model)
        result = {
            "name": name,
            "params_m": params / 1e6,
            "test_acc": test_acc,
            "size_mb": size,
            "gflops": flops,
            "test_loss": test_loss,
        }
        results.append(result)
        print(
            f"    acc={test_acc:.4f}  loss={test_loss:.4f}  "
            f"params={params:,}  size={size:.2f}MB  gflops={flops:.3f}"
        )

    if results:
        onnx_results = validate_onnx_exports(cfg)
        if onnx_results:
            results.extend(onnx_results)
        print_benchmark_table(results)
        plot_benchmark_comparison(results, ckpt_dir / "benchmark_comparison.png")
        summary_path = ckpt_dir / "benchmark_comparison.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Comparison summary saved: {summary_path}")
        _write_markdown_report(cfg, results)
        sync_runtime_outputs(cfg, "checkpoint_dir")
    else:
        print("  [WARN] No checkpoints found for benchmark comparison.")

    return results


def run_export(cfg: dict, checkpoint_path: Path) -> dict:
    """Export to TorchScript (.pt) and torch.export (.pt2). Returns dict of paths."""
    device = "cpu"
    require_int8_onnx = cfg.get("export", {}).get("int8_onnx", True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, _ = _make_student_for_checkpoint(cfg, checkpoint_path, device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Source checkpoint size: {model_size_mb(checkpoint_path):.2f} MB")

    export_dir = Path(cfg["paths"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    results = {}

    # TorchScript
    try:
        traced = torch.jit.trace(model, dummy)
        ts_path = export_dir / "mobilenetv4.pt"
        traced.save(str(ts_path))  # type: ignore
        print(f"TorchScript -> {ts_path}")
        print(f"  TorchScript size: {model_size_mb(ts_path):.2f} MB")
        results["torchscript"] = str(ts_path)
        log_artifact(ts_path)
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # torch.export
    try:
        ep = torch.export.export(model, (dummy,))
        ep_path = export_dir / "mobilenetv4.pt2"
        torch.export.save(ep, ep_path)  # type: ignore[arg-type]
        print(f"torch.export -> {ep_path}")
        print(f"  torch.export size: {model_size_mb(ep_path):.2f} MB")
        results["torch_export"] = str(ep_path)
        log_artifact(ep_path)
    except Exception as e:
        print(f"torch.export failed: {e}")

    # ONNX (for Snapdragon AR1 QNN compilation path)
    if cfg.get("export", {}).get("onnx", True):
        try:
            onnx_checkpoint = _resolve_fp32_student_checkpoint(cfg, checkpoint_path)
            onnx_model = (
                make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"])
                .to(device)
                .eval()
            )
            load_checkpoint(onnx_checkpoint, onnx_model, device)
            onnx_path = export_dir / "mobilenetv4.onnx"
            torch.onnx.export(
                model=onnx_model,
                args=(dummy,),
                f=str(onnx_path),
                opset_version=cfg.get("export", {}).get("opset", 17),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "logits": {0: "batch"},
                },
            )
            print(f"ONNX -> {onnx_path}")
            print(f"  ONNX source checkpoint: {onnx_checkpoint}")
            print(f"  ONNX size: {model_size_mb(onnx_path):.2f} MB")
            results["onnx"] = str(onnx_path)
            results["onnx_source_checkpoint"] = str(onnx_checkpoint)
            log_artifact(onnx_path)

            if cfg.get("export", {}).get("int8_onnx", True):
                int8_onnx_path = export_dir / "mobilenetv4_int8.onnx"
                if _export_calibrated_int8_onnx(cfg, onnx_path, int8_onnx_path):
                    print(f"INT8 ONNX -> {int8_onnx_path}")
                    print(f"  INT8 ONNX size: {model_size_mb(int8_onnx_path):.2f} MB")
                    results["int8_onnx"] = str(int8_onnx_path)
                    results["deployment_onnx"] = str(int8_onnx_path)
                    results["deployment_precision"] = "int8_qdq"
                    log_artifact(int8_onnx_path)
                    onnx_validation = validate_onnx_exports(cfg)
                    results["onnx_validation"] = onnx_validation
                    _require_onnx_validation_pass(onnx_validation)
                else:
                    raise RuntimeError(
                        "INT8 ONNX export is required for Ray-Ban NPU deployment "
                        "but was not produced."
                    )
        except Exception as e:
            print(f"ONNX export failed: {e}")
            if require_int8_onnx:
                raise

    log_params(
        {
            "export.image_size": cfg["data"]["image_size"],
            "export.arch": cfg["finetune"]["arch"],
        }
    )
    sync_runtime_outputs(cfg, "export_dir")
    return results
