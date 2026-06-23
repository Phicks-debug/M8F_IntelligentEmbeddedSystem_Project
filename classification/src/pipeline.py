"""
Pipeline stage runners for mushroom classification.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.core import (
    _live_mem_snapshot,
    benchmark_latency,
    build_scheduler,
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
from src.tracking import log_artifact, log_metrics, log_params
from src.utils import get_device


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
) -> float:
    """Phase A: head-only probe training shared by `run_train_teacher` and
    `run_finetune`. Freezes params whose name does NOT satisfy
    `freeze_predicate(name)`, builds `opt_probe` (AdamW on the trainable
    subset) with `CrossEntropyLoss(label_smoothing)`, runs `probe_epochs`
    of standard train/val on the probe loaders, and logs the best probe
    val_acc. Returns the best probe val_acc.

    Caller passes `probe_epochs` resolved from config (so the historical
    asymmetry between the teacher default 5 and the student default 3 is
    preserved).
    """
    for name, param in model.named_parameters():
        if not freeze_predicate(name):
            param.requires_grad_(False)

    opt_probe = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg[cfg_key]["lr"],
        weight_decay=cfg[cfg_key]["weight_decay"],
    )

    print(f"\n[{banner}]")
    best_probe = 0.0
    for epoch in range(1, probe_epochs + 1):
        train_one_epoch(
            model, train_loader, crit, opt_probe, epoch, device,
            mixed_precision=mixed_precision,
        )
        _, val_acc = evaluate(
            model, val_loader, eval_crit, device,
            mixed_precision=mixed_precision,
        )
        print(f"  Probe epoch {epoch}: val_acc={val_acc:.4f}")
        if val_acc > best_probe:
            best_probe = val_acc
    print(f"  Best probe val_acc: {best_probe:.4f}")
    log_metrics({log_metric_key: best_probe})
    return best_probe


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
) -> tuple[float, float]:
    """Resume-aware training loop shared by `run_train_teacher` Phase B,
    `run_finetune` Phase B, and `run_distill`'s main loop.

    Builds optimizer + scheduler from ``cfg[cfg_key]``, restores from the
    ``*.last.pth`` companion if present, runs the per-epoch train -> memory
    snapshot -> validate -> log -> save-best-and-resume -> early-stop body,
    then loads the best ckpt and evaluates on the test split.

    Returns ``(best_val_acc, test_acc)`` so the caller can log them with its
    own prefix without leaking the helper's metric-naming convention.
    """
    device = model_device(model)
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg[cfg_key]["lr"],
        weight_decay=cfg[cfg_key]["weight_decay"],
    )
    sched = build_scheduler(opt, cfg[cfg_key]["warmup"], cfg[cfg_key]["epochs"])
    last_ckpt_path = _last_ckpt_path(ckpt_path)
    start_epoch, best_acc, stall = _try_resume(ckpt_path, model, opt, sched)
    patience = cfg.get("patience", 10)
    target_epochs = cfg[cfg_key]["epochs"]

    if start_epoch < target_epochs:
        for epoch in range(start_epoch + 1, target_epochs + 1):
            train_kwargs: dict = {}
            if teacher is not None and distill_cfg is not None:
                train_kwargs = {"teacher": teacher, "distill_cfg": distill_cfg}
            train_one_epoch(
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
            _, val_acc = evaluate(
                model,
                val_loader,
                eval_crit,
                device,
                mixed_precision=mixed_precision,
            )
            print(f"  Epoch {epoch}: val_acc={val_acc:.4f}")
            log_metrics({log_metric_key: val_acc}, step=epoch)

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
    return best_acc, test_acc


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

    train_probe, val_probe, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    train_full, val_full, test_full = get_dataloaders(
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

    # Phase A: Probe
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["finetune"].get("smoothing", 0.0))
    eval_crit = nn.CrossEntropyLoss()
    _run_probe(
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

    # Phase B: Full fine-tune
    print("\n[TEACHER FULL] Fine-tuning all parameters")
    for param in model.parameters():
        param.requires_grad_(True)

    best_acc, test_acc = _run_resumable_epoch_loop(
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

    return ckpt_path


def run_finetune(cfg: dict) -> Path:
    """Fine-tune MobileNetV4 student. Returns path to best checkpoint."""
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))
    mp = cfg["data"].get("mixed_precision", False)

    train_probe, val_probe, _ = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=False,
    )
    train_full, val_full, test_full = get_dataloaders(
        data_dir=Path(cfg["data"]["dir"]),
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        batch_size=cfg["finetune"]["batch_size"],
        workers=cfg["data"]["workers"],
        mixup=True,
    )

    model = make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"]).to(device)
    print(f"Student params: {count_params(model):,}")

    log_params(
        {
            "finetune.arch": cfg["finetune"]["arch"],
            "finetune.epochs": cfg["finetune"]["epochs"],
            "finetune.lr": cfg["finetune"]["lr"],
            "finetune.batch_size": cfg["finetune"]["batch_size"],
        }
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "mobilenetv4_finetuned.pth"

    # Phase A: Probe
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["finetune"].get("smoothing", 0.0))
    eval_crit = nn.CrossEntropyLoss()
    _run_probe(
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
    )

    # Phase B: Full fine-tune
    print("\n[FULL] Fine-tuning all parameters")
    for param in model.parameters():
        param.requires_grad_(True)

    best_acc, test_acc = _run_resumable_epoch_loop(
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
    )
    print(f"  Student test_acc: {test_acc:.4f}")
    log_metrics({"finetune_test_acc": test_acc, "finetune_best_val_acc": best_acc})
    log_artifact(ckpt_path)

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

    train, val, test = get_dataloaders(
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

    opt = optim.AdamW(
        student.parameters(),
        lr=cfg["distill"]["lr"],
        weight_decay=cfg["distill"]["weight_decay"],
    )
    build_scheduler(opt, cfg["distill"]["warmup"], cfg["distill"]["epochs"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["finetune"].get("smoothing", 0.0))
    eval_crit = nn.CrossEntropyLoss()

    distill_cfg = {"T": cfg["distill"]["temperature"], "alpha": cfg["distill"]["alpha"]}

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "mobilenetv4_distilled.pth"

    best_acc, test_acc = _run_resumable_epoch_loop(
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

    return ckpt_path


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def run_quantize(cfg: dict, checkpoint_path: Path) -> Path:
    """Quantize student for edge deployment.

    Default backend applies **FULL INT8** (weights + activations), which is the
    format the Qualcomm Hexagon NPU on Meta Ray-Ban's Snapdragon AR1 Gen 1
    compiles for. A weight-only backend is kept as a legacy fallback that
    is smaller but forces activations to FP32 (CPU/NPU offload path).

    Backends (`cfg["quantize"]["backend"]`):
      - ``torchao_int8_dynamic_activation_int8_weight``: full INT8, dynamic
        activation quant at runtime. (default; no calibration data needed)
      - ``torchao_int8_weight_only``: legacy weight-only INT8
      - ``pytorch_dynamic_qint8``: PyTorch dynamic fallback (no torchao)

    After quantization, computes SNR in dB between FP32 and INT8 logits on a
    small validation subset. Target band: **20-40 dB** (acceptable-good).
    """
    device = "cpu"

    # Two copies of the loaded student: one stays FP32 (SNR reference),
    # the other is quantized in-place.
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
    model = copy.deepcopy(fp32_model)

    original_mb = _size_mb(checkpoint_path)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_path = ckpt_dir / "mobilenetv4_quantized.pth"

    backend = cfg.get("quantize", {}).get(
        "backend", "torchao_int8_dynamic_activation_int8_weight"
    )

    backend_used = backend
    try:
        if backend == "torchao_int8_dynamic_activation_int8_weight":
            from torchao.quantization import (
                int8_dynamic_activation_int8_weight,  # type: ignore
                quantize_,
            )

            quantize_(model, int8_dynamic_activation_int8_weight())
        elif backend == "torchao_int8_weight_only":
            from torchao.quantization import int8_weight_only, quantize_  # type: ignore

            quantize_(model, int8_weight_only())
        else:
            raise ValueError(
                f"Unknown quantize backend: {backend!r}. Use one of "
                "'torchao_int8_dynamic_activation_int8_weight', "
                "'torchao_int8_weight_only'."
            )

        save_checkpoint(model, out_path)
        quantized_mb = _size_mb(out_path)
        saved_mb = original_mb - quantized_mb
        saved_pct = (saved_mb / original_mb * 100) if original_mb else 0.0
        print(f"Quantized model saved to {out_path}")
        print(f"  Backend: {backend_used}")
        print(f"  Original size: {original_mb:.2f} MB")
        print(f"  Quantized size: {quantized_mb:.2f} MB")
        print(f"  Space saved: {saved_mb:.2f} MB ({saved_pct:.1f}%)")

        # --- Quantization SNR scoring ---
        snr_cfg = cfg.get("quantize", {}).get("snr", {})
        _, val_snr, _ = get_dataloaders(
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
            }
        )
    except ImportError as e:
        print(f"torchao unavailable ({e}); falling back to PyTorch dynamic INT8")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        save_checkpoint(model, out_path)
        print(f"Dynamically quantized model saved to {out_path}")
        log_params({"quantization": "pytorch_dynamic_qint8"})
        log_metrics(
            {
                "original_size_mb": original_mb,
                "quantized_size_mb": _size_mb(out_path),
            }
        )

    log_artifact(out_path)
    return out_path


def run_benchmark(cfg: dict, checkpoint_path: Path) -> dict:
    """Benchmark a checkpoint. Returns dict of metrics."""
    device = get_device(cfg.get("device", "auto"))
    mp = cfg["data"].get("mixed_precision", False)
    model = (
        make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"])
        .to(device)
        .eval()
    )

    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, device)

    _, _, test_loader = get_dataloaders(
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

    log_metrics(metrics)
    log_params({"benchmark.device": device, "benchmark.runs": 100})
    return metrics


def run_export(cfg: dict, checkpoint_path: Path) -> dict:
    """Export to TorchScript (.pt) and torch.export (.pt2). Returns dict of paths."""
    device = "cpu"
    model = (
        make_student(cfg["finetune"]["arch"], cfg["data"]["num_classes"])
        .to(device)
        .eval()
    )

    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, device)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Source checkpoint size: {_size_mb(checkpoint_path):.2f} MB")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        print(f"  TorchScript size: {_size_mb(ts_path):.2f} MB")
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
        print(f"  torch.export size: {_size_mb(ep_path):.2f} MB")
        results["torch_export"] = str(ep_path)
        log_artifact(ep_path)
    except Exception as e:
        print(f"torch.export failed: {e}")

    # ONNX (for Snapdragon AR1 QNN compilation path)
    if cfg.get("export", {}).get("onnx", True):
        try:
            onnx_path = export_dir / "mobilenetv4.onnx"
            # Let ONNX auto-name the output (avoids a warning when the
            # source model `forward()` has no `-> str` annotation, which is
            # the case for vanilla timm CNNs). Batch axis is still dynamic.
            torch.onnx.export(
                model=model,
                args=(dummy,),
                f=str(onnx_path),
                opset_version=cfg.get("export", {}).get("opset", 17),
                input_names=["input"],
                dynamic_axes={"input": {0: "batch"}},
            )
            print(f"ONNX -> {onnx_path}")
            print(f"  ONNX size: {_size_mb(onnx_path):.2f} MB")
            results["onnx"] = str(onnx_path)
            log_artifact(onnx_path)
        except Exception as e:
            print(f"ONNX export failed: {e}")

    log_params(
        {
            "export.image_size": cfg["data"]["image_size"],
            "export.arch": cfg["finetune"]["arch"],
        }
    )
    return results
