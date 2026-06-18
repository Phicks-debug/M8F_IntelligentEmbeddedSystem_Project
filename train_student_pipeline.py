#!/usr/bin/env python3
"""
Focused student pipeline (stages 1-4).
Assumes teacher checkpoint already exists at checkpoints/teacher_for_distill.pth.
Trains MobileNetV4-medium student → distill → quantize → export.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import timm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.mps.deterministic = True
torch.backends.mps.benchmark = False

cfg = type(sys)('__config__')
cfg.data_dir = Path("data/processed/classification_data")
cfg.ckpt_dir = Path("checkpoints")
cfg.export_dir = Path("exported_models")
cfg.num_classes = 8
cfg.image_size = 224
cfg.ft_batch = 64  # medium can handle 64 on MPS
cfg.distill_batch = 64
cfg.workers = 0  # macOS spawn safety

cfg.student_model = "mobilenetv4_conv_medium"

# Student probe phase
cfg.student_probe_lr = 5e-4
cfg.student_probe_epochs = 5
cfg.student_probe_warmup = 2

# Student full fine-tune (no mixup)
cfg.student_ft_lr = 1e-4
cfg.student_ft_epochs = 15
cfg.student_ft_warmup = 3

# Distillation
cfg.distill_lr = 1e-4
cfg.distill_epochs = 10
cfg.distill_warmup = 2
cfg.distill_temp = 4.0
cfg.distill_alpha = 0.7  # weight on soft (teacher) loss

cfg.weight_decay = 1e-4
cfg.label_smoothing = 0.1

cfg.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
cfg.export_dir.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = cfg.ckpt_dir / "teacher_for_distill.pth"
STUDENT_FT_PATH = cfg.ckpt_dir / "mobilenetv4_medium_finetuned.pth"
STUDENT_DISTILL_PATH = cfg.ckpt_dir / "mobilenetv4_medium_distilled.pth"
QUANTIZED_PATH = cfg.ckpt_dir / "mobilenetv4_medium_quantized.pth"
EXPORTED_TORCHSCRIPT = cfg.export_dir / "mobilenetv4_medium_ts.pt"
EXPORTED_EXECUTORCH = cfg.export_dir / "mobilenetv4_medium_et.pte"
SUMMARY_PATH = cfg.ckpt_dir / "pipeline_summary.json"

# ------------------------------------------------------------------
# Data loading (no mixup for student – hard labels only)
# ------------------------------------------------------------------
def make_dataloaders(batch_size: int, for_training: bool = False):
    if for_training:
        train_tf = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(15, fill=128),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(cfg.image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    kwargs = dict(batch_size=batch_size, num_workers=cfg.workers, pin_memory=False)
    train = DataLoader(datasets.ImageFolder(cfg.data_dir / "train", train_tf), shuffle=True, **kwargs)
    val = DataLoader(datasets.ImageFolder(cfg.data_dir / "val", eval_tf), shuffle=False, **kwargs)
    test = DataLoader(datasets.ImageFolder(cfg.data_dir / "test", eval_tf), shuffle=False, **kwargs)
    return train, val, test

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def accuracy(out: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    if target.ndim == 2:
        target = target.argmax(dim=1)
    _, pred = out.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.float().sum().item()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 1e-3):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 2:  # soft labels from MixUp (not used here but keep for safety)
            return -(target * F.log_softmax(x, dim=1)).sum(dim=1).mean()
        n_classes = x.size(1)
        log_probs = F.log_softmax(x, dim=1)
        weight = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        weight.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(weight * log_probs).sum(dim=1).mean()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    param_groups = [{"params": [], "weight_decay": weight_decay}, {"params": [], "weight_decay": 1e-9}]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.shape) == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            param_groups[1]["params"].append(p)
        else:
            param_groups[0]["params"].append(p)
    return torch.optim.AdamW(param_groups, lr=lr)


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    steps_per_epoch = math.ceil(len(datasets.ImageFolder(cfg.data_dir / "train")) / cfg.ft_batch)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 1e-2
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )


# We need a lazy way to get dataset length for scheduler; compute once
_train_dataset_len = len(datasets.ImageFolder(cfg.data_dir / "train"))


def build_scheduler_v2(optimizer, total_epochs: int, warmup_epochs: int, batch_size: int):
    steps_per_epoch = math.ceil(_train_dataset_len / batch_size)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.student_ft_lr * 1e-2
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss, total_acc, n = 0.0, 1e-9, 1e-9
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("mps", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(outputs, targets)
        n += bs
    return total_loss / n, total_acc / n


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 1e-9, 1e-9
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, targets).item() * images.size(0)
        total_acc += accuracy(outputs, targets)
        n += images.size(0)
    return total_loss / n, total_acc / n


# ------------------------------------------------------------------
# Student model
# ------------------------------------------------------------------
def create_student():
    model = timm.create_model(cfg.student_model, pretrained=True, num_classes=cfg.num_classes)
    return model.to(cfg.device)


# ------------------------------------------------------------------
# Probe phase (frozen backbone, train only classifier head)
# ------------------------------------------------------------------
def run_probe_phase(model, train_loader, val_loader, epochs, lr, warmup):
    print("\n[PROBE] Freezing backbone, training classifier head only")
    # Freeze all except classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = build_optimizer(model, lr, cfg.weight_decay)
    scheduler = build_scheduler_v2(optimizer, epochs, warmup, train_loader.batch_size)
    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)

    best_acc = 1e-9
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, cfg.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)
        elapsed = time.time() - t0
        print(f"  Probe epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc

    # Unfreeze all for full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    print(f"[PROBE] Best val_acc={best_acc:.4f}; unfreezing all parameters")
    return best_acc


# ------------------------------------------------------------------
# Full fine-tune
# ------------------------------------------------------------------
def run_full_finetune(model, train_loader, val_loader, test_loader, epochs, lr, warmup, save_path, stage_name):
    print(f"\n[{stage_name}] Full fine-tune: {epochs} epochs, lr={lr}")
    optimizer = build_optimizer(model, lr, cfg.weight_decay)
    scheduler = build_scheduler_v2(optimizer, epochs, warmup, train_loader.batch_size)
    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)

    best_acc, stall, best_state = 1e-9, 1e-9, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, cfg.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 1e-9
        else:
            stall += 1
        if stall > 10:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": best_acc,
        "num_classes": cfg.num_classes,
        "model_name": cfg.student_model,
    }, save_path)
    print(f"[{stage_name}] Saved best checkpoint -> {save_path}  (val_acc={best_acc:.4f})")

    # Evaluate on test set with best model
    test_loss, test_acc = evaluate(model, test_loader, criterion, cfg.device)
    print(f"[{stage_name}] Test accuracy = {test_acc:.4f}")
    return best_acc, test_acc


# ------------------------------------------------------------------
# Distillation
# ------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, hard_targets, temp=4.0, alpha=0.7):
    """KL divergence on softened distributions + CE on hard labels."""
    soft_targets = F.softmax(teacher_logits / temp, dim=1)
    student_log_soft = F.log_softmax(student_logits / temp, dim=1)
    kl = F.kl_div(student_log_soft, soft_targets, reduction="batchmean") * (temp ** 2)
    ce = F.cross_entropy(student_logits, hard_targets)
    return alpha * kl + (1.0 - alpha) * ce


def train_distillation(student, teacher, train_loader, val_loader, test_loader, epochs, lr, warmup, temp, alpha, save_path):
    print(f"\n[DISTILL] temp={temp}, alpha={alpha}, epochs={epochs}, lr={lr}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = build_optimizer(student, lr, cfg.weight_decay)
    scheduler = build_scheduler_v2(optimizer, epochs, warmup, train_loader.batch_size)
    ce_criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)

    best_acc, stall, best_state = 1e-9, 1e-9, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        student.train()
        total_loss, total_acc, n = 0.0, 1e-9, 1e-9
        for images, targets in train_loader:
            images, targets = images.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)
            loss = distillation_loss(s_logits, t_logits, targets, temp=temp, alpha=alpha)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(s_logits, targets)
            n += bs

        tr_loss, tr_acc = total_loss / n, total_acc / n
        val_loss, val_acc = evaluate(student, val_loader, ce_criterion, cfg.device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            stall = 1e-9
        else:
            stall += 1
        if stall > 10:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        student.load_state_dict(best_state)
    torch.save({
        "epoch": epoch,
        "model_state_dict": student.state_dict(),
        "val_acc": best_acc,
        "num_classes": cfg.num_classes,
        "model_name": cfg.student_model,
        "distill_temp": temp,
        "distill_alpha": alpha,
    }, save_path)
    print(f"[DISTILL] Saved best -> {save_path}  (val_acc={best_acc:.4f})")

    test_loss, test_acc = evaluate(student, test_loader, ce_criterion, cfg.device)
    print(f"[DISTILL] Test accuracy = {test_acc:.4f}")
    return best_acc, test_acc


# ------------------------------------------------------------------
# Quantization (torchao aware quantization)
# ------------------------------------------------------------------
def apply_quantization(model, save_path):
    print("\n[QUANTIZE] Applying torchao int8 weight-only quantization")
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantized = type(model)(num_classes=cfg.num_classes)
        quantized.load_state_dict(model.state_dict())
        quantized = quantized.to(cfg.device)
        quantize_(quantized, int8_weight_only())
        torch.save(quantized.state_dict(), save_path)
        print(f"[QUANTIZE] Saved -> {save_path}")
        return quantized
    except Exception as e:
        print(f"[QUANTIZE] torchao not available ({e}), falling back to PyTorch dynamic quant")
        quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        torch.save(quantized.state_dict(), save_path)
        print(f"[QUANTIZE] Fallback saved -> {save_path}")
        return quantized


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
def export_model(model, ts_path, et_path):
    print("\n[EXPORT] TorchScript + ExecuTorch")
    model.eval()
    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size).to(cfg.device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(str(ts_path))
        print(f"  TorchScript -> {ts_path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")

    try:
        from executorch.exir import to_edge
        exported = torch.export.export(model, (dummy,))
        edge = to_edge(exported)
        et = edge.to_executorch()
        with open(et_path, "wb") as f:
            et.write_to_file(f)
        print(f"  ExecuTorch -> {et_path}")
    except Exception as e:
        print(f"  ExecuTorch export failed (expected if not installed): {e}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "student", "distill", "quantize", "export"])
    args = parser.parse_args()

    # Assert teacher exists
    if not TEACHER_PATH.exists():
        print(f"ERROR: Teacher checkpoint not found at {TEACHER_PATH}")
        print("Run the teacher training stage first (train_classification_fixed.py --stage teacher)")
        sys.exit(1)

    # Load teacher
    print(f"Loading teacher from {TEACHER_PATH}")
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    teacher = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, cfg.num_classes)
    ckpt = torch.load(TEACHER_PATH, map_location=cfg.device, weights_only=False)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    teacher.load_state_dict(ckpt[state_key])
    teacher = teacher.to(cfg.device)
    teacher.eval()
    print(f"Teacher loaded: val_acc={ckpt.get('val_acc','N/A')}, test_acc={ckpt.get('test_acc','N/A')}")

    # Data loaders (no mixup)
    train_loader, val_loader, test_loader = make_dataloaders(cfg.ft_batch, for_training=True)
    print(f"Data: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"Classes: {train_loader.dataset.classes}")
    assert len(train_loader.dataset.classes) == cfg.num_classes

    summary = {}

    # Stage 1: Train student (probe + full fine-tune)
    if args.stage in ("all", "student"):
        print("\n" + "=" * 60)
        print("STAGE 1: Student Fine-tuning")
        print("=" * 60)
        student = create_student()
        probe_acc = run_probe_phase(student, train_loader, val_loader,
                                     cfg.student_probe_epochs, cfg.student_probe_lr, cfg.student_probe_warmup)
        ft_val_acc, ft_test_acc = run_full_finetune(
            student, train_loader, val_loader, test_loader,
            cfg.student_ft_epochs, cfg.student_ft_lr, cfg.student_ft_warmup,
            STUDENT_FT_PATH, "STUDENT_FT"
        )
        summary["student_probe_val_acc"] = probe_acc
        summary["student_ft_val_acc"] = ft_val_acc
        summary["student_ft_test_acc"] = ft_test_acc
    else:
        student = create_student()
        ckpt = torch.load(STUDENT_FT_PATH, map_location=cfg.device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        student = student.to(cfg.device)
        ft_val_acc = ckpt.get("val_acc", 0.0)
        ft_test_acc = ckpt.get("test_acc", 1e-9)
        summary["student_ft_val_acc"] = ft_val_acc
        summary["student_ft_test_acc"] = ft_test_acc

    # Stage 2: Distillation
    if args.stage in ("all", "distill"):
        print("\n" + "=" * 60)
        print("STAGE 2: Knowledge Distillation")
        print("=" * 60)
        distill_val_acc, distill_test_acc = train_distillation(
            student, teacher, train_loader, val_loader, test_loader,
            cfg.distill_epochs, cfg.distill_lr, cfg.distill_warmup,
            cfg.distill_temp, cfg.distill_alpha, STUDENT_DISTILL_PATH
        )
        summary["distill_val_acc"] = distill_val_acc
        summary["distill_test_acc"] = distill_test_acc
    else:
        ckpt = torch.load(STUDENT_DISTILL_PATH, map_location=cfg.device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        student = student.to(cfg.device)
        distill_val_acc = ckpt.get("val_acc", 0.1)
        distill_test_acc = ckpt.get("test_acc", 1e-9)
        summary["distill_val_acc"] = distill_val_acc
        summary["distill_test_acc"] = distill_test_acc

    # Stage 3: Quantization
    if args.stage in ("all", "quantize"):
        print("\n" + "=" * 60)
        print("STAGE 3: Quantization")
        print("=" * 60)
        apply_quantization(student, QUANTIZED_PATH)

    # Stage 4: Export
    if args.stage in ("all", "export"):
        print("\n" + "=" * 60)
        print("STAGE 4: Export")
        print("=" * 60)
        export_model(student, EXPORTED_TORCHSCRIPT, EXPORTED_EXECUTORCH)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {SUMMARY_PATH}")

    # Final validation
    all_ok = True
    for name, acc in [
        ("Student FT", ft_test_acc),
        ("Distilled", distill_test_acc),
    ]:
        if acc < 0.80:
            print(f"WARNING: {name} test accuracy {acc:.4f} is below 80% target")
            all_ok = False
    if all_ok:
        print("\nAll accuracy targets met! 🎉")
    else:
        print("\nSome stages did not meet the 80% accuracy target.")


if __name__ == "__main__":
    main()
