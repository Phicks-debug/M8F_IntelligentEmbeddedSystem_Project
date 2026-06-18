"""
Mushroom Classification — End-to-End Training Pipeline
=====================================================
Stages:
  0. Fine-tune Teacher (EfficientNet-B3) on mushroom data
  1. Fine-tune Student (MobileNetV4) — frozen backbone probe + full fine-tune
  2. Distill knowledge from trained teacher to student
  3. Quantize student (torchao int8)
  4. Export (TorchScript + ExecuTorch)
"""

import json
import typing
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

if torch.backends.mps.is_available():
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_dir: Path = Path("data/processed/classification_data")
    ckpt_dir: Path = Path("checkpoints")
    export_dir: Path = Path("exported_models")

    num_classes: int = 8
    image_size: int = 224

    # Student (MobileNetV4)
    ft_arch: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    ft_epochs: int = 50
    ft_probe_epochs: int = 3
    ft_lr: float = 1e-3
    ft_wd: float = 1e-4
    ft_batch: int = 64
    ft_warmup: int = 2
    ft_smoothing: float = 0.0

    # Teacher (EfficientNet-B3)
    teacher_arch: str = "efficientnet_b3"
    teacher_epochs: int = 30
    teacher_probe_epochs: int = 5
    teacher_lr: float = 5e-4
    teacher_wd: float = 1e-4
    teacher_batch: int = 32
    teacher_warmup: int = 3

    # Distillation
    distill_epochs: int = 30
    distill_lr: float = 3e-4
    distill_wd: float = 1e-4
    distill_batch: int = 64
    distill_T: float = 6.0
    distill_alpha: float = 0.8
    distill_warmup: int = 3

    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 0
    seed: int = 42
    patience: int = 10

    quick_test: bool = False


cfg = Config()
for d in [cfg.ckpt_dir, cfg.export_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"PyTorch {torch.__version__}, device={cfg.device}, workers={cfg.workers}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transforms(is_train: bool):
    if is_train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(15, fill=128),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(cfg.image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _mixup_collate(batch):
    imgs, lbls = torch.utils.data.default_collate(batch)
    if imgs.size(0) > 1:
        imgs, lbls = v2.MixUp(alpha=0.2, num_classes=cfg.num_classes)(
            v2.CutMix(alpha=1.0, num_classes=cfg.num_classes)(imgs, lbls)
        )
    return imgs, lbls


def dataloaders(batch_size: int, mixup: bool = False):
    train_tf = build_transforms(is_train=True)
    eval_tf = build_transforms(is_train=False)

    train_ds = datasets.ImageFolder(cfg.data_dir / "train", train_tf)
    val_ds = datasets.ImageFolder(cfg.data_dir / "val", eval_tf)
    test_ds = datasets.ImageFolder(cfg.data_dir / "test", eval_tf)

    assert len(train_ds.classes) == cfg.num_classes, (
        f"cfg.num_classes={cfg.num_classes} != ImageFolder classes={len(train_ds.classes)}. "
        "Re-run data preprocessing."
    )

    train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=False,
        collate_fn=_mixup_collate if mixup else None,
    )
    val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False,
    )
    test = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_student(num_classes: int):
    import timm
    return timm.create_model(
        cfg.ft_arch,
        pretrained=True,
        num_classes=num_classes,
    )


def make_teacher(num_classes: int):
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    last = typing.cast(nn.Linear, m.classifier[1])
    m.classifier[1] = nn.Linear(last.in_features, num_classes)
    return m


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Metrics & Utilities
# ---------------------------------------------------------------------------
def accuracy(out: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """Top-k accuracy. Returns COUNT of correct predictions (caller divides by n)."""
    if target.ndim == 2:
        target = target.argmax(dim=1)
    _, pred = out.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.float().sum().item()


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Warmup + cosine annealing LR scheduler (step once per epoch)."""
    if warmup_epochs > 1:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)


def save_checkpoint(model: nn.Module, path: Path, **kwargs):
    torch.save({"state_dict": model.state_dict(), **kwargs}, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: Path, model: nn.Module, device: str):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model, state


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    teacher: nn.Module | None = None,
    distill_cfg: dict | None = None,
):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if teacher is not None and distill_cfg is not None:
            with torch.no_grad():
                t_logits = teacher(images)
            loss = distillation_loss(
                outputs, t_logits, targets,
                T=distill_cfg["T"],
                alpha=distill_cfg["alpha"],
                criterion=criterion,
            )
            acc_targets = targets.argmax(dim=1) if targets.ndim == 2 else targets
            acc_val = accuracy(outputs, acc_targets)
        else:
            if isinstance(targets, tuple):
                loss = criterion(outputs, targets[0])
                acc_val = accuracy(outputs, targets[0].argmax(dim=1))
            else:
                loss = criterion(outputs, targets)
                acc_targets = targets.argmax(dim=1) if targets.ndim == 2 else targets
                acc_val = accuracy(outputs, acc_targets)

        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += acc_val
        n += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc_val / bs:.4f}"})

    return total_loss / n, total_acc / n


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, targets).item() * images.size(0)
        total_acc += accuracy(outputs, targets)
        n += images.size(0)
    if n == 0:
        return 0.1, 0.1
    return total_loss / n, total_acc / n


def distillation_loss(s_logits, t_logits, targets, T: float, alpha: float, criterion):
    ce = criterion(s_logits, targets)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean",
    ) * (T ** 2)
    return alpha * kl + (1 - alpha) * ce


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def run_probe_phase(model, train_loader, val_loader, epochs, lr, wd, device, tag):
    """Freeze backbone, train only head/classifier."""
    print(f"\n  [{tag}] Probe phase — training classifier head only ({epochs} epochs)")
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name:
            param.requires_grad_(False)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    sched = build_scheduler(opt, 1, epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.ft_smoothing)
    eval_crit = nn.CrossEntropyLoss()

    best_acc, stall = 1e-9, 1e-9
    for epoch in range(1, epochs + 1):
        if cfg.quick_test and epoch > 1:
            break
        train_loss, train_acc = train_one_epoch(model, train_loader, crit, opt, epoch, device)
        if sched is not None:
            sched.step()
        val_loss, val_acc = evaluate(model, val_loader, eval_crit, device)
        print(f"    Epoch {epoch:2d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            stall = 0
        else:
            stall += 1
            if stall >= cfg.patience:
                print(f"    Early stop at epoch {epoch}")
                break
    print(f"    Best probe val_acc: {best_acc:.4f}")
    return best_acc


def run_full_finetune(model, train_loader, val_loader, epochs, lr, wd, warmup, device, ckpt_path, tag):
    """Unfreeze all and fine-tune entire model."""
    print(f"\n  [{tag}] Full fine-tune — all parameters ({epochs} epochs max)")
    for param in model.parameters():
        param.requires_grad_(True)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = build_scheduler(opt, warmup, epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.ft_smoothing)
    eval_crit = nn.CrossEntropyLoss()

    best_acc, stall = 1e-9, 1e-9
    for epoch in range(1, epochs + 1):
        if cfg.quick_test and epoch > 1:
            break
        train_loss, train_acc = train_one_epoch(model, train_loader, crit, opt, epoch, device)
        if sched is not None:
            sched.step()
        val_loss, val_acc = evaluate(model, val_loader, eval_crit, device)
        print(f"    Epoch {epoch:2d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckpt_path, epoch=epoch, val_acc=val_acc)
            stall = 0
        else:
            stall += 1
            if stall >= cfg.patience:
                print(f"    Early stop at epoch {epoch}")
                break
    print(f"    Best full-tune val_acc: {best_acc:.4f}")
    return best_acc


# ---------------------------------------------------------------------------
# Stage 0: Train Teacher (EfficientNet-B3)
# ---------------------------------------------------------------------------
def stage_train_teacher():
    print("\n" + "=" * 60)
    print("STAGE 1: Fine-tune Teacher (EfficientNet-B3)")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    teacher = make_teacher(cfg.num_classes).to(cfg.device)
    print(f"  Teacher params: {count_params(teacher):,}")

    train_probe, val_probe, _ = dataloaders(cfg.teacher_batch, mixup=False)
    _ = run_probe_phase(teacher, train_probe, val_probe, cfg.teacher_probe_epochs,
                        cfg.teacher_lr, cfg.teacher_wd, cfg.device, "Teacher")

    train_full, val_full, test_full = dataloaders(cfg.teacher_batch, mixup=True)
    ckpt = cfg.ckpt_dir / "teacher_efficientnet_b3.pth"
    best = run_full_finetune(teacher, train_full, val_full, cfg.teacher_epochs,
                             cfg.teacher_lr, cfg.teacher_wd, cfg.teacher_warmup,
                             cfg.device, ckpt, "Teacher")

    # Load best checkpoint before final test evaluation
    if ckpt.exists():
        load_checkpoint(ckpt, teacher, cfg.device)

    test_loss, test_acc = evaluate(teacher, test_full, nn.CrossEntropyLoss(), cfg.device)
    print(f"  Teacher test_acc: {test_acc:.4f}")

    save_checkpoint(teacher, cfg.ckpt_dir / "teacher_for_distill.pth",
                      val_acc=best, test_acc=test_acc)
    print(f"  Teacher checkpoint → {ckpt}")
    return ckpt, best


# ---------------------------------------------------------------------------
# Stage 1: Fine-tune Student (MobileNetV4)
# ---------------------------------------------------------------------------
def stage_finetune_student():
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tune Student (MobileNetV4)")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    student = make_student(cfg.num_classes).to(cfg.device)
    print(f"  Student params: {count_params(student):,}")

    # Phase A: probe (freeze backbone, train head)
    train_probe, val_probe, _ = dataloaders(cfg.ft_batch, mixup=False)
    _ = run_probe_phase(student, train_probe, val_probe, cfg.ft_probe_epochs,
                        cfg.ft_lr, cfg.ft_wd, cfg.device, "Student")

    # Phase B: full fine-tune (unfreeze all, no mixup — small model can't handle it)
    train_full, val_full, test_full = dataloaders(cfg.ft_batch, mixup=False)
    ckpt = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    best = run_full_finetune(student, train_full, val_full, cfg.ft_epochs,
                             cfg.ft_lr, cfg.ft_wd, cfg.ft_warmup,
                             cfg.device, ckpt, "Student")

    # Load best checkpoint before final test evaluation
    if ckpt.exists():
        load_checkpoint(ckpt, student, cfg.device)

    test_loss, test_acc = evaluate(student, test_full, nn.CrossEntropyLoss(), cfg.device)
    print(f"  Student test_acc: {test_acc:.4f}")
    return ckpt, best, test_acc


# ---------------------------------------------------------------------------
# Stage 2: Distill from trained teacher to student
# ---------------------------------------------------------------------------
def stage_distill():
    print("\n" + "=" * 60)
    print("STAGE 3: Knowledge Distillation")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    teacher = make_teacher(cfg.num_classes).to(cfg.device)
    teacher_ckpt = cfg.ckpt_dir / "teacher_for_distill.pth"
    if teacher_ckpt.exists():
        load_checkpoint(teacher_ckpt, teacher, cfg.device)
        print(f"  Loaded trained teacher from {teacher_ckpt}")
    else:
        print(f"  Warning: no trained teacher found at {teacher_ckpt}; using pretrained ImageNet teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = make_student(cfg.num_classes).to(cfg.device)
    ft_ckpt = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    if ft_ckpt.exists():
        load_checkpoint(ft_ckpt, student, cfg.device)
        print(f"  Loaded fine-tuned student from {ft_ckpt}")

    train_loader, val_loader, test_loader = dataloaders(cfg.distill_batch, mixup=False)
    opt = optim.AdamW(student.parameters(), lr=cfg.distill_lr, weight_decay=cfg.distill_wd)
    sched = build_scheduler(opt, cfg.distill_warmup, cfg.distill_epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.ft_smoothing)
    eval_crit = nn.CrossEntropyLoss()

    distill_cfg = {"T": cfg.distill_T, "alpha": cfg.distill_alpha}

    best_acc, stall = 0.0, 0
    ckpt_path = cfg.ckpt_dir / "mobilenetv4_distilled.pth"

    for epoch in range(1, cfg.distill_epochs + 1):
        if cfg.quick_test and epoch > 1:
            break
        train_loss, train_acc = train_one_epoch(
            student, train_loader, crit, opt, epoch, cfg.device,
            teacher=teacher, distill_cfg=distill_cfg,
        )
        if sched is not None:
            sched.step()
        val_loss, val_acc = evaluate(student, val_loader, eval_crit, cfg.device)
        print(f"    Epoch {epoch:2d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(student, ckpt_path, epoch=epoch, val_acc=val_acc)
            stall = 0
        else:
            stall += 1
            if stall >= cfg.patience:
                print(f"    Early stop at epoch {epoch}")
                break

    # Final evaluation on test set
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, student, cfg.device)
    test_loss, test_acc = evaluate(student, test_loader, eval_crit, cfg.device)
    print(f"  Distilled student test_acc: {test_acc:.4f}")
    return ckpt_path, best_acc, test_acc


# ---------------------------------------------------------------------------
# Stage 3: Quantization
# ---------------------------------------------------------------------------
def stage_quantize():
    print("\n" + "=" * 60)
    print("STAGE 4: Quantization (torchao int8)")
    print("=" * 60)

    model = make_student(cfg.num_classes).to("cpu").eval()
    src = cfg.ckpt_dir / "mobilenetv4_distilled.pth"
    if not src.exists():
        src = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    if not src.exists():
        print("  No checkpoint found; skipping quantization")
        return None

    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True)["state_dict"])
    print(f"  Loaded checkpoint from {src}")

    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(model, int8_weight_only())
        out = cfg.ckpt_dir / "mobilenetv4_quantized.pth"
        save_checkpoint(model, out)
        print("  Quantization complete (int8 weight-only)")
        return out
    except ImportError:
        print("  torchao not installed. Skipping quantization.")
        return None


# ---------------------------------------------------------------------------
# Stage 4: Export
# ---------------------------------------------------------------------------
def stage_export():
    print("\n" + "=" * 60)
    print("STAGE 5: Export")
    print("=" * 60)

    model = make_student(cfg.num_classes).to("cpu").eval()
    for src_name in ["mobilenetv4_quantized.pth", "mobilenetv4_distilled.pth", "mobilenetv4_finetuned.pth"]:
        src = cfg.ckpt_dir / src_name
        if src.exists():
            model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True)["state_dict"])
            print(f"  Loaded {src_name}")
            break
    else:
        print("  No checkpoint found; skipping export")
        return

    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    try:
        traced = torch.jit.trace(model, dummy)
        ts_path = cfg.export_dir / "mobilenetv4.pt"
        traced.save(str(ts_path))
        print(f"  TorchScript → {ts_path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")

    try:
        ep = torch.export.export(model, (dummy,))
        ep_path = cfg.export_dir / "mobilenetv4.ep"
        ep.save(str(ep_path))
        print(f"  ExecuTorch  → {ep_path}")
    except Exception as e:
        print(f"  ExecuTorch export failed: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def save_summary(results: dict):
    summary = {
        "model": cfg.ft_arch,
        "teacher": cfg.teacher_arch,
        "num_classes": cfg.num_classes,
        "image_size": cfg.image_size,
        "device": cfg.device,
        "stages": results,
    }
    out = cfg.ckpt_dir / "pipeline_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "teacher", "student", "distill", "quantize", "export"])
    parser.add_argument("--quick-test", action="store_true", help="1 epoch per phase smoke test")
    args = parser.parse_args()

    if args.quick_test:
        cfg.quick_test = True
        cfg.ft_epochs = 1
        cfg.teacher_epochs = 1
        cfg.distill_epochs = 1
        print("\n*** QUICK TEST MODE (1 epoch per phase) ***\n")

    results = {}

    if args.stage in ("all", "teacher"):
        teacher_ckpt, teacher_val = stage_train_teacher()
        results["teacher"] = {"checkpoint": str(teacher_ckpt), "val_acc": teacher_val}

    if args.stage in ("all", "student"):
        student_ckpt, student_val, student_test = stage_finetune_student()
        results["student"] = {"checkpoint": str(student_ckpt), "val_acc": student_val, "test_acc": student_test}

    if args.stage in ("all", "distill"):
        distill_ckpt, distill_val, distill_test = stage_distill()
        results["distill"] = {"checkpoint": str(distill_ckpt), "val_acc": distill_val, "test_acc": distill_test}

    if args.stage in ("all", "quantize"):
        q_ckpt = stage_quantize()
        if q_ckpt:
            results["quantize"] = {"checkpoint": str(q_ckpt)}

    if args.stage in ("all", "export"):
        stage_export()

    if args.stage == "all":
        save_summary(results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
