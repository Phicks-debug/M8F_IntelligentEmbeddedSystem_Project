"""
Core utilities: data loading, model creation, metrics, and training primitives.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm


@dataclass
class ResumedState:
    """Carries the bookkeeping restored from a resume-state file."""

    epoch: int
    best_acc: float
    stall: int


def _amp_ctx(device: str, enabled: bool):
    """Return `torch.autocast(...)` if FP16 autocast is enabled and the device
    supports it; otherwise a no-op context. Used to halve activation memory on
    Apple MPS (PyTorch 2.x) and CUDA when `data.mixed_precision: true`.
    """
    if enabled and device in ("mps", "cuda"):
        return torch.autocast(device_type=device, dtype=torch.float16)
    return nullcontext()


def _live_mem_snapshot(device: str, threshold: float = 0.8) -> Optional[str]:
    """Return a one-line memory snapshot string. When system RSS exceeds
    `threshold` (default 80%), the line is prefixed with `[WARN]`.

    Reports:
      - Process RSS / system total via psutil (always, when psutil is installed).
      - MPS-allocated pool via `torch.mps.current_allocated_memory()` (MPS only).
      - CUDA-allocated pool via `torch.cuda.memory_allocated()` (CUDA only).

    On any driver error (e.g. probing MPS before the device is initialized),
    the device-level fields are silently skipped and the RSS line is still
    returned. Returns ``None`` if psutil isn't installed.
    """
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process()
    rss = proc.memory_info().rss
    total = psutil.virtual_memory().total
    pct = rss / total
    line = f"RAM {rss / 1e9:.2f}GB / {total / 1e9:.2f}GB ({pct * 100:.0f}%)"
    if device == "mps":
        try:
            mp = torch.mps.current_allocated_memory()
            line += f" | MPS allocated {mp / 1e9:.2f}GB"
        except Exception:
            pass
    elif device == "cuda":
        try:
            cu = torch.cuda.memory_allocated()
            line += f" | CUDA allocated {cu / 1e9:.2f}GB"
        except Exception:
            pass
    if pct > threshold:
        return (
            f"[WARN] {line} -- above {threshold * 100:.0f}% threshold; "
            "consider dropping finetune.batch_size / distill.batch_size"
        )
    return line


def save_resume_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    stall: int,
) -> None:
    """Persist a *resume-state* file capturing model + optimizer + scheduler
    state plus the looping bookkeeping, so a crash mid-stage is recoverable.

    Captures: model.state_dict, optimizer.state_dict, scheduler.state_dict
    (when not None), torch + python.random + numpy RNG state, plus epoch,
    best_acc, stall. Atomic write (`<path>.tmp` then `os.replace`) so a
    crash during write never leaves a half-truncated file observable.

    Pair with `try_load_resume_state`. The companion "best" ckpt saved by
    `save_checkpoint` is left untouched so downstream stages see the same
    filename.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_torch": torch.get_rng_state().cpu(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "stall": int(stall),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_load_resume_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> Optional[ResumedState]:
    """Restore state from a resume-state file.

    Returns ``None`` if the file is missing or unreadable (corrupt, wrong
    shape, primitive failure); prints a single ``[WARN]`` line on a soft
    failure so the user can decide whether to delete the bad file.
    """
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_torch"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        return ResumedState(
            epoch=int(state["epoch"]),
            best_acc=float(state["best_acc"]),
            stall=int(state["stall"]),
        )
    except (RuntimeError, ValueError, KeyError, AttributeError, EOFError) as exc:
        print(f"[WARN] Could not load resume state from {path}: {exc}; starting fresh.")
        return None


def build_transforms(image_size: int, is_train: bool) -> v2.Compose:
    """Build train or eval transforms."""
    if is_train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15.0, 15.0), fill=128),
                v2.RandAugment(num_ops=2, magnitude=9),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _mixup_collate(batch, num_classes: int):
    """Collate function with CutMix + MixUp."""
    imgs, lbls = torch.utils.data.default_collate(batch)
    if imgs.size(0) > 1:
        imgs, lbls = v2.MixUp(alpha=0.2, num_classes=num_classes)(
            v2.CutMix(alpha=1.0, num_classes=num_classes)(imgs, lbls)
        )
    return imgs, lbls


def get_dataloaders(
    data_dir: Path,
    image_size: int,
    num_classes: int,
    batch_size: int,
    workers: int = 0,
    mixup: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, val, test DataLoaders for ImageFolder dataset.
    Expected structure: data_dir/{train,val,test}/class_name/*.jpg
    """
    train_tf = build_transforms(image_size, is_train=True)
    eval_tf = build_transforms(image_size, is_train=False)

    train_ds = datasets.ImageFolder(data_dir / "train", train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", eval_tf)

    assert len(train_ds.classes) == num_classes, (
        f"num_classes={num_classes} != ImageFolder classes={len(train_ds.classes)}. "
        "Re-run data preprocessing."
    )

    # Bias check: warn if class counts are skewed (max/min ratio > warn_ratio).
    assert_class_balance(train_ds, warn_ratio=2.0)

    kwargs: dict = dict(batch_size=batch_size, num_workers=workers, pin_memory=False)

    train = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=(lambda batch: _mixup_collate(batch, num_classes))
        if mixup
        else None,
        **kwargs,
    )
    val = DataLoader(val_ds, shuffle=False, **kwargs)
    test = DataLoader(test_ds, shuffle=False, **kwargs)

    return train, val, test


def assert_class_balance(ds, warn_ratio: float = 2.0) -> dict:
    """Print per-class counts from a dataset and warn if imbalance exceeds ratio.

    Helps catch dataset skew **before** training starts (which can bias a
    classifier toward majority classes). Returns ``dict[label -> count]``.

    Args:
        ds: any object exposing a list-like ``.targets`` attribute (this
            matches torchvision's ``ImageFolder`` / ``Subset``). Falls back
            to enumerating ``ds[i][1]`` otherwise.
        warn_ratio: emit a WARN line when ``max_count / min_count > warn_ratio``.

    Returns:
        ``{label: count}`` mapping the same labels returned by ``ds[i][1]``.
    """
    from collections import Counter

    targets_attr = getattr(ds, "targets", None)
    if targets_attr is None:
        try:
            targets_attr = [int(ds[i][1]) for i in range(len(ds))]
        except Exception as exc:
            raise RuntimeError(
                "assert_class_balance: ds has no .targets and no indexable (sample, label) access"
            ) from exc
    counts = Counter(targets_attr)
    n_total = sum(counts.values())
    print(f"Class counts (train n={n_total}): {dict(sorted(counts.items()))}")
    if counts:
        max_c = max(counts.values())
        min_c = min(counts.values())
        ratio = max_c / max(min_c, 1)
        if ratio > warn_ratio:
            print(
                f"  [WARN] Class imbalance ratio = {ratio:.2f}× (max/min) exceeds "
                f"threshold {warn_ratio:.1f}×. Consider WeightedRandomSampler or "
                "class-weighted CrossEntropyLoss."
            )
    return dict(counts)


def make_student(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV4 student model via timm."""
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def make_teacher(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B3 teacher model via torchvision."""
    from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    m = efficientnet_b3(weights=weights)
    in_features = cast(int, m.classifier[1].in_features)
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def count_params(m: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(model: nn.Module, path: Path, **kwargs) -> None:
    """Save model state dict plus metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **kwargs}, path)


def load_checkpoint(
    path: Path, model: nn.Module, device: str, strict: bool = True
) -> tuple:
    """Load checkpoint into model. Returns (model, metadata)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"], strict=strict)
    return model, state


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
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
    )


def distillation_loss(s_logits, t_logits, targets, T: float, alpha: float, criterion):
    """KL divergence on softened distributions + CE on hard labels."""
    ce = criterion(s_logits, targets)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean",
    ) * (T**2)
    return alpha * kl + (1 - alpha) * ce


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
    teacher: Optional[nn.Module] = None,
    distill_cfg: Optional[Dict] = None,
    mixed_precision: bool = False,
) -> tuple:
    """Train for one epoch. Optionally performs knowledge distillation."""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with _amp_ctx(device, mixed_precision):
            outputs = model(images)
            if teacher is not None and distill_cfg is not None:
                with torch.no_grad():
                    t_logits = teacher(images)
                loss = distillation_loss(
                    outputs,
                    t_logits,
                    targets,
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
                    acc_targets = (
                        targets.argmax(dim=1) if targets.ndim == 2 else targets
                    )
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
    mixed_precision: bool = False,
) -> tuple:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with _amp_ctx(device, mixed_precision):
            outputs = model(images)
        total_loss += criterion(outputs, targets).item() * images.size(0)
        total_acc += accuracy(outputs, targets)
        n += images.size(0)
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_acc / n


def count_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> float:
    """Estimate FLOPs using fvcore if available."""
    try:
        from fvcore.nn import FlopCountAnalysis

        dummy = torch.randn(*input_size)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total() / 1e9  # GFLOPs
    except ImportError:
        return 0.0


def benchmark_latency(
    model: nn.Module, device: str, input_size: tuple = (1, 3, 224, 224), runs: int = 100
) -> float:
    """Measure average inference latency in ms."""
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy)

    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(runs):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000  # ms


def model_size_mb(path: Path) -> float:
    """Return model file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def quantization_snr_db(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 32,
) -> dict[str, float]:
    """Compute Signal-to-Noise Ratio (dB) between FP32 and INT8 model outputs.

    SNR_dB = 10 * log10(P_signal / P_noise)

    where P_signal = mean over batch of mean(|fp32_logit|^2)
    and  P_noise  = mean over batch of mean(|(fp32 - int8)_logit|^2).

    Computed on logits (pre-softmax). Also returns softmax-level SNR for
    interpretability: a high logit-SNR with low softmax-SNR suggests the
    quantization is preserving prediction magnitudes but compressing
    probability mass.

    Qualitative bands for INT8 quantized vision models (logit SNR):
      < 15 dB  — severe degradation, do not deploy
      15-20 dB — degraded (noticeable accuracy drop)
      20-30 dB — acceptable (small accuracy drop)
      30-40 dB — good (target band for deployment)
      > 40 dB  — excellent (no measurable degradation)

    Args:
        fp32_model: baseline FP32 model (eval mode).
        int8_model: quantized INT8 model (eval mode).
        loader: small DataLoader for SNR evaluation; does not need labels.
        device: device for inference (typically `cpu` for INT8).
        max_batches: cap on batches to bound runtime.

    Returns:
        Dict with `snr_db_logit`, `snr_db_softmax`, `signal_power`,
        `noise_power`. Returns `inf` for snr when noise power is 0.
    """
    import math

    fp32_model.eval()
    int8_model.eval()

    sig_powers: list[float] = []
    noise_powers: list[float] = []
    sig_softmax: list[float] = []
    noise_softmax: list[float] = []
    seen = 0

    with torch.inference_mode():
        for images, _ in loader:
            if seen >= max_batches:
                break
            images = images.to(device)
            fp32_out = fp32_model(images)
            int8_out = int8_model(images)
            diff = fp32_out - int8_out
            sig_powers.append((fp32_out**2).mean().item())
            noise_powers.append((diff**2).mean().item())
            fp32_p = F.softmax(fp32_out, dim=1)
            int8_p = F.softmax(int8_out, dim=1)
            sig_softmax.append((fp32_p**2).mean().item())
            noise_softmax.append(((fp32_p - int8_p) ** 2).mean().item())
            seen += 1

    if seen == 0:
        raise ValueError("quantization_snr_db: loader produced 0 batches")

    sig = sum(sig_powers) / seen
    noise = sum(noise_powers) / seen
    sig_s = sum(sig_softmax) / seen
    noise_s = sum(noise_softmax) / seen

    snr_logit = 10.0 * math.log10(max(sig / max(noise, 1e-12), 1e-12))
    snr_softmax = 10.0 * math.log10(max(sig_s / max(noise_s, 1e-12), 1e-12))

    return {
        "snr_db_logit": snr_logit,
        "snr_db_softmax": snr_softmax,
        "signal_power_logit": sig,
        "noise_power_logit": noise,
    }
