"""
Core utilities: data loading, model creation, metrics, and training primitives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm


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
) -> tuple:
    """Train for one epoch. Optionally performs knowledge distillation."""
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
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> tuple:
    """Evaluate model on a dataset."""
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
