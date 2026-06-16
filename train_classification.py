"""
Mushroom Classification — Cloud GPU training script.
"""

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

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")

@dataclass
class Config:
    data_dir: Path = Path("data/processed/classification_data_aug")
    ckpt_dir: Path = Path("checkpoints")
    export_dir: Path = Path("exported_models")

    num_classes: int = 8
    image_size: int = 224

    ft_arch: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    ft_epochs: int = 50
    ft_probe_epochs: int = 5
    ft_lr: float = 5e-4
    ft_wd: float = 1e-4
    ft_batch: int = 64
    ft_warmup: int = 5
    ft_smoothing: float = 0.1

    teacher_arch: str = "efficientnet_b3"
    distill_epochs: int = 30
    distill_lr: float = 3e-4
    distill_wd: float = 1e-4
    distill_batch: int = 64
    distill_T: float = 6.0
    distill_alpha: float = 0.8
    distill_warmup: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 8
    seed: int = 42
    patience: int = 10

cfg = Config()
for d in [cfg.ckpt_dir, cfg.export_dir]:
    d.mkdir(parents=True, exist_ok=True)


def dataloaders(batch_size, mixup=False):
    train_tf = v2.Compose([
        v2.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(15, fill=128),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = v2.Compose([
        v2.Resize(256), v2.CenterCrop(cfg.image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def collate(batch):
        imgs, lbls = torch.utils.data.default_collate(batch)
        if mixup and imgs.size(0) > 1:
            cutmix = v2.CutMix(cfg.num_classes)
            mixup_t = v2.MixUp(cfg.num_classes, alpha=0.2)
            imgs, lbls = mixup_t(cutmix(imgs, lbls))
        return imgs, lbls

    train = DataLoader(datasets.ImageFolder(cfg.data_dir / "train", train_tf),
                       batch_size, shuffle=True, num_workers=cfg.workers,
                       pin_memory=True, collate_fn=collate if mixup else None)
    val = DataLoader(datasets.ImageFolder(cfg.data_dir / "val", eval_tf),
                     batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    test = DataLoader(datasets.ImageFolder(cfg.data_dir / "test", eval_tf),
                      batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train, val, test



def mobilenetv4():
    import timm
    return timm.create_model(cfg.ft_arch, pretrained=True, num_classes=cfg.num_classes)

def efficientnet_b3():
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    last = typing.cast(nn.Linear, m.classifier[1])
    m.classifier[1] = nn.Linear(last.in_features, cfg.num_classes)
    return m


def acc(out, target, k=1):
    _, pred = out.topk(k, 1, True, True)
    return pred.eq(target.view(1, -1).expand_as(pred))[:k].reshape(-1).float().sum(0).item()

def scheduler(opt, warmup, total):
    s1 = LinearLR(opt, 0.01, 1.0, total_iters=warmup)
    s2 = CosineAnnealingLR(opt, T_max=total - warmup, eta_min=1e-6)
    return SequentialLR(opt, [s1, s2], milestones=[warmup])

def train_epoch(model, loader, crit, opt, sched, epoch):
    model.train()
    loss_t, acc_t = 0, 0
    for imgs, targs in tqdm(loader, desc=f"Epoch {epoch}"):
        imgs, targs = imgs.to(cfg.device), targs.to(cfg.device)
        opt.zero_grad()
        out = model(imgs)
        if isinstance(targs, tuple):
            loss = crit(out, targs[0])
            ac = acc(out, targs[0].argmax(dim=1))
        else:
            loss = crit(out, targs)
            ac = acc(out, targs)
        loss.backward()
        opt.step()
        sched.step()
        loss_t += loss.item() * imgs.size(0)
        acc_t += ac
    return loss_t / len(loader.dataset), acc_t / len(loader.dataset)

@torch.inference_mode()
def evaluate(model, loader, crit):
    model.eval()
    loss_t, acc_t = 0, 0
    for imgs, targs in loader:
        imgs, targs = imgs.to(cfg.device), targs.to(cfg.device)
        out = model(imgs)
        loss_t += crit(out, targs).item() * imgs.size(0)
        acc_t += acc(out, targs)
    return loss_t / len(loader.dataset), acc_t / len(loader.dataset)

def save(m, pth, **kw):
    torch.save({"state_dict": m.state_dict(), **kw}, pth)
    print(f"  saved {pth}")



def stage_finetune():
    print("Stage 1: Fine-tune MobileNetV4")
    torch.manual_seed(cfg.seed)
    model = mobilenetv4().to(cfg.device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.ft_smoothing)

    # Phase A: Probe — freeze backbone, train classifier head only
    print("  Phase A: probe — training classifier head")
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name:
            param.requires_grad_(False)
    opt_probe = optim.AdamW(model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_wd)
    train_probe, val_probe, _ = dataloaders(cfg.ft_batch, mixup=False)
    best, stall = 0, 0
    for ep in range(1, cfg.ft_probe_epochs + 1):
        train_epoch(model, train_probe, crit, opt_probe, scheduler(opt_probe, 0, cfg.ft_probe_epochs), ep)
        _, val_acc = evaluate(model, val_probe, nn.CrossEntropyLoss())
        print(f"    probe {ep:2d}  val_acc: {val_acc:.2f}")
        if val_acc > best:
            best = val_acc; stall = 0
        else:
            stall += 1
            if stall >= cfg.patience: break
    print(f"    probe done — best: {best:.2f}")

    # Phase B: Full fine-tune — unfreeze all, apply MixUp
    print("  Phase B: full fine-tune")
    for param in model.parameters():
        param.requires_grad_(True)
    opt = optim.AdamW(model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_wd)
    sched = scheduler(opt, cfg.ft_warmup, cfg.ft_epochs)
    train, val, _ = dataloaders(cfg.ft_batch, mixup=True)
    best, stall = 0, 0
    pth = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    for ep in range(1, cfg.ft_epochs + 1):
        train_epoch(model, train, crit, opt, sched, ep)
        _, val_acc = evaluate(model, val, nn.CrossEntropyLoss())
        print(f"    epoch {ep:2d}  val_acc: {val_acc:.2f}")
        if val_acc > best:
            best = val_acc; save(model, pth, epoch=ep, val_acc=val_acc); stall = 0
        else:
            stall += 1
            if stall >= cfg.patience: break
    print(f"    fine-tune done — best: {best:.2f}")
    return pth

def stage_distill():
    print("Stage 2: Knowledge Distillation")
    teacher = efficientnet_b3().to(cfg.device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = mobilenetv4().to(cfg.device)
    ft = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    if ft.exists():
        student.load_state_dict(torch.load(ft, map_location=cfg.device)["state_dict"])

    def distill_loss(s_logits, t_logits, targets):
        ce = nn.CrossEntropyLoss(label_smoothing=cfg.ft_smoothing)(s_logits, targets)
        kl = F.kl_div(
            F.log_softmax(s_logits / cfg.distill_T, dim=1),
            F.softmax(t_logits / cfg.distill_T, dim=1),
            reduction="batchmean",
        ) * (cfg.distill_T ** 2)
        return cfg.distill_alpha * kl + (1 - cfg.distill_alpha) * ce

    opt = optim.AdamW(student.parameters(), lr=cfg.distill_lr, weight_decay=cfg.distill_wd)
    sched = scheduler(opt, cfg.distill_warmup, cfg.distill_epochs)
    train, val, _ = dataloaders(cfg.distill_batch, mixup=True)

    best = 0
    stall = 0
    pth = cfg.ckpt_dir / "mobilenetv4_distilled.pth"
    for ep in range(1, cfg.distill_epochs + 1):
        student.train()
        for imgs, targs in tqdm(train, desc=f"Distill {ep}"):
            imgs, targs = imgs.to(cfg.device), targs.to(cfg.device)
            with torch.no_grad(): t_logits = teacher(imgs)
            opt.zero_grad()
            s_logits = student(imgs)
            if isinstance(targs, tuple):
                loss = distill_loss(s_logits, t_logits, targs[0])
            else:
                loss = distill_loss(s_logits, t_logits, targs)
            loss.backward(); opt.step(); sched.step()

        _, val_acc = evaluate(student, val, nn.CrossEntropyLoss())
        print(f"  val_acc: {val_acc:.2f}")
        if val_acc > best:
            best = val_acc; save(student, pth, epoch=ep, val_acc=val_acc); stall = 0
        else:
            stall += 1
            if stall >= cfg.patience: break
    print(f"  best: {best:.2f}")
    return pth

def stage_quantize():
    print("Stage 3: Quantization")
    model = mobilenetv4().to("cpu").eval()
    src = cfg.ckpt_dir / "mobilenetv4_distilled.pth"
    if not src.exists(): src = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    if not src.exists(): return print("  no checkpoint found")
    model.load_state_dict(torch.load(src, map_location="cpu")["state_dict"])
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(model, int8_weight_only())
        save(model, cfg.ckpt_dir / "mobilenetv4_quantized.pth")
    except ImportError:
        print("  pip install torchao for quantization")

def stage_export():
    print("Stage 4: Export")
    model = mobilenetv4().to("cpu").eval()
    src = cfg.ckpt_dir / "mobilenetv4_quantized.pth"
    if not src.exists(): src = cfg.ckpt_dir / "mobilenetv4_distilled.pth"
    if not src.exists(): src = cfg.ckpt_dir / "mobilenetv4_finetuned.pth"
    if not src.exists(): return print("  no checkpoint found")
    model.load_state_dict(torch.load(src, map_location="cpu")["state_dict"])

    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(cfg.export_dir / "mobilenetv4.pt"))
    print(f"  exported {cfg.export_dir / 'mobilenetv4.pt'}")
    try:
        ep = torch.export.export(model, (dummy,))
        ep.save(str(cfg.export_dir / "mobilenetv4.ep"))
        print(f"  exported {cfg.export_dir / 'mobilenetv4.ep'} (for ExecuTorch)")
    except: 
        pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all","finetune","distill","quantize","export"])
    args = p.parse_args()
    stages = {"finetune": stage_finetune, "distill": stage_distill,
              "quantize": stage_quantize, "export": stage_export}
    if args.stage == "all":
        stage_finetune()
        stage_distill()
        stage_quantize()
        stage_export()
    else:
        stages[args.stage]()
