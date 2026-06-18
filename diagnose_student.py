"""Diagnose student training — focused test without mixup, lower LR."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm
import timm

num_classes = 8
image_size = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
data_dir = "data/processed/classification_data"
seed = 42

def build_tf(is_train):
    if is_train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(15, fill=128),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return v2.Compose([
        v2.ToImage(), v2.Resize(256), v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_loaders(batch_size):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", build_tf(True))
    val_ds = datasets.ImageFolder(f"{data_dir}/val", build_tf(False))
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train, val


def make_student():
    return timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=num_classes).to(device)


def accuracy(out, target):
    if target.ndim == 2:
        target = target.argmax(dim=1)
    _, pred = out.topk(1, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.float().sum().item()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_acc, n = 0.0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        total_acc += accuracy(out, lbls)
        n += imgs.size(0)
    return total_acc / n


def train_epoch(model, loader, crit, opt):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for imgs, lbls in tqdm(loader, desc="train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = crit(out, lbls)
        loss.backward()
        opt.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out, lbls)
        n += bs
    return total_loss / n, total_acc / n


def run_probe(model, train_loader, val_loader, epochs, lr, wd):
    print("\n--- PROBE (head only) ---")
    for name, p in model.named_parameters():
        if "head" not in name and "classifier" not in name:
            p.requires_grad_(False)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, crit, opt)
        val_acc = evaluate(model, val_loader)
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")


def run_full(model, train_loader, val_loader, epochs, lr, wd):
    print("\n--- FULL FINE-TUNE (all params, no mixup) ---")
    for p in model.parameters():
        p.requires_grad_(True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, stall = 0.0, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, crit, opt)
        val_acc = evaluate(model, val_loader)
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            stall = 0
            torch.save(model.state_dict(), "checkpoints/student_test_best.pth")
        else:
            stall += 1
            if stall >= 15:
                print(f"  Early stop at epoch {epoch}")
                break
    return best_acc


# Run
torch.manual_seed(seed)
student = make_student()
train_probe, val_probe = make_loaders(64)
run_probe(student, train_probe, val_probe, 5, 1e-4, 1e-4)

train_full, val_full = make_loaders(64)
best = run_full(student, train_full, val_full, 30, 1e-4, 1e-4)
print(f"\nBest val_acc: {best:.4f}")
