"""
Reporting and visualization utilities for training analysis.

Generates:
  - Training progress curves (probe + full fine-tune per stage)
  - Confusion matrix heatmap
  - Confidence analysis (per-class precision/recall, calibration, distribution)
  - Multi-model benchmark comparison charts
"""

import json
from pathlib import Path
from typing import Any, List, Literal, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _savefig(path: Path, dpi: int = 150) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def plot_training_curves(
    history: dict,
    output_path: Path,
    title: str = "Training Progress",
) -> None:
    """Plot training curves from a history dict.

    Args:
        history: dict with keys ``probe`` and/or ``full``.
            ``probe``: {"epochs": [...], "val_acc": [...]}
            ``full``:  {"epochs": [...], "train_loss": [...],
                        "train_acc": [...], "val_loss": [...],
                        "val_acc": [...]}
        output_path: where to save the PNG.
        title: plot title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe = history.get("probe")
    full = history.get("full")

    n_subplots = 0
    if probe is not None:
        n_subplots += 1
    if full is not None:
        n_subplots += 2  # loss + accuracy

    if n_subplots == 0:
        print("  [WARN] No training history to plot.")
        return

    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4), squeeze=False)
    axes = axes[0]
    ax_idx = 0

    if probe is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        epochs = probe["epochs"]
        ax.plot(epochs, probe["val_acc"], "o-", color="#2ecc71", label="Probe val_acc")
        ax.set_title("Probe Phase")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    if full is not None:
        epochs = full["epochs"]
        # Loss subplot
        ax = axes[ax_idx]
        ax_idx += 1
        if "train_loss" in full:
            ax.plot(
                epochs, full["train_loss"], "-", color="#3498db", label="Train loss"
            )
        if "val_loss" in full:
            ax.plot(epochs, full["val_loss"], "-", color="#e74c3c", label="Val loss")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Accuracy subplot
        ax = axes[ax_idx]
        ax_idx += 1
        if "train_acc" in full:
            ax.plot(epochs, full["train_acc"], "-", color="#3498db", label="Train acc")
        if "val_acc" in full:
            ax.plot(epochs, full["val_acc"], "-", color="#e74c3c", label="Val acc")
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _savefig(output_path)


def save_history(history: dict, path: Path) -> None:
    """Save training history as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: Literal["true", "pred", "all"] | None = "true",
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: ground-truth labels (N,).
        y_pred: predicted labels (N,).
        class_names: list of class names.
        output_path: PNG save path.
        normalize: 'true' (rows sum to 1), 'pred' (cols sum to 1),
            'all' (whole matrix sums to 1), or None.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.6))
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Fraction" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fig.tight_layout()
    _savefig(output_path)


def plot_confidence_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate a 2×2 grid of confidence-related plots.

    Subplots:
      1. Per-class precision / recall / F1 bar chart.
      2. Confidence histogram (correct vs incorrect predictions).
      3. Calibration curve (reliability diagram).
      4. Per-class average confidence when correct vs incorrect.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    max_conf = probs.max(axis=1)
    correct_mask = y_pred == y_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Confidence & Calibration Analysis", fontsize=13, fontweight="bold")

    # ---- 1. Per-class precision / recall / F1 ----
    ax = axes[0, 0]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        zero_division=cast(Any, 0),
    )
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax.bar(x, recall, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ---- 2. Confidence histogram ----
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(
        max_conf[correct_mask],
        bins=bins,
        alpha=0.6,
        label="Correct",
        color="#2ecc71",
        edgecolor="white",
    )
    ax.hist(
        max_conf[~correct_mask],
        bins=bins,
        alpha=0.6,
        label="Incorrect",
        color="#e74c3c",
        edgecolor="white",
    )
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ---- 3. Calibration curve ----
    ax = axes[1, 0]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    bin_accs = []
    bin_confs = []
    bin_counts = []
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (max_conf > bl) & (max_conf <= bu)
        if in_bin.sum() == 0:
            continue
        acc = correct_mask[in_bin].mean()
        conf = max_conf[in_bin].mean()
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(in_bin.sum())

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "o-", color="#9b59b6", label="Model")
        ax.bar(
            bin_confs,
            bin_accs,
            width=0.08,
            alpha=0.3,
            color="#9b59b6",
            edgecolor="white",
        )
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 4. Per-class avg confidence (correct vs incorrect) ----
    ax = axes[1, 1]
    conf_correct = []
    conf_incorrect = []
    for c in range(len(class_names)):
        mask_c = y_true == c
        conf_c = max_conf[mask_c]
        corr_c = correct_mask[mask_c]
        conf_correct.append(conf_c[corr_c].mean() if corr_c.any() else 0)
        conf_incorrect.append(conf_c[~corr_c].mean() if (~corr_c).any() else 0)

    x = np.arange(len(class_names))
    width = 0.35
    ax.bar(x - width / 2, conf_correct, width, label="Correct", color="#2ecc71")
    ax.bar(x + width / 2, conf_incorrect, width, label="Incorrect", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg Confidence")
    ax.set_title("Per-Class Avg Confidence")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(output_path)


def plot_benchmark_comparison(
    results: List[dict],
    output_path: Path,
) -> None:
    """Plot a multi-model benchmark comparison chart.

    Args:
        results: list of dicts, each with keys:
            ``name``, ``params_m``, ``test_acc``, ``size_mb``, ``gflops``.
        output_path: PNG save path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    params = [r["params_m"] for r in results]
    accs = [r["test_acc"] for r in results]
    sizes = [r["size_mb"] for r in results]
    gflops = [r.get("gflops", 0.0) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Benchmark Comparison", fontsize=13, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    bars_colors = [colors[i % len(colors)] for i in range(len(names))]

    # Params
    ax = axes[0, 0]
    ax.barh(names, params, color=bars_colors)
    ax.set_xlabel("Params (M)")
    ax.set_title("Parameter Count")
    ax.invert_yaxis()
    for i, v in enumerate(params):
        ax.text(v + 0.1, i, f"{v:.1f}M", va="center", fontsize=9)

    # Accuracy
    ax = axes[0, 1]
    ax.barh(names, accs, color=bars_colors)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Test Accuracy")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=9)

    # Model size
    ax = axes[1, 0]
    ax.barh(names, sizes, color=bars_colors)
    ax.set_xlabel("Size (MB)")
    ax.set_title("Checkpoint Size")
    ax.invert_yaxis()
    for i, v in enumerate(sizes):
        ax.text(v + 0.2, i, f"{v:.1f}MB", va="center", fontsize=9)

    # FLOPs
    ax = axes[1, 1]
    ax.barh(names, gflops, color=bars_colors)
    ax.set_xlabel("GFLOPs")
    ax.set_title("Compute (GFLOPs)")
    ax.invert_yaxis()
    for i, v in enumerate(gflops):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(output_path)


def print_benchmark_table(results: List[dict]) -> None:
    """Print a formatted ASCII benchmark comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Model':<40} {'Params(M)':>10} {'TestAcc':>10} {'Size(MB)':>10} {'GFLOPs':>10}"
    print(header)
    print("-" * 80)
    for r in results:
        line = (
            f"{r['name']:<40} "
            f"{r['params_m']:>10.2f} "
            f"{r['test_acc']:>10.4f} "
            f"{r['size_mb']:>10.2f} "
            f"{r.get('gflops', 0.0):>10.3f}"
        )
        print(line)
    print("=" * 80)


@torch.inference_mode()
def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
    mixed_precision: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and collect predictions + softmax probabilities.

    Returns:
        (test_loss, test_acc, y_true, y_pred, probs)
    """
    from src.core import _amp_ctx, accuracy

    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with _amp_ctx(device, mixed_precision):
            outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        total_loss += criterion(outputs, targets).item() * images.size(0)
        total_acc += accuracy(outputs, targets)
        n += images.size(0)

        preds = outputs.argmax(dim=1).cpu().numpy()
        labels = targets.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    if n == 0:
        probs_arr = np.empty((0, 0), dtype=np.float32)
        return 0.0, 0.0, y_true, y_pred, probs_arr
    probs_arr = np.concatenate(all_probs, axis=0)
    return total_loss / n, total_acc / n, y_true, y_pred, probs_arr
