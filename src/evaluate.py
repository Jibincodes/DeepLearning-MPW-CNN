"""
Evaluation utilities: per-class metrics, confusion matrix, and plots.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on *loader* and collect ground-truth / predictions.

    Returns:
        (y_true, y_pred) as NumPy integer arrays.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    for inputs, targets in tqdm(loader, desc="Predicting"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_targets.extend(targets.numpy().tolist())
        all_preds.extend(predicted.cpu().numpy().tolist())

    return np.array(all_targets), np.array(all_preds)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print a per-class precision / recall / F1 table."""
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot and return a normalised confusion matrix.

    The matrix is normalised row-wise so that each cell shows the fraction
    of samples in the true class that were predicted as the column class.
    """
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), row_sums, out=cm_norm, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot loss and accuracy curves from a *history* dict produced by
    :func:`src.train.train`.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(True)

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], label="Train acc")
    ax_acc.plot(epochs, history["val_acc"], label="Val acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.grid(True)

    plt.tight_layout()
    return fig


def compare_architectures(
    results: dict,
    metric: str = "val_acc",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Overlay *metric* curves for multiple architecture runs.

    Args:
        results: ``{label: history_dict}`` mapping.
        metric:  Key in each history dict to plot.
        figsize: Figure dimensions.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, history in results.items():
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], marker="o", markersize=3, label=label)

    metric_label = metric.replace("_", " ").title()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Architecture Comparison — {metric_label}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
