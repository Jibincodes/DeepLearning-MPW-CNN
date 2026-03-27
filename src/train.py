"""
Training loop and utilities for CNN experiments.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core training / evaluation helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (average_loss, accuracy_percent)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="  Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on *loader*.

    Returns:
        (average_loss, accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="  Val  ", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    scheduler_name: str = "step",
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train *model* for *num_epochs* and return history metrics.

    Args:
        model:          The neural network to train.
        train_loader:   DataLoader for the training set.
        val_loader:     DataLoader for the validation / test set.
        num_epochs:     Total number of epochs.
        learning_rate:  Initial learning rate.
        optimizer_name: ``'adam'``, ``'rmsprop'``, or ``'sgd'``.
        weight_decay:   L2 regularisation coefficient.
        scheduler_name: ``'step'``, ``'cosine'``, or ``'none'``.
        device:         Computation device; auto-detected if *None*.
        verbose:        Print per-epoch summary when *True*.

    Returns:
        Dictionary with keys ``'train_loss'``, ``'train_acc'``,
        ``'val_loss'``, ``'val_acc'`` each mapping to a list of values
        (one per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            "Choose 'adam', 'rmsprop', or 'sgd'."
        )

    # LR scheduler
    if scheduler_name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    elif scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        scheduler = None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:03d}/{num_epochs}]  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
                f"({elapsed:.1f}s)"
            )

    if verbose:
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    return history
