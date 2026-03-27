"""
Hyperparameter sweep utilities.

Runs a grid search over the specified hyperparameter space and returns
a summary DataFrame sorted by best validation accuracy.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from src.data_loader import get_dataloaders
from src.models import get_model, count_parameters
from src.train import train


@dataclass
class HparamGrid:
    """
    Defines the hyperparameter search grid.

    Each attribute is a list of values to sweep over; the Cartesian product
    of all lists constitutes the full search space.
    """

    image_size: List[int] = field(default_factory=lambda: [64, 128])
    batch_size: List[int] = field(default_factory=lambda: [32, 64])
    learning_rate: List[float] = field(default_factory=lambda: [1e-3, 1e-4])
    optimizer_name: List[str] = field(default_factory=lambda: ["adam"])
    weight_decay: List[float] = field(default_factory=lambda: [1e-4])
    dropout: List[float] = field(default_factory=lambda: [0.3, 0.5])
    scheduler_name: List[str] = field(default_factory=lambda: ["cosine"])
    architecture: List[str] = field(default_factory=lambda: ["simple", "medium", "deep"])
    num_epochs: List[int] = field(default_factory=lambda: [10])


def _grid_iter(grid: HparamGrid):
    """Yield individual hyperparameter configurations from *grid*."""
    keys = [f.name for f in grid.__dataclass_fields__.values()]
    values = [getattr(grid, k) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_sweep(
    data_root: str,
    grid: Optional[HparamGrid] = None,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run a full grid-search sweep and return a summary DataFrame.

    Args:
        data_root:   Path to the dataset root (contains ``train/`` and
                     ``validate/``).
        grid:        Hyperparameter grid to sweep.  Defaults to a small
                     demonstration grid if *None*.
        num_workers: DataLoader worker processes.
        device:      Computation device; auto-detected if *None*.
        verbose:     Forward verbosity to the training loop.

    Returns:
        :class:`~pandas.DataFrame` with one row per configuration, sorted
        descending by ``best_val_acc``.
    """
    if grid is None:
        grid = HparamGrid()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = list(_grid_iter(grid))
    print(f"Starting sweep: {len(configs)} configurations on {device}")

    records: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs, 1):
        print(
            f"\n[{idx}/{len(configs)}] "
            + ", ".join(f"{k}={v}" for k, v in cfg.items())
        )

        train_loader, val_loader, _ = get_dataloaders(
            data_root=data_root,
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            augment=True,
        )

        model = get_model(
            architecture=cfg["architecture"],
            num_classes=10,
            input_size=cfg["image_size"],
            dropout=cfg["dropout"],
        )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=cfg["num_epochs"],
            learning_rate=cfg["learning_rate"],
            optimizer_name=cfg["optimizer_name"],
            weight_decay=cfg["weight_decay"],
            scheduler_name=cfg["scheduler_name"],
            device=device,
            verbose=verbose,
        )

        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        final_train_acc = history["train_acc"][-1]

        record = copy.copy(cfg)
        record["best_val_acc"] = round(best_val_acc, 2)
        record["best_epoch"] = best_epoch
        record["final_train_acc"] = round(final_train_acc, 2)
        record["n_params"] = count_parameters(model)
        records.append(record)

        print(
            f"  → best_val_acc={best_val_acc:.2f}%  "
            f"(epoch {best_epoch})  "
            f"params={record['n_params']:,}"
        )

    df = pd.DataFrame(records).sort_values("best_val_acc", ascending=False)
    df = df.reset_index(drop=True)
    return df
