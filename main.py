"""
Main experiment runner for CNN depth and hyperparameter studies.

Usage
-----
Basic architecture comparison (64×64 images, 10 epochs):
    python main.py --data_root /path/to/icosimal_img_class_03/data_uniform_224_224_sets

Full study with larger images and more epochs:
    python main.py --data_root /path/to/... --image_size 128 --num_epochs 30

Hyperparameter sweep (small grid, quick):
    python main.py --data_root /path/to/... --sweep

Save results and figures to a custom directory:
    python main.py --data_root /path/to/... --output_dir results/
"""

import argparse
import os
from pathlib import Path

import torch

from src.data_loader import get_dataloaders
from src.evaluate import (
    compare_architectures,
    plot_confusion_matrix,
    plot_training_history,
    predict,
    print_classification_report,
)
from src.hyperparameter_tuning import HparamGrid, run_sweep
from src.models import count_parameters, get_model
from src.train import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_architecture_comparison(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Train SimpleCNN, MediumCNN, and DeepCNN and compare their results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — Architecture Depth Comparison")
    print("=" * 70)

    train_loader, val_loader, class_names = get_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
    )

    arch_results = {}

    for arch in ("simple", "medium", "deep"):
        print(f"\n--- Training {arch.upper()} CNN ---")
        model = get_model(
            architecture=arch,
            num_classes=len(class_names),
            input_size=image_size,
        )
        print(f"  Trainable parameters: {count_parameters(model):,}")

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            device=device,
            verbose=True,
        )
        arch_results[arch] = history

        # Per-architecture plots
        fig = plot_training_history(history, title=f"{arch.capitalize()} CNN")
        fig.savefig(output_dir / f"history_{arch}.png", dpi=150)
        fig.clf()

        # Confusion matrix for the best architecture
        y_true, y_pred = predict(model, val_loader, device)
        print_classification_report(y_true, y_pred, class_names)
        cm_fig = plot_confusion_matrix(
            y_true, y_pred, class_names, title=f"Confusion Matrix — {arch.capitalize()} CNN"
        )
        cm_fig.savefig(output_dir / f"confusion_{arch}.png", dpi=150)
        cm_fig.clf()

    # Overlay comparison
    comp_fig = compare_architectures(arch_results, metric="val_acc")
    comp_fig.savefig(output_dir / "comparison_val_acc.png", dpi=150)
    comp_fig.clf()

    comp_fig = compare_architectures(arch_results, metric="val_loss")
    comp_fig.savefig(output_dir / "comparison_val_loss.png", dpi=150)
    comp_fig.clf()

    print(f"\nArchitecture comparison figures saved to: {output_dir}")


def run_hyperparameter_study(
    data_root: str,
    output_dir: Path,
    device: torch.device,
    quick: bool = True,
) -> None:
    """Run a grid sweep and save results to CSV."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Hyperparameter Tuning Sweep")
    print("=" * 70)

    if quick:
        # Small grid suitable for demonstration / CI
        grid = HparamGrid(
            image_size=[64],
            batch_size=[32],
            learning_rate=[1e-3, 1e-4],
            optimizer_name=["adam", "sgd"],
            weight_decay=[1e-4],
            dropout=[0.3, 0.5],
            scheduler_name=["cosine"],
            architecture=["simple", "medium", "deep"],
            num_epochs=[5],
        )
    else:
        # Full grid
        grid = HparamGrid(
            image_size=[64, 128],
            batch_size=[32, 64],
            learning_rate=[1e-3, 3e-4, 1e-4],
            optimizer_name=["adam", "sgd"],
            weight_decay=[1e-4],
            dropout=[0.3, 0.5],
            scheduler_name=["cosine", "step"],
            architecture=["simple", "medium", "deep"],
            num_epochs=[15],
        )

    df = run_sweep(
        data_root=data_root,
        grid=grid,
        device=device,
        verbose=False,
    )

    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved to: {csv_path}")
    print("\nTop 10 configurations:")
    print(df.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN experiments on the iCoSimal V3 animal dataset."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the dataset root that contains 'train/' and 'validate/' "
            "subdirectories, e.g. icosimal_img_class_03/data_uniform_224_224_sets"
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize images to image_size×image_size (default: 64)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer (default: adam)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the hyperparameter sweep (Experiment 2)",
    )
    parser.add_argument(
        "--sweep_full",
        action="store_true",
        help="Run the full (larger) hyperparameter grid instead of the quick one",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory for saving figures and CSVs (default: results/)",
    )
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="Skip the architecture comparison experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.no_comparison:
        run_architecture_comparison(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            output_dir=output_dir,
            device=device,
        )

    if args.sweep or args.sweep_full:
        run_hyperparameter_study(
            data_root=args.data_root,
            output_dir=output_dir,
            device=device,
            quick=not args.sweep_full,
        )


if __name__ == "__main__":
    main()
