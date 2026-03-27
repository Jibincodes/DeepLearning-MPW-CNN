"""
CNN architectures of progressively increasing depth and complexity.

Three architectures are provided:
    - SimpleCNN   : 2 convolutional layers  (shallow baseline)
    - MediumCNN   : 4 convolutional layers  (intermediate)
    - DeepCNN     : 6 convolutional layers with BatchNorm and Dropout
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Shallow CNN with 2 convolutional layers.

    Designed as a minimal baseline to establish a lower-bound on performance
    and to illustrate the benefit of adding more depth.

    Architecture:
        Conv(3→32, k=3) → ReLU → MaxPool(2)
        Conv(32→64, k=3) → ReLU → MaxPool(2)
        Flatten → FC(64·H·W → 256) → ReLU → Dropout → FC(256 → num_classes)
    """

    def __init__(self, num_classes: int = 10, input_size: int = 64, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flat_size = 64 * (input_size // 4) * (input_size // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MediumCNN(nn.Module):
    """
    Intermediate CNN with 4 convolutional layers.

    Adds two additional convolutional blocks (with increased channel depth)
    over SimpleCNN to demonstrate the benefit of going deeper.

    Architecture:
        Conv(3→32, k=3) → ReLU → MaxPool(2)
        Conv(32→64, k=3) → ReLU → MaxPool(2)
        Conv(64→128, k=3) → ReLU → MaxPool(2)
        Conv(128→256, k=3) → ReLU → MaxPool(2)
        Flatten → FC(256·H·W → 512) → ReLU → Dropout → FC(512 → num_classes)
    """

    def __init__(self, num_classes: int = 10, input_size: int = 64, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flat_size = 256 * (input_size // 16) * (input_size // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """
    Deep CNN with 6 convolutional layers, BatchNorm, and Dropout.

    Extends MediumCNN with BatchNormalization after each convolution for
    better gradient flow and regularisation, and uses Dropout in the
    classifier head.  This architecture aims to show the best classification
    performance of the three.

    Architecture:
        [Conv(3→32, k=3) → BN → ReLU] × 2 → MaxPool(2)
        [Conv(32→64, k=3) → BN → ReLU] × 2 → MaxPool(2)
        [Conv(64→128, k=3) → BN → ReLU] × 2 → MaxPool(2)  (if input ≥ 64)
        AdaptiveAvgPool(4×4)
        Flatten → FC(128·16 → 512) → ReLU → Dropout → FC(512 → num_classes)
    """

    def __init__(self, num_classes: int = 10, input_size: int = 64, dropout: float = 0.5):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def get_model(
    architecture: str,
    num_classes: int = 10,
    input_size: int = 64,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Factory function to instantiate a CNN model by name.

    Args:
        architecture: One of ``'simple'``, ``'medium'``, or ``'deep'``.
        num_classes:  Number of output classes.
        input_size:   Spatial size of the (square) input images.
        dropout:      Dropout probability used in the classifier head.

    Returns:
        The requested :class:`nn.Module`.
    """
    architectures = {
        "simple": SimpleCNN,
        "medium": MediumCNN,
        "deep": DeepCNN,
    }
    if architecture not in architectures:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from {list(architectures.keys())}."
        )
    return architectures[architecture](
        num_classes=num_classes, input_size=input_size, dropout=dropout
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
