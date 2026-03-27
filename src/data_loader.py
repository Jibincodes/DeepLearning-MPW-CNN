"""
Data loading utilities for the iCoSimal V3 dataset.

Expected dataset directory structure:
    icosimal_img_class_03/data_uniform_224_224_sets/
    |-- train/
    |   |-- cat/
    |   |-- chicken/
    |   |-- ...
    |-- validate/
        |-- cat/
        |-- chicken/
        |-- ...
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = [
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "rabbit",
    "sheep",
    "squirrel",
    "zebra",
]

NUM_CLASSES = len(CLASSES)


def get_transforms(image_size: int = 224, augment: bool = True):
    """
    Return train and validation transforms for the given image size.

    Args:
        image_size: Target size for resizing images (square).
        augment:    Whether to apply data augmentation to the training set.

    Returns:
        A tuple (train_transform, val_transform).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def get_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
):
    """
    Create DataLoader objects for training and validation splits.

    Args:
        data_root:   Path to the dataset root, i.e. the directory that
                     contains ``train/`` and ``validate/`` subdirectories.
        image_size:  Target side length (pixels) for resizing images.
        batch_size:  Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        augment:     Whether to apply data augmentation to the training set.

    Returns:
        A tuple (train_loader, val_loader, class_names).
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "validate"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Please download the iCoSimal V3 dataset and set data_root to the "
            "directory containing 'train/' and 'validate/' subdirectories."
        )
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Please download the iCoSimal V3 dataset and set data_root to the "
            "directory containing 'train/' and 'validate/' subdirectories."
        )

    train_transform, val_transform = get_transforms(
        image_size=image_size, augment=augment
    )

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(
        f"Dataset sizes  — train: {len(train_dataset)}, "
        f"validation: {len(val_dataset)}"
    )

    return train_loader, val_loader, class_names
