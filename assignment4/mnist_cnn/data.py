"""MNIST datasets and loaders with train-only augmentation."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_transform(augment: bool) -> transforms.Compose:
    steps = []
    if augment:
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)
                ),
                transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    return transforms.Compose(steps)


def _limited(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def make_loaders(
    data_dir: Path,
    batch_size: int,
    validation_size: int,
    seed: int,
    augment: bool,
    workers: int,
    limit_train: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    plain_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=build_transform(False)
    )
    augmented_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=build_transform(augment)
    )

    effective_size = min(len(plain_dataset), limit_train or len(plain_dataset))
    if validation_size >= effective_size:
        raise ValueError("validation_size must be smaller than the training set")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(plain_dataset), generator=generator)[:effective_size]
    validation_indices = indices[:validation_size].tolist()
    train_indices = indices[validation_size:].tolist()

    train_loader = DataLoader(
        Subset(augmented_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(plain_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def make_test_loader(
    data_dir: Path,
    batch_size: int,
    workers: int,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=build_transform(False)
    )
    dataset = _limited(dataset, limit_test, seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )