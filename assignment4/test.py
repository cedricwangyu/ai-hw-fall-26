"""Evaluate a trained checkpoint on the untouched MNIST test split."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from mnist_cnn.data import MNIST_MEAN, MNIST_STD, make_test_loader
from mnist_cnn.engine import predict, run_epoch
from mnist_cnn.model import MnistCNN
from mnist_cnn.utils import load_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/augmented/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/augmented"))
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)


def save_figures(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_dir: Path,
) -> torch.Tensor:
    matrix = confusion_matrix(labels, predictions)
    figure, axis = plt.subplots(figsize=(7, 6))
    plot = axis.imshow(matrix.numpy(), cmap="Blues")
    for row in range(10):
        for column in range(10):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=7)
    axis.set(xlabel="Predicted label", ylabel="True label", title="MNIST confusion matrix")
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    mistakes = torch.where(labels != predictions)[0][:25]
    figure, axes = plt.subplots(5, 5, figsize=(8, 8))
    for axis, index in zip(axes.flat, mistakes):
        image = images[index] * MNIST_STD[0] + MNIST_MEAN[0]
        axis.imshow(image.squeeze().clamp(0, 1), cmap="gray")
        axis.set_title(f"true {labels[index]} / pred {predictions[index]}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(mistakes) :]:
        axis.axis("off")
    figure.suptitle("Misclassified test images")
    figure.tight_layout()
    figure.savefig(output_dir / "misclassified_examples.png", dpi=180)
    plt.close(figure)
    return matrix


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = MnistCNN(checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = make_test_loader(
        args.data_dir, args.batch_size, args.workers, args.limit_test
    )
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    images, labels, predictions = predict(model, loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = save_figures(images, labels, predictions, args.output_dir)
    per_class = matrix.diag() / matrix.sum(dim=1)
    results = {
        "test_examples": len(loader.dataset),
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "per_class_accuracy": {str(i): value.item() for i, value in enumerate(per_class)},
        "checkpoint_epoch": checkpoint["epoch"],
        "augmentation": checkpoint["augmentation"],
    }
    (args.output_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {100 * metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()