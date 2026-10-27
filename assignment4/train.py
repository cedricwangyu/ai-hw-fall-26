"""Train the CNN using only the MNIST training split."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from mnist_cnn.data import make_loaders
from mnist_cnn.engine import run_epoch
from mnist_cnn.model import MnistCNN
from mnist_cnn.utils import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/augmented"))
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    return parser.parse_args()


def save_history(history: list, output_dir: Path) -> None:
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(
            epochs, [100 * row[f"{split}_accuracy"] for row in history], label=split
        )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(
        args.data_dir,
        args.batch_size,
        args.validation_size,
        args.seed,
        not args.no_augmentation,
        args.workers,
        args.limit_train,
    )
    model = MnistCNN(args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_accuracy = -1.0
    print(f"Training on {device} with {len(train_loader.dataset)} examples")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train={100 * train_metrics['accuracy']:.2f}% "
            f"validation={100 * validation_metrics['accuracy']:.2f}%"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                    "augmentation": not args.no_augmentation,
                    "seed": args.seed,
                },
                args.output_dir / "best_model.pt",
            )

    save_history(history, args.output_dir)
    print(f"Best checkpoint saved to {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()