"""Generate a short PDF presentation from the saved experiment artifacts."""

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"


def add_title(figure: plt.Figure, title: str, subtitle: str = "") -> None:
    figure.text(0.07, 0.9, title, fontsize=27, fontweight="bold", color="#17324d")
    if subtitle:
        figure.text(0.07, 0.84, subtitle, fontsize=13, color="#536878")


def image_slide(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="#f7f4ed")
    add_title(figure, title, caption)
    axis = figure.add_axes((0.08, 0.08, 0.84, 0.7))
    axis.imshow(mpimg.imread(image_path))
    axis.axis("off")
    pdf.savefig(figure, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    baseline = json.loads((OUTPUTS / "baseline/test_results.json").read_text())
    augmented = json.loads((OUTPUTS / "augmented/test_results.json").read_text())
    presentation_path = ROOT / "presentation.pdf"

    with PdfPages(presentation_path) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27), facecolor="#f7f4ed")
        figure.text(0.07, 0.66, "MNIST Image Recognition", fontsize=34, fontweight="bold", color="#17324d")
        figure.text(0.07, 0.56, "A compact convolutional neural network", fontsize=20, color="#d1495b")
        figure.text(0.07, 0.18, "Training, augmentation, and error analysis", fontsize=14, color="#536878")
        pdf.savefig(figure, facecolor=figure.get_facecolor())
        plt.close(figure)

        figure = plt.figure(figsize=(11.69, 8.27), facecolor="#f7f4ed")
        add_title(figure, "Experimental design", "Strict separation between model development and final testing")
        bullets = [
            "Data: 60,000 MNIST training images and 10,000 test images",
            "Training split: 55,000 train + 5,000 validation",
            "Architecture: four convolution layers, batch normalization, dropout",
            "Optimization: AdamW, cosine learning-rate schedule, cross-entropy loss",
            "Bonus experiment: random affine and perspective augmentation on train only",
        ]
        for index, text in enumerate(bullets):
            figure.text(0.1, 0.7 - index * 0.105, f"{index + 1}.  {text}", fontsize=16, color="#243b53")
        pdf.savefig(figure, facecolor=figure.get_facecolor())
        plt.close(figure)

        image_slide(
            pdf,
            "Learning behavior",
            OUTPUTS / "augmented/training_curves.png",
            "Training and validation metrics for the augmented CNN",
        )

        figure = plt.figure(figsize=(11.69, 8.27), facecolor="#f7f4ed")
        add_title(figure, "Test-set comparison", "One final evaluation on the untouched official test split")
        names = ["Baseline", "Augmented"]
        values = [100 * baseline["test_accuracy"], 100 * augmented["test_accuracy"]]
        axis = figure.add_axes((0.14, 0.17, 0.72, 0.58))
        bars = axis.bar(names, values, color=["#5b8e7d", "#d1495b"], width=0.55)
        lower = max(0.0, min(values) - 0.5)
        axis.set_ylim(lower, 100.0)
        axis.set_ylabel("Test accuracy (%)")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}%", ha="center", fontsize=16)
        pdf.savefig(figure, facecolor=figure.get_facecolor())
        plt.close(figure)

        image_slide(
            pdf,
            "Where the model is confused",
            OUTPUTS / "augmented/confusion_matrix.png",
            "Rows are true classes; columns are predictions",
        )
        image_slide(
            pdf,
            "Error analysis",
            OUTPUTS / "augmented/misclassified_examples.png",
            "Ambiguous handwriting accounts for many remaining mistakes",
        )

        figure = plt.figure(figsize=(11.69, 8.27), facecolor="#f7f4ed")
        add_title(figure, "Conclusions")
        delta = 100 * (augmented["test_accuracy"] - baseline["test_accuracy"])
        conclusions = [
            f"The augmented CNN reached {100 * augmented['test_accuracy']:.2f}% test accuracy.",
            f"Augmentation changed accuracy by {delta:+.2f} percentage points.",
            "The test split was never used for training or checkpoint selection.",
            "Saved JSON metrics and figures make the result reproducible and auditable.",
        ]
        for index, text in enumerate(conclusions):
            figure.text(0.1, 0.68 - index * 0.12, f"-  {text}", fontsize=17, color="#243b53")
        pdf.savefig(figure, facecolor=figure.get_facecolor())
        plt.close(figure)

    print(f"Wrote {presentation_path}")


if __name__ == "__main__":
    main()