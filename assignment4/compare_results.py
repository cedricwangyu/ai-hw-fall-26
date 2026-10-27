"""Compare test metrics from the baseline and augmented experiments."""

import json
from pathlib import Path


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    baseline = load_results(Path("outputs/baseline/test_results.json"))
    augmented = load_results(Path("outputs/augmented/test_results.json"))
    improvement = 100 * (augmented["test_accuracy"] - baseline["test_accuracy"])
    rows = [
        "# Test Results",
        "",
        "Both models were selected using a validation subset of the MNIST training split. "
        "The official test split was used once for final evaluation.",
        "",
        "| Experiment | Augmentation | Test accuracy | Test loss |",
        "|---|---:|---:|---:|",
        f"| Baseline | No | {100 * baseline['test_accuracy']:.2f}% | {baseline['test_loss']:.4f} |",
        f"| Augmented CNN | Yes | {100 * augmented['test_accuracy']:.2f}% | {augmented['test_loss']:.4f} |",
        "",
        f"Augmentation changed test accuracy by **{improvement:+.2f} percentage points**.",
        "",
        "Generated from the committed JSON result files by `compare_results.py`.",
    ]
    Path("RESULTS.md").write_text("\n".join(rows) + "\n")
    print("Wrote RESULTS.md")


if __name__ == "__main__":
    main()