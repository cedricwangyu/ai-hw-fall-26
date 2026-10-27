# Test Results

Both models were selected using a validation subset of the MNIST training split. The official test split was used once for final evaluation.

| Experiment | Augmentation | Test accuracy | Test loss |
|---|---:|---:|---:|
| Baseline | No | 99.50% | 0.0154 |
| Augmented CNN | Yes | 99.54% | 0.0114 |

Augmentation changed test accuracy by **+0.04 percentage points**.

Generated from the committed JSON result files by `compare_results.py`.
