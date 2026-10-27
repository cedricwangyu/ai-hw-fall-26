# MNIST Image Recognition with a CNN

This project trains and evaluates a small convolutional neural network on MNIST.
It keeps the official splits separate: model fitting and checkpoint selection use
only the 60,000-image training split, while final metrics use the untouched
10,000-image test split.

## Model

The network contains four 3 by 3 convolution layers in two blocks, batch
normalization, ReLU activations, max pooling, dropout, and a two-layer classifier.
The bonus experiment applies small affine and perspective transformations to
training images only and compares the result with an unaugmented baseline.

## Setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Train and test the baseline:

```bash
python train.py --no-augmentation --output-dir outputs/baseline
python test.py --checkpoint outputs/baseline/best_model.pt --output-dir outputs/baseline
```

Train and test with augmentation:

```bash
python train.py --output-dir outputs/augmented
python test.py --checkpoint outputs/augmented/best_model.pt --output-dir outputs/augmented
```

Generate the comparison report and bonus PDF presentation:

```bash
python compare_results.py
python make_presentation.py
```

Run automated smoke tests:

```bash
python -m pytest -q
```

On Apple Silicon, `--device auto` selects MPS. CUDA is selected when available,
and CPU is the fallback. Use `--device cpu` to override this behavior.

## Repository contents

- `train.py`: training/validation loop and best-checkpoint selection
- `test.py`: test-only evaluation, per-class metrics, confusion matrix, mistakes
- `mnist_cnn/`: model, data pipeline, engine, and utility modules
- `outputs/`: JSON metrics and generated result figures
- `RESULTS.md`: compact comparison generated from JSON metrics
- `presentation.pdf`: seven-slide presentation generated from experiment artifacts

Model checkpoints and downloaded MNIST files are intentionally excluded from Git
because they can be reproduced with the commands above.