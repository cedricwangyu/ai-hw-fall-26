import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_cnn.engine import run_epoch
from mnist_cnn.model import MnistCNN


def test_model_output_shape() -> None:
    model = MnistCNN()
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_training_epoch_updates_model() -> None:
    torch.manual_seed(0)
    model = MnistCNN()
    loader = DataLoader(
        TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8) % 10), batch_size=4
    )
    before = model.classifier[-1].weight.detach().clone()
    metrics = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        torch.optim.SGD(model.parameters(), lr=0.01),
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert not torch.equal(before, model.classifier[-1].weight)