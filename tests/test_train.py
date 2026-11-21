"""Make sure the network really learns, not merely runs without an error."""

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from prepare import make_loaders
from train import run_epoch, set_seed, train_many


def test_one_step_changes_every_layer(sample):
    set_seed()
    net = BikeNet(sample.input_size)
    layers = (net.layers[0], net.layers[2], net.layers[4])
    before = [layer.weight.detach().clone() for layer in layers]
    loader, _, _ = make_loaders(sample, batch_size=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    run_epoch(net, loader, torch.nn.MSELoss(), optimizer)
    for old, layer in zip(before, layers, strict=True):
        assert not torch.equal(old, layer.weight)


def test_network_can_memorize_one_small_batch():
    set_seed()
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    y = 2 * x + 1
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    net = BikeNet(1, hidden_size=16)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
    start = run_epoch(net, loader, loss_fn)["loss"]
    for _ in range(350):
        run_epoch(net, loader, loss_fn, optimizer)
    end = run_epoch(net, loader, loss_fn)["loss"]
    assert end < start * 0.01


def test_validation_does_not_change_weights(sample):
    set_seed()
    net = BikeNet(sample.input_size)
    before = copy.deepcopy(net.state_dict())
    _, val_loader, _ = make_loaders(sample)
    run_epoch(net, val_loader, torch.nn.MSELoss())
    assert all(
        torch.equal(net.state_dict()[key], value) for key, value in before.items()
    )


def test_training_history_and_best_weights(sample):
    set_seed()
    net = BikeNet(sample.input_size)
    train_loader, val_loader, _ = make_loaders(sample)
    history, best_epoch = train_many(net, train_loader, val_loader, epochs=6)
    assert len(history) == 6
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_mae"}
    assert best_epoch == min(history, key=lambda row: row["val_loss"])["epoch"]
    actual = run_epoch(net, val_loader, torch.nn.MSELoss())["loss"]
    assert abs(actual - history[best_epoch - 1]["val_loss"]) < 1e-4
