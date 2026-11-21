"""Test different neural-network widths."""

import pytest
import torch

from model import BikeNet


@pytest.mark.parametrize("hidden_size", [8, 16, 32, 128])
def test_network_output_shape(hidden_size):
    net = BikeNet(input_size=8, hidden_size=hidden_size)
    assert net(torch.randn(7, 8)).shape == (7, 1)


def test_network_has_at_least_two_relu_hidden_layers():
    net = BikeNet(8)
    assert sum(isinstance(x, torch.nn.ReLU) for x in net.modules()) >= 2
    assert isinstance(list(net.layers)[-1], torch.nn.Linear)
