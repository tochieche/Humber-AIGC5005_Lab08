"""The bike-count neural network, made from two hidden layers."""

from torch import nn


class BikeNet(nn.Module):
    """Turn 8 input numbers into ONE predicted bike-rental count."""

    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # Like a gate: negatives turn into zero.
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # No sigmoid here: rental counts are NOT probabilities.
        )

    def forward(self, inputs):
        return self.layers(inputs)
