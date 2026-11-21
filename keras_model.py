"""Building the same two-hidden-layer bike model using Keras 3"""

import keras


def build_keras_model(input_size, hidden_size=32):
    """The shape and activations match model.BikeNet exactly."""
    return keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(hidden_size, activation="relu"),
            keras.layers.Dense(hidden_size, activation="relu"),
            keras.layers.Dense(1),  # A bike count needs NO output activation.
        ],
        name="bike_demand_keras",
    )


def keras_parameter_count(model):
    """Counting all the trainable numbers in the network."""
    return model.count_params()


def copy_torch_weights(torch_model, keras_model):
    """Copying the SAME learned numbers to prove the networks compute alike.

    PyTorch keeps each matrix as (outputs, inputs), but Keras stores it
    as (inputs, outputs). The .T flips rows and columns to match.
    """
    import torch  # Imported here because this helper needs both libraries.

    torch_layers = [
        layer for layer in torch_model.layers if isinstance(layer, torch.nn.Linear)
    ]
    keras_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]
    if len(torch_layers) != len(keras_layers):
        raise ValueError("The two networks must have the same number of Dense layers")
    for old, new in zip(torch_layers, keras_layers, strict=True):
        weights = old.weight.detach().cpu().numpy().T  # IMPORTANT transpose!
        biases = old.bias.detach().cpu().numpy()
        if new.kernel.shape != weights.shape:
            raise ValueError("The layer sizes differ; weight copy would be invalid")
        new.set_weights([weights.copy(), biases.copy()])
    return keras_model


def prediction_difference(torch_model, keras_model, x_values):
    """Largest difference between the two models for the same input rows."""
    import numpy as np