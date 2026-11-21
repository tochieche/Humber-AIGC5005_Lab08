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