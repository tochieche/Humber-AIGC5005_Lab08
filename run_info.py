"""Printed the software backend and explained which reports are provisional."""

import keras


def describe_backend():
    """The required lab backend is TensorFlow; torch is a fallback for demos."""
    backend = keras.backend.backend()
    if backend == "tensorflow":
        return "Keras 3 using TensorFlow (required for official Lab 08 results)."
    return (
        f"Keras 3 using {backend} (local compatibility run only; "
        "rerun with TensorFlow for submission)."
    )
