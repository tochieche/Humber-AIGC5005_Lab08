"""Handed Keras the EXACT arrays that Lab 07 handed to PyTorch."""


def prepared_arrays(prepared):
    """Reuse Lab 07's scalers, splits, and float32 tensors without re-fitting.

    The target has shape (N, 1) in both versions, avoiding accidental NxN
    broadcasting when calculating how wrong a guess was.
    """
    return {
        "x_train": prepared.train_dataset.features.numpy(),
        "y_train": prepared.train_dataset.targets.numpy(),
        "x_val": prepared.val_dataset.features.numpy(),
        "y_val": prepared.val_dataset.targets.numpy(),
        "x_test": prepared.test_dataset.features.numpy(),
        "y_test": prepared.test_dataset.targets.numpy(),
    }
