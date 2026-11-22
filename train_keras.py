"""Trained the Keras version."""

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from keras_model import build_keras_model
from prepare import prepare_data
from settings import BATCH_SIZE, DATA_SEED, EPOCHS, HIDDEN_SIZE, LEARNING_RATE
from shared_data import prepared_arrays
from train import plot_curves, write_csv

REPORTS = Path(__file__).resolve().parent / "reports"


def train_keras(
    prepared,
    seed=DATA_SEED,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    hidden_size=HIDDEN_SIZE,
    checkpoint_path=None,
):
    """Fit Keras with Lab 07's hyperparameters and record EVERY epoch.

    For fairness, patience=epochs means training always gets all its rounds.
    The best validation weights are still restored after the final round.
    """
    keras.backend.clear_session()  # Do not carry an earlier Keras model forward.
    keras.utils.set_random_seed(seed)  # Seed BEFORE creating the layers.
    arrays = prepared_arrays(prepared)
    network = build_keras_model(prepared.input_size, hidden_size)
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",  # Exactly the same mean squared error used in PyTorch.
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    if checkpoint_path is None:
        checkpoint_path = REPORTS / "best_keras.keras"
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
    ]
    record = network.fit(
        arrays["x_train"],
        arrays["y_train"],
        validation_data=(arrays["x_val"], arrays["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    history = [
        {
            "epoch": i + 1,
            "train_loss": float(record.history["loss"][i]),
            "val_loss": float(record.history["val_loss"][i]),
            "val_mae": float(record.history["val_mae"][i]),
        }
        for i in range(len(record.history["loss"]))
    ]
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    return network, history, best_epoch


def test_mae(model, prepared):
    """Look at the held-out test answers ONCE, after training is complete."""
    arrays = prepared_arrays(prepared)
    predicted = model.predict(arrays["x_test"], verbose=0).reshape(-1)
    actual = arrays["y_test"].reshape(-1)  # Do not broadcast as an NxN table.
    return float(np.mean(np.abs(predicted - actual)))


def main(epochs=EPOCHS):
    """Produce the Keras history, graph, checkpoint, and final test score."""
    REPORTS.mkdir(exist_ok=True)
    prepared = prepare_data(seed=DATA_SEED)
    network, history, best_epoch = train_keras(prepared, epochs=epochs)
    write_csv(
        REPORTS / "history_keras.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_mae"],
    )
    plot_curves(
        history,
        best_epoch,
        REPORTS / "learning_curves_keras.png",
        "Keras bike network: training vs validation",
    )
    result = {
        "seed": DATA_SEED,
        "best_epoch": best_epoch,
        "validation_mae": history[best_epoch - 1]["val_mae"],
        "test_mae": test_mae(network, prepared),
        "keras_backend": keras.backend.backend(),
        "epochs": epochs,
    }
    (REPORTS / "keras_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Keras ({keras.backend.backend()}) best epoch: {best_epoch}")
    print(f"Validation MAE: {result['validation_mae']:.3f} bikes")
    print(f"Final test MAE: {result['test_mae']:.3f} bikes")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the bike model with Keras")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    main(epochs=args.epochs)
