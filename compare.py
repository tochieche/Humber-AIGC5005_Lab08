"""Trained TWO networks on THREE seeds: `python compare.py`."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import keras
import torch

from model import BikeNet
from prepare import make_loaders, prepare_data
from settings import BATCH_SIZE, DATA_SEED, EPOCHS, HIDDEN_SIZE, LEARNING_RATE, SEEDS
from train import run_epoch, set_seed, train_many, write_csv
from train_keras import test_mae, train_keras

REPORTS = Path(__file__).resolve().parent / "reports"
COLUMNS = [
    "framework", "seed", "best_epoch", "validation_mae", "test_mae", "seconds"
]


def compare_one_seed(prepared, seed, epochs=EPOCHS, checkpoint_dir=None):
    """Use one common split; time the actual training, NOT preparing data."""
    if checkpoint_dir is None:
        checkpoint_dir = REPORTS
    checkpoint_dir = Path(checkpoint_dir)
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        prepared, batch_size=BATCH_SIZE, seed=seed
    )
    pytorch_model = BikeNet(prepared.input_size, hidden_size=HIDDEN_SIZE)
    start = perf_counter()
    pytorch_history, pytorch_best = train_many(
        pytorch_model, train_loader, val_loader, epochs=epochs, lr=LEARNING_RATE
    )
    pytorch_seconds = perf_counter() - start
    # This happens just once, AFTER restoring the best validation weights.
    pytorch_test = run_epoch(pytorch_model, test_loader, torch.nn.MSELoss())["mae"]
    torch_result = {
        "framework": "PyTorch",
        "seed": seed,
        "best_epoch": pytorch_best,
        "validation_mae": pytorch_history[pytorch_best - 1]["val_mae"],
        "test_mae": pytorch_test,
        "seconds": round(pytorch_seconds, 3),
    }

    start = perf_counter()
    keras_model, keras_history, keras_best = train_keras(
        prepared,
        seed=seed,
        epochs=epochs,
        checkpoint_path=checkpoint_dir / f"best_keras_seed{seed}.keras",
    )
    keras_seconds = perf_counter() - start
    keras_result = {
        "framework": "Keras",
        "seed": seed,
        "best_epoch": keras_best,
        "validation_mae": keras_history[keras_best - 1]["val_mae"],
        "test_mae": test_mae(keras_model, prepared),
        "seconds": round(keras_seconds, 3),
    }
    return [torch_result, keras_result]