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
COLUMNS = ["framework", "seed", "best_epoch", "validation_mae", "test_mae", "seconds"]


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


def main(epochs=EPOCHS):
    """Create all six scores with one command and clean seed-by-seed runs.

    Every seed starts in its own Python process. This avoids one framework
    carrying memory or hidden state from an earlier seed into the next.
    """
    REPORTS.mkdir(exist_ok=True)
    results = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory() as folder:
            result_path = Path(folder) / "two_scores.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single-seed",
                    str(seed),
                    "--seed-output",
                    str(result_path),
                    "--epochs",
                    str(epochs),
                ],
                check=True,
            )
            # A short-lived child keeps large neural-network libraries fresh.
            two_results = json.loads(result_path.read_text(encoding="utf-8"))
        results.extend(two_results)
        for result in two_results:
            print(
                f"{result['framework']:7} seed={seed} "
                f"val MAE={result['validation_mae']:.3f} "
                f"test MAE={result['test_mae']:.3f} "
                f"seconds={result['seconds']:.1f}",
                flush=True,
            )
    write_csv(REPORTS / "comparison.csv", results, COLUMNS)
    print("Saved reports/comparison.csv with six training runs")
    return results


def save_single_seed(seed, epochs, output):
    """Internal worker: do ONE pair in a fresh process, then save results."""
    prepared = prepare_data(seed=DATA_SEED)
    results = compare_one_seed(prepared, seed, epochs=epochs)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Worker finished seed {seed} using Keras {keras.backend.backend()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair 3-seed framework benchmark")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--single-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    if args.single_seed is not None:
        if not args.seed_output:
            parser.error("--seed-output must be supplied to a worker")
        save_single_seed(args.single_seed, args.epochs, args.seed_output)
    else:
        main(epochs=args.epochs)
