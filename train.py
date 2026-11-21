"""Run the whole project with one command: python train.py.

The network guesses a bike count, checks how wrong it was, 
and gently changes its numbers. 

It does this many times.
"""

import copy
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import make_loaders, prepare_data

matplotlib.use("Agg")  # Save graphs on computers without a screen.
REPORTS = Path(__file__).resolve().parent / "reports"
SEED = 42
EPOCHS = 90


def set_seed(seed=SEED):
    """Make the random choices repeatable so experiments are fair."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Reproducible and fast for this small CPU lab.
    torch.use_deterministic_algorithms(True)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Train one pass if optimizer is given; otherwise only check accuracy.

    MAE means 'mean absolute error': if MAE = 5, guesses miss by about 5 bikes.
    The MSE loss punishes large mistakes more than tiny ones.
    """
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_absolute_error = 0.0
    count = 0
    for features, actual in loader:  # Take a small batch of examples.
        with torch.set_grad_enabled(training):
            predicted = model(features)  # Forward the network guesses.
            if predicted.shape != actual.shape:
                raise ValueError("Targets and guesses must have the same shape")
            loss = loss_fn(predicted, actual)  # Find how wrong it is.
            if training:
                optimizer.zero_grad()  # Clear yesterday's gradient notes.
                loss.backward()  # Backward: find which weights to change.
                optimizer.step()  # Actually change the weights.
        # Keep totals weighted by batch size, including the last shorter batch.
        total_squared_error += torch.sum((predicted - actual) ** 2).item()
        total_absolute_error += torch.sum(torch.abs(predicted - actual)).item()
        count += actual.numel()
    return {"loss": total_squared_error / count, "mae": total_absolute_error / count}


def train_many(model, train_loader, val_loader, epochs=EPOCHS, lr=0.003):
    """Practice for epochs rounds; restore the best validation weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)  # The best quiz score wins.
    return history, best_epoch


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(history, best_epoch, path, title):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Training MSE")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation MSE")
    plt.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch (one complete pass through training examples)")
    plt.ylabel("Mean squared error (lower is better)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hour_data(frame, path):
    """Show why one straight line struggles with busy morning/evening hours."""
    hourly = frame.groupby("hour")[TARGET].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.xlabel("Hour of the day")
    plt.ylabel("Mean bikes rented")
    plt.title("Average bike demand changes during the day (all stations)")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def baseline_results(prepared):
    """Compare with (1) one constant number and (2) one straight-line model."""
    train_y = prepared.train_frame[TARGET].to_numpy()
    val_y = prepared.val_frame[TARGET].to_numpy()
    test_y = prepared.test_frame[TARGET].to_numpy()
    training_mean = float(train_y.mean())
    mean_val = np.full_like(val_y, training_mean, dtype=float)
    mean_test = np.full_like(test_y, training_mean, dtype=float)
    straight_line = LinearRegression()
    straight_line.fit(
        prepared.train_dataset.features.numpy(), train_y
    )  # Only train rows!
    linear_val = straight_line.predict(prepared.val_dataset.features.numpy())
    linear_test = straight_line.predict(prepared.test_dataset.features.numpy())
    results = {}
    for name, predictions_val, predictions_test in (
        ("Training mean", mean_val, mean_test),
        ("Linear regression", linear_val, linear_test),
    ):
        results[name] = {
            "validation_mae": float(mean_absolute_error(val_y, predictions_val)),
            "test_mae": float(mean_absolute_error(test_y, predictions_test)),
            "validation_mse": float(mean_squared_error(val_y, predictions_val)),
            "test_mse": float(mean_squared_error(test_y, predictions_test)),
        }
    return results


def do_experiment(prepared, name, lr, width, batch_size, epochs=EPOCHS):
    # Start every experiment at the same seed to make comparisons fair.
    set_seed(SEED)
    train_loader, val_loader, _ = make_loaders(prepared, batch_size, SEED)
    model = BikeNet(prepared.input_size, hidden_size=width)
    history, best_epoch = train_many(model, train_loader, val_loader, epochs, lr)
    best = history[best_epoch - 1]
    result = {
        "run": name,
        "learning_rate": lr,
        "hidden_width": width,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_mae": best["val_mae"],
        "best_val_loss": best["val_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }
    return model, history, result


def main():
    REPORTS.mkdir(exist_ok=True)
    set_seed()
    frame = make_rentals()
    prepared = prepare_data(frame, seed=SEED)
    plot_hour_data(frame, REPORTS / "rentals_by_hour.png")

    # Each extra run changes exactly ONE setting compared with main.
    settings = [
        ("main", 0.003, 32, 64),
        ("slow_learning", 0.0001, 32, 64),
        ("wide_network", 0.003, 128, 64),
    ]
    experiments = []
    main_model = None
    main_history = None
    main_best_epoch = None
    for name, lr, width, batch_size in settings:
        model, history, result = do_experiment(prepared, name, lr, width, batch_size)
        experiments.append(result)
        plot_curves(
            history,
            result["best_epoch"],
            REPORTS / f"{name}_curves.png",
            f"{name.replace('_', ' ').title()}: training vs validation",
        )
        print(
            f"{name}: best epoch={result['best_epoch']}, "
            f"validation MAE={result['best_val_mae']:.3f} bikes"
        )
        if name == "main":
            main_model = model
            main_history = history
            main_best_epoch = result["best_epoch"]

    plot_curves(
        main_history,
        main_best_epoch,
        REPORTS / "learning_curves.png",
        "Main network: training vs validation",
    )
    write_csv(
        REPORTS / "history.csv",
        main_history,
        ["epoch", "train_loss", "val_loss", "val_mae"],
    )
    write_csv(
        REPORTS / "experiments.csv",
        experiments,
        list(experiments[0]),
    )

    # Pick by VALIDATION only. Do not use test rows for model selection.
    chosen_run = min(experiments, key=lambda run: run["best_val_loss"])
    if chosen_run["run"] == "main":
        chosen_model = main_model
    else:
        chosen_model, _, _ = do_experiment(
            prepared,
            chosen_run["run"],
            chosen_run["learning_rate"],
            chosen_run["hidden_width"],
            chosen_run["batch_size"],
        )
    _, val_loader, test_loader = make_loaders(prepared)
    selected_val = run_epoch(chosen_model, val_loader, torch.nn.MSELoss())
    # Test is used ONCE at the end, after choosing the network.
    selected_test = run_epoch(chosen_model, test_loader, torch.nn.MSELoss())
    baselines = baseline_results(prepared)
    baselines["Neural network"] = {
        "validation_mae": selected_val["mae"],
        "test_mae": selected_test["mae"],
        "validation_mse": selected_val["loss"],
        "test_mse": selected_test["loss"],
    }
    metrics = {
        "seed": SEED,
        "split": {"train": 3600, "validation": 1200, "test": 1200},
        "selection_criterion": "lowest validation MSE",
        "selected_network": chosen_run["run"],
        "baseline_and_network_metrics": baselines,
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("Final selected model:", chosen_run["run"])
    print("Final test MAE:", round(selected_test["mae"], 3), "bikes")
    return metrics, experiments


if __name__ == "__main__":
    main()
