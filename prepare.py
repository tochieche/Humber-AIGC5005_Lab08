"""Split rows safely and turn them into small batches of tensors."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]


class BikeDataset(Dataset):
    """A schoolbag of examples: give back ONE example when asked for it."""

    def __init__(self, features, targets):
        self.features = torch.as_tensor(
            np.asarray(features, dtype=np.float32), dtype=torch.float32
        )
        # Shape [N, 1] matches the network's prediction shape [N, 1].
        self.targets = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
        )
        if len(self.features) != len(self.targets):
            raise ValueError("The features and target must have equal row counts")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


@dataclass
class PreparedData:
    train_frame: object
    val_frame: object
    test_frame: object
    preprocessor: ColumnTransformer
    train_dataset: BikeDataset
    val_dataset: BikeDataset
    test_dataset: BikeDataset

    @property
    def input_size(self):
        return self.train_dataset.features.shape[1]


def make_splits(frame=None, seed=42):
    """First split, THEN learn preprocessing from training rows alone."""
    if frame is None:
        frame = make_rentals()
    # 60% train, 20% validation, 20% test.
    train, remaining = train_test_split(frame, test_size=0.4, random_state=seed)
    val, test = train_test_split(remaining, test_size=0.5, random_state=seed)
    return train.copy(), val.copy(), test.copy()


def prepare_data(frame=None, seed=42):
    """Fit the scaler and one-hot encoder ONLY on the training subset."""
    train, val, test = make_splits(frame, seed)
    transformer = ColumnTransformer(
        [
            ("numbers", StandardScaler(), NUMERIC),
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    x_train = transformer.fit_transform(train)  # Learn from train ONLY.
    x_val = transformer.transform(val)  # Never call fit() on val/test.
    x_test = transformer.transform(test)
    return PreparedData(
        train_frame=train,
        val_frame=val,
        test_frame=test,
        preprocessor=transformer,
        train_dataset=BikeDataset(x_train, train[TARGET].to_numpy()),
        val_dataset=BikeDataset(x_val, val[TARGET].to_numpy()),
        test_dataset=BikeDataset(x_test, test[TARGET].to_numpy()),
    )


def make_loaders(prepared, batch_size=64, seed=42):
    """Shuffle the practice pile, but not the validation and test piles."""
    train_gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            prepared.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=train_gen,
        ),
        DataLoader(prepared.val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(prepared.test_dataset, batch_size=batch_size, shuffle=False),
    )
