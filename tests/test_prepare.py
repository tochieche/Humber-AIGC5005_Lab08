import numpy as np
import torch

from bikes_data import make_rentals
from prepare import make_loaders, make_splits


def test_dataset_shape_dtype_and_length(sample):
    features, answer = sample.train_dataset[0]
    assert len(sample.train_dataset) == 180
    assert features.shape == (sample.input_size,)
    assert answer.shape == (1,)
    assert features.dtype == torch.float32
    assert answer.dtype == torch.float32


def test_split_sizes_and_no_overlap():
    train, val, test = make_splits(make_rentals(), seed=42)
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(set(train.index) | set(val.index) | set(test.index)) == 6000
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_scaler_fitted_only_on_training(sample):
    x_train = sample.train_dataset.features.numpy()[:, :5]
    x_val = sample.val_dataset.features.numpy()[:, :5]
    assert np.allclose(x_train.mean(axis=0), 0, atol=1e-5)
    assert not np.allclose(x_val.mean(axis=0), 0, atol=1e-5)


def test_training_shuffle_repeatable(sample):
    a, val_a, test_a = make_loaders(sample, batch_size=8, seed=123)
    b, val_b, test_b = make_loaders(sample, batch_size=8, seed=123)
    assert torch.equal(next(iter(a))[0], next(iter(b))[0])
    assert not torch.equal(next(iter(a))[0], sample.train_dataset.features[:8])
    assert torch.equal(next(iter(val_a))[0], next(iter(val_b))[0])
    assert torch.equal(next(iter(test_a))[0], next(iter(test_b))[0])
