# Humber-AIGC5005_Lab08

# Lakeshore Bike Demand: Comparison of the same model in PyTorch and Keras

## What it does

This project predicts how many bikes may be rented at one station in one hour.
It compares **PyTorch** (a model we trained using our own loop) with **Keras 3**
(a model trained using `compile()` and `fit()`). The two versions use the same
Lab 07 `bikes_data.py`, the same 60/20/20 split, and the same eight input
features. 

## How to run it

Use **Python 3.12**, conda-forge, and Keras with TensorFlow. 

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
python -c "import torch, tensorflow, keras; print(keras.backend.backend())"
python train.py
python train_keras.py
python compare.py
python -m ruff format --check .
python -m ruff check .
python -m pytest
```


## Data and split

`bikes_data.py` makes 6,000 **made-up** station-hours. `prepare.py` splits
these into **3,600 training**, **1,200 validation**, and **1,200 test** rows.
Only training rows teach the scaler; the validation rows help choose the best
epoch. `shared_data.py` passes the original PyTorch arrays directly to Keras.

## Model and training

`model.py` and `keras_model.py` both use eight inputs, two 32-neuron hidden
layers with ReLU, and one **linear** output (no sigmoid). Both use MSE loss,
Adam optimizer, a learning rate of 0.003, a batch size of 64, 90 epochs, and
seeds 42, 43, and 44. `keras_model.copy_torch_weights()` transposes PyTorch's
weight matrices and checks the same output to within 0.0001 bikes.

The PyTorch loop in `train.py` manually tracks the lowest validation MSE and
restores its best weights. Keras uses `EarlyStopping` with
`restore_best_weights=True` and `ModelCheckpoint` to save the best model.


## Results 

Run the three scripts to recreate these artifacts:

| File | What it shows |
| --- | --- |
| `reports/learning_curves.png` | PyTorch main learning curve |
| `reports/learning_curves_keras.png` | Keras main learning curve |
| `reports/history.csv` | PyTorch training history |
| `reports/history_keras.csv` | Keras training history |
| `reports/experiments.csv` | PyTorch Lab 07 settings experiments |
| `reports/metrics.json` | PyTorch baselines and final score |
| `reports/keras_metrics.json` | Keras best epoch and final score |
| `reports/comparison.csv` | All six same-model comparisons |

**Important environment note:** The project requires TensorFlow.

## Experiments

`train.py` also compares a slow learning rate and a wider network from Lab 07.
These are separate from the controlled 32-neuron, 90-epoch, three-seed Lab 08
framework comparison in `compare.py`.

## The Keras version

`keras_model.py` creates identical layers and copies weights with `.T`
(transpose). `train_keras.py` calls `compile()` and `fit()` and saves the
best epoch plus a graph. `compare.py` writes one row per framework and seed.
I reviewed the six measurements in `reports/comparison.csv`, then summarized 
the evidence-based recommendation in `COMPARISON.md`.

## Tests

The `tests/` folder keeps the original Lab 07 tests and adds Keras tests
for output shapes, parameter counts, matching predictions, matching data,
learning one batch, save/load, best-weight restoration, repeatability,
and fair comparison reporting. To run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

- Seeds create repeatable runs but **not equal initial weights** in the two
  frameworks. A weight-copy test separately proves structural equivalence.
- CPU, TensorFlow version, and OS affect measured timing. The team should
  remeasure on its own computer.

### Short local smoke run (not the official 90-epoch results)

If you are only checking that the program opens, you can run
`python train_keras.py --epochs 4` and `python compare.py --epochs 4`.
These commands overwrite their output CSV/PNG with **four-epoch provisional**
results. 
