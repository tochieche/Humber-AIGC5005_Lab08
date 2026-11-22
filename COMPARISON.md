# PyTorch vs Keras: what should Lakeshore Bike Share use?

I need to compare two versions of the **same** bike-rental
prediction network. Both models have two hidden layers of 32 neurons, ReLU,
and one linear output. Both read the same 3,600 training rows, 1,200 validation
rows, and 1,200 test rows. The training settings are MSE loss, Adam optimizer,
a learning rate of 0.003, batch size 64, and 90 epochs per seed. I repeat
the comparison for seeds 42, 43, and 44. The CSV in `reports/comparison.csv`
records six actual runs, including validation MAE, test MAE, best epoch,
and training seconds.

**Difference 1:** In `train.py`, PyTorch calculates predictions,
loss, gradients, and optimizer steps explicitly in `run_epoch()`. In
`train_keras.py`, Keras performs the same jobs using `compile()` and `fit()`.
This is less loop code to look after. **Difference 2:** 
PyTorch saves a copy of its best `state_dict()` in memory, while
Keras uses `EarlyStopping(restore_best_weights=True)` plus a disk checkpoint
named `best_keras.keras`. **Difference 3: ** Across the three local 90-epoch runs,
the average test MAE was **2.852 bikes** for PyTorch and **2.843 bikes** for
Keras. Average training time was **4.06 seconds** for PyTorch and **21.44
seconds** for Keras. 

**My conditional recommendation is to use Keras** if the weekly task
stays a small, straightforward prediction model. A short `fit()` call and
built-in callbacks make weekly retraining easier to read and maintain. The
team gives up some of the direct, step-by-step control that PyTorch offers
inside its training loop. 
**I would switch to PyTorch** if the team plans to customize gradients,
needs complex model debugging, or chooses a Hugging Face model primarily
supported in PyTorch next term. 
