"""One set of settings for the TWO frameworks for the comparison."""

DATA_SEED = 42  # This fixes which rows belong to train, validation, and test.
SEEDS = (42, 43, 44)  # Three different starts for a trustworthy comparison.
HIDDEN_SIZE = 32  # Both neural networks have two layers of this width.
LEARNING_RATE = 0.003  # How big one learning step is.
BATCH_SIZE = 64  # How many rows the model studies at a time.
EPOCHS = 90  # Each framework gets the same number of study rounds.
