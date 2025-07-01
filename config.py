import os

# === Random Seed ===
SEED = 42

# === Data Paths ===
SAVE_ROOT = "source"
DATASET_PATH = os.path.join(SAVE_ROOT, "dataset")
MODEL_PATH = os.path.join(SAVE_ROOT, "model")

# === Tokenizer and Model Base Name ===
BASE_MODEL_NAME = "facebook/opt-125m"

# === Data Usage Ratios ===
TOTAL_DATASET = 5000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# === Training Hyperparameters ===
NUM_EPOCHS = 20
BATCH_SIZE = 8
# LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# === ZO-SGD
# ZO_MU = 0.005
# ZO_PERTURBATIONS = 10

ZO_MU = 1e-3
ZO_PERTURBATIONS = 5
LEARNING_RATE = 5e-6


SPLIT_LAYER_IDX = 6





