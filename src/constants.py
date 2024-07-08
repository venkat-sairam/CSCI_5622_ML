from src.utils import os
from src.utils import datetime

ROOT_DIR = os.getcwd()
DATA_DIR = "data"
PROCESSED_DIR = "processed"
RAW_DATA_DIR = "raw"
EXTERNAL_DIR = "external"
INTERIM_DIR = "interim"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
timestamp_format = "%Y-%m-%d_%H-%M-%S"

CURRENT_TIME_STAMP = datetime.now().strftime(timestamp_format)


EXTERNAL_DIR = os.path.join(ROOT_DIR, DATA_DIR, EXTERNAL_DIR)
PROCESSED_DIR_PATH = os.path.join(ROOT_DIR, DATA_DIR, PROCESSED_DIR)
INTERIM_DIR_PATH = os.path.join(ROOT_DIR, DATA_DIR, INTERIM_DIR)
RAW_DATA_DIR_PATH = os.path.join(ROOT_DIR, DATA_DIR, RAW_DATA_DIR)

EXTERNAL_TRAIN_FILE_PATH = os.path.join(EXTERNAL_DIR, TRAIN_FILE_NAME)
EXTERNAL_TEST_FILE_PATH = os.path.join(EXTERNAL_DIR, TEST_FILE_NAME)

PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR_PATH, TRAIN_FILE_NAME)
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR_PATH, TEST_FILE_NAME)

RAW_TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR_PATH, TRAIN_FILE_NAME)
RAW_TEST_DATA_PATH = os.path.join(RAW_DATA_DIR_PATH, TEST_FILE_NAME)

# Artifacts directory constants
ARTIFACTS = "artifacts"
ARTIFACTS_DIR_PATH = os.path.join(ROOT_DIR, ARTIFACTS, CURRENT_TIME_STAMP)

# Data Transformation constants
TRANSFORMED_DIR_NAME = "transformed_data"
TRANSFORMED_TRAIN_FILE_NAME = "transformed_train_data.npy"
TRANSFORMED_TEST_FILE_NAME = "transformed_test_data.npy"
TRANSFORMED_DIR_PATH = os.path.join(ARTIFACTS_DIR_PATH, TRANSFORMED_DIR_NAME)
TRANSFORMED_TEST_FILE_PATH = os.path.join(TRANSFORMED_DIR_PATH, TRANSFORMED_TEST_FILE_NAME)
TRANSFORMED_TRAIN_FILE_PATH = os.path.join(TRANSFORMED_DIR_PATH, TRANSFORMED_TRAIN_FILE_NAME)


PREPROCESSED_DIR = "preprocessed_object"
PREPROCESSED_OBJECT_DIR_PATH = os.path.join(ARTIFACTS_DIR_PATH, PREPROCESSED_DIR)
PREPROCESSED_OBJECT_TRAIN_FILE_NAME = "preprocessed__model_object.pkl"

PREPROCESSED_OBJECT_FILE_PATH = os.path.join(
    PREPROCESSED_OBJECT_DIR_PATH, PREPROCESSED_OBJECT_TRAIN_FILE_NAME
)


# Model Trainer Constants

MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME = "trained_model.pkl"
MODEL_TRAINER_DIR_PATH = os.path.join(ARTIFACTS_DIR_PATH, MODEL_TRAINER_DIR_NAME)
MODEL_TRAINER_TRAINED_MODEL_FILE_PATH = os.path.join(
    MODEL_TRAINER_DIR_PATH, MODEL_TRAINER_TRAINED_MODEL_FILE_NAME
)

DECISION_TREE_PARAMS = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [10, 20, 30, 50, 75],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [
        2,
        4,
    ],
    "max_features": [
        "sqrt",
        "log2",
    ],
    "random_state": [42],
}
LOGISTIC_REGRESSION_PARAMS = {
    "penalty": ["l1", "l2"],
    "C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
    ],
    "solver": ["liblinear", "saga"],
    "max_iter": [10000, 12000],
    "random_state": [42],
}
