from src.utils import os
from src.utils import datetime
ROOT_DIR = os.getcwd()
DATA_DIR = "data"
PROCESSED_DIR = "processed"
RAW_DATA_DIR = "raw"
EXTERNAL_DIR = "external"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
timestamp_format = "%Y-%m-%d_%H-%M-%S"

CURRENT_TIME_STAMP =  datetime.now().strftime(timestamp_format)


EXTERNAL_DIR = os.path.join(ROOT_DIR,DATA_DIR, EXTERNAL_DIR)
PROCESSED_DIR_PATH = os.path.join(ROOT_DIR, DATA_DIR, PROCESSED_DIR)
RAW_DATA_DIR_PATH = os.path.join(ROOT_DIR, DATA_DIR, RAW_DATA_DIR)

EXTERNAL_TRAIN_FILE_PATH = os.path.join(EXTERNAL_DIR, TRAIN_FILE_NAME)
EXTERNAL_TEST_FILE_PATH = os.path.join(EXTERNAL_DIR, TEST_FILE_NAME)

PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR_PATH, TRAIN_FILE_NAME)
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR_PATH, TEST_FILE_NAME)

RAW_TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR_PATH, TRAIN_FILE_NAME)
RAW_TEST_DATA_PATH = os.path.join(RAW_DATA_DIR_PATH, TEST_FILE_NAME)
