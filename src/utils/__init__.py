from src.exception import CustomException
from src.logger import logging
import pandas as pd
import os
import numpy as np
import sys
from datetime import datetime
from dill import dump


def read_from_csv_file(file_path: str) -> pd.DataFrame:
    try:
        logging.info(f"reading csv file at: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Finished reading csv file at: {file_path}")
        return df
    except Exception as e:
        raise CustomException(e)


def create_directory(filepath: str):
    try:

        os.makedirs(filepath, exist_ok=True)
        logging.info(f"Created directory at: {filepath}")
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, object: object):
    logging.info(f"saving object at: {file_path}")
    try:
        create_directory(os.path.dirname(file_path))
        with open(file_path, mode="wb") as file_handle:
            dump(object, file_handle)
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array(file_path: str, array):
    try:
        logging.info(f"saving numpy array at: {file_path}")
        create_directory(os.path.dirname(file_path))
        with open(file= file_path, mode="wb") as file:
            np.save(file, array)
        logging.info(f"Finished saving numpy array at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
