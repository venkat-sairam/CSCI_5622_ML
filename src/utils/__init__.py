from src.exception import CustomException
from src.logger import logging
import pandas as pd
import os
import numpy as np
import sys
from datetime import datetime
from dill import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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
        with open(file=file_path, mode="wb") as file:
            np.save(file, array)
        logging.info(f"Finished saving numpy array at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array(file_path: str):
    try:
        logging.info(f"loading numpy array from: {file_path}")
        with open(file=file_path, mode="rb") as file:
            array = np.load(file, allow_pickle=True)
        logging.info(f"Finished loading numpy array from: {file_path}")
        return array
    except Exception as e:
        raise CustomException(e, sys)


def train_model(X_train, y_train, list_of_models_With_grid_params):

    metrics_list = []
    for _, model_name in enumerate(list_of_models_With_grid_params):
        classifier = list_of_models_With_grid_params[model_name]["model"]
        grid_parameters = list_of_models_With_grid_params[model_name]["grid_params"]
        logging.info(f"Training model: {model_name}")

        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        logging.info(f"Applying Grid Search CV on: {model_name}")
        grid_search_clf = GridSearchCV(
            estimator=classifier,
            param_grid=grid_parameters,
            cv=stratified_cv,
            scoring="balanced_accuracy",
            refit="balanced_accuracy",
            verbose=1,
            n_jobs=-1,
        )
        grid_search_clf.fit(X_train, y_train)
        metrics_list.append(
            {
                'model_name': {
                    "best_params": grid_search_clf.best_params_,
                    "best_balanced_accuracy": grid_search_clf.best_score_,
                    "model_object": grid_search_clf,
                }
            }
        )
   
    return metrics_list
