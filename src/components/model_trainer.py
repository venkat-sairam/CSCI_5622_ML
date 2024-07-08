from src.logger import logging
from src.exception import CustomException
from src.utils import *
from dataclasses import dataclass
from src.constants import (
    MODEL_TRAINER_TRAINED_MODEL_FILE_PATH,
    DECISION_TREE_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    ROOT_DIR,
)
from sklearn.metrics import balanced_accuracy_score
from src.entity.artifact_entity import DataTransformtionArtifact
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelTrainerConfig(object):
    trained_model_file_path = MODEL_TRAINER_TRAINED_MODEL_FILE_PATH


class ModelTrainer(object):

    def __init__(
        self,
        data_transformation_artifact: DataTransformtionArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        try:
            logging.info("Initializing model trainer module")
            transformed_train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            transformed_test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info("Loaded transformed train/test files")
            train_array = load_numpy_array(transformed_train_file_path)
            test_array = load_numpy_array(transformed_test_file_path)

            logging.info("Splitting the train/test arrays into input and target features...")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            classification_models_inputs = {
                "Decision_Tree_classifier": {
                    "model": DecisionTreeClassifier(),
                    "grid_params": DECISION_TREE_PARAMS,
                },
                "Logistic_Regression_Classifier": {
                    "model": LogisticRegression(),
                    "grid_params": LOGISTIC_REGRESSION_PARAMS,
                },
            }
            logging.info("Training the models...")
            metrics_list = train_model(
                X_train=x_train,
                y_train=y_train,
                list_of_models_With_grid_params=classification_models_inputs,
            )
            logging.info(f"Metrics: { metrics_list}")
            logging.info("Model training completed successfully.")
            for model in metrics_list:

                trained_object = model["model_name"]["model_object"]
                best_model = trained_object.best_estimator_
                # Get feature importances
                importances = best_model.feature_importances_

                # Select the top 20% features
                num_features = int(0.2 * len(importances))
                top_features_indices = np.argsort(importances)[-num_features:]

                # Subset the training and test sets with the selected features
                x_train_selected = x_train[:, top_features_indices]
                x_test_selected = x_test[:, top_features_indices]

                # Refit the model
                best_model.fit(x_train_selected, y_train)

                # Evaluate the refitted model on the test set
                test_score = best_model.score(x_test_selected, y_test)
                logging.info(f"Accuracy of the test set with top 20% features: {test_score}")

                save_object(file_path=MODEL_TRAINER_TRAINED_MODEL_FILE_PATH, object=trained_object)
                # save_model(model["model"], self.model_trainer_config.trained_model_file_path)
        except Exception as e:
            raise CustomException(e, sys)
