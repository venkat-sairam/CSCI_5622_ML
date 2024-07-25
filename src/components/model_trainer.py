from src.logger import logging
from src.exception import CustomException
from src.utils import *
from dataclasses import dataclass
from src.constants import (
    MODEL_TRAINER_TRAINED_MODEL_FILE_PATH,
    DECISION_TREE_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    ROOT_DIR,
    RANDOM_FOREST_PARAMS,
)
from sklearn.metrics import balanced_accuracy_score
from src.entity.artifact_entity import DataTransformtionArtifact, ModelTrainerArtifact
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
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
                # "Logistic_Regression_Classifier": {
                #     "model": LogisticRegression(),
                #     "grid_params": LOGISTIC_REGRESSION_PARAMS,
                # },
                # "Random Forest Classifier":{
                #     "model": RandomForestClassifier(),
                #     "grid_params": RANDOM_FOREST_PARAMS
                # }
            }
            logging.info("Training the models...")
            metrics_list = train_model(
                X_train=x_train,
                y_train=y_train,
                list_of_models_With_grid_params=classification_models_inputs,
            )
            logging.info(f"Metrics: { metrics_list}")
            logging.info("Model training completed successfully.")
            best_model_list = [{"score": 0, "model_name": None, "top_features_indices": None}]

            for model in metrics_list:

                trained_object = model["model_details"]["model_object"]
                trained_model_name = model["model_details"]["model_name"]

                best_model = trained_object.best_estimator_

                x_train_selected = x_train
                x_test_selected = x_test
                top_features_indices=None

                importances = None
                # if hasattr(best_model, "feature_importances_"):
                #     # Get feature importances
                #     importances = best_model.feature_importances_

                #     # Select the top 20% features
                #     num_features = int(0.35 * len(importances))
                #     top_features_indices = np.argsort(importances)[-num_features:]

                #     # Subset the training and test sets with the selected features
                #     x_train_selected = x_train[:, top_features_indices]
                #     x_test_selected = x_test[:, top_features_indices]

                # Refit the model
                best_model.fit(x_train_selected, y_train)

                # Evaluate the refitted model on the test set
                test_score = best_model.score(x_test_selected, y_test)
                logging.info(f"Test score of {trained_model_name} = {test_score}")

                # Update the best model, score if the current model's accuracy is higher
                current_best_score = best_model_list[0].get("score")

                if test_score > current_best_score:
                    best_model_list[0]["score"] = test_score
                    best_model_list[0]["model_name"] = trained_model_name
                    best_model_list[0]["trained_model_object"] = best_model
                    best_model_list[0]["top_feature_indices"] = top_features_indices

            best_model_name = best_model_list[0]["model_name"]
            best_model_score = best_model_list[0]["score"]
            best_model_objecct = best_model_list[0]["trained_model_object"]

            logging.info(
                f"Best Accuracy of  {best_model_name} with the test set is  ==> {best_model_score}"
            )
            logging.info(f"Best Model is: {best_model_objecct}")

            save_object(
                file_path=MODEL_TRAINER_TRAINED_MODEL_FILE_PATH,
                object=best_model_list[0]["trained_model_object"],
            )
            logging.info(
                f"Trained {best_model_list[0]['model_name']} model object saved at: {MODEL_TRAINER_TRAINED_MODEL_FILE_PATH}"
            )

            model_trainer_artifact_details = ModelTrainerArtifact(
                is_model_trained=True,
                trained_model_file_path=MODEL_TRAINER_TRAINED_MODEL_FILE_PATH,
                model_metrics=best_model_list,
            )
            logging.info(f"Model trainer artifact details: {model_trainer_artifact_details}")
            logging.info("Model trainer completed successfully.")
            return model_trainer_artifact_details
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
