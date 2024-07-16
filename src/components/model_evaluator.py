from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from dataclasses import dataclass
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    DataTransformtionArtifact,
)
from src.utils import sys, read_from_csv_file, load_object


class ModelEvaluator(object):

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformtion_artifact: DataTransformtionArtifact,
    ) -> None:
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformtion_artifact = data_transformtion_artifact

    def get_input_and_target_features_from_a_dataframe(self, df, columns_to_drop, target_feature):
        X = df.drop(columns=columns_to_drop)
        y = df[target_feature]
        return X, y

    def evaluate_model(self):
        try:
            trained_model_metrics = self.model_trainer_artifact.model_metrics

            top_feature_indices= None

            if trained_model_metrics[0]["top_feature_indices"] is not None:
                top_feature_indices = trained_model_metrics[0]["top_feature_indices"]
            best_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"evaluate :==> best model is : {best_model}")
            test_data = read_from_csv_file(file_path=self.data_ingestion_artifact.test_file_path)

            test_input_features, test_target_feature = (
                self.get_input_and_target_features_from_a_dataframe(
                    df=test_data,
                    columns_to_drop=["depression", "gender", "participant_id"],
                    target_feature=["depression"],
                )
            )
            logging.info(
                f"Loading the preprocessed object from {self.data_transformtion_artifact.preprocessed_object_file_path}"
            )
            preprocessed_object = load_object(
                file_path=self.data_transformtion_artifact.preprocessed_object_file_path
            )

            logging.info("Applyng SMOTE on the testing dataset")
            X_test_resampled, y_test_resampled = preprocessed_object["smote"].fit_resample(
                preprocessed_object["preprocessor"].transform(test_input_features),
                test_target_feature,
            )
            X_test_resampled = X_test_resampled[:, top_feature_indices] if top_feature_indices is not None else X_test_resampled
            
            test_score = best_model.score(X_test_resampled, y_test_resampled)
            logging.info(f"Test score of {trained_model_metrics[0]['model_name']} = {test_score}")

            logging.info(
                f"transformed size of the test dataset after applying SMOTE: {X_test_resampled.shape, y_test_resampled.shape}"
            )
        except Exception as e:
            raise CustomException(e, sys)
