from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from dataclasses import dataclass
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    DataTransformtionArtifact,
)
from src.utils import sys, read_from_csv_file, load_object, DataFrame
from imblearn.under_sampling import RandomUnderSampler

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

    def evaluate_model(self, data):
        try:
            trained_model_metrics = self.model_trainer_artifact.model_metrics

            top_feature_indices = None

            if (
                trained_model_metrics
                and trained_model_metrics[0].get("top_feature_indices") is not None
            ):
                top_feature_indices = trained_model_metrics[0]["top_feature_indices"]

            best_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"Evaluating best model: {best_model}")

            # test_data = read_from_csv_file(file_path=self.data_ingestion_artifact.test_file_path)
            test_data = read_from_csv_file(data)
            logging.info(f"{'>-' * 20} Shape of the test dta is:  {test_data.shape} {'-<' *20}")

            test_input_features, test_target_feature = (
                self.get_input_and_target_features_from_a_dataframe(
                    df=test_data,
                    columns_to_drop=["depression", "gender", "participant_id"],
                    target_feature="depression",  # Changed to string as single value
                )
            )
            logging.info(
                f"{'>-' * 20} Shape of the test dta post removal is:  {test_input_features.shape, test_target_feature.shape} {'-<' *20}"
            )

            logging.info(
                f"Loading preprocessed object from {self.data_transformtion_artifact.preprocessed_object_file_path}"
            )
            preprocessed_object = load_object(
                file_path=self.data_transformtion_artifact.preprocessed_object_file_path
            )

            logging.info("removed SMOTE on the testing dataset")
            X_test_resampled, y_test_resampled = preprocessed_object.transform(test_input_features),test_target_feature,

            logging.info("Adding column names to the resampled X_test")
            X_test_resampled_with_cols = DataFrame(
                X_test_resampled, columns=test_input_features.columns
            )
            logging.info("Successfully added column names to the resampled X_test")
            x_test_cols = test_input_features.columns
            # Select top features if specified
            if top_feature_indices is not None:
                X_test_resampled = X_test_resampled_with_cols.iloc[:, top_feature_indices]
                top_features_col_names = X_test_resampled_with_cols.columns[top_feature_indices]
                x_test_cols = top_features_col_names
            X_test_resampled = DataFrame(X_test_resampled, columns=x_test_cols)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            y_test_resampled = DataFrame(y_test_resampled)

            # Fetch participant IDs and genders corresponding to X_test_resampled indices
            participant_ids = test_data["participant_id"]
            genders = test_data["gender"]

            participants_test = participant_ids.iloc[X_test_resampled.index].reset_index(drop=True)
            genders_test = genders.iloc[X_test_resampled.index].reset_index(drop=True)

            # Evaluate model on resampled test data
            test_score = best_model.score(X_test_resampled, y_test_resampled)
            logging.info(f"Test score of {trained_model_metrics[0]['model_name']} = {test_score}")

            logging.info(
                f"Transformed size of the test dataset after applying SMOTE: {X_test_resampled.shape}, {y_test_resampled.shape}"
            )

            # Predict probabilities and create participant results DataFrame
            y_pred_proba = best_model.predict_proba(X_test_resampled)[:, 1].flatten()

            participants_results = DataFrame(
                {
                    "participant_id": participants_test,
                    "true_label": y_test_resampled.values.flatten(),
                    "predicted_prob": y_pred_proba,
                    "gender": genders_test,
                }
            )

            participants_results_grouped = participants_results.groupby('participant_id').agg({
                'true_label': 'first',  # Assumes that the 'true_label' is the same for all turns of a participant
                'predicted_prob': 'mean',  # Average prediction probability per participant
                'gender': 'first'  # Gender is the same for all turns of a participant
            })

            # Threshold the predictions at 0.5 for classification
            participants_results_grouped['final_prediction'] = (participants_results_grouped['predicted_prob'] >= 0.5).astype(int)

            # Calculate the metrics
            accuracy_A = accuracy_score(participants_results_grouped['true_label'], participants_results_grouped['final_prediction'])
            balanced_accuracy_BA = balanced_accuracy_score(participants_results_grouped['true_label'], participants_results_grouped['final_prediction'])

            # Separate the data for female and male participants
            female_data = participants_results_grouped[participants_results_grouped['gender'] == 0]
            male_data = participants_results_grouped[participants_results_grouped['gender'] == 1]

            # Calculate the metrics for female and male participants
            accuracy_female = accuracy_score(female_data['true_label'], female_data['final_prediction'])
            balanced_accuracy_female = balanced_accuracy_score(female_data['true_label'], female_data['final_prediction'])
            accuracy_male = accuracy_score(male_data['true_label'], male_data['final_prediction'])
            balanced_accuracy_male = balanced_accuracy_score(male_data['true_label'], male_data['final_prediction'])

            # Calculate Equality of Opportunity (EO)
            conf_matrix_female = confusion_matrix(female_data['true_label'], female_data['final_prediction'])
            conf_matrix_male = confusion_matrix(male_data['true_label'], male_data['final_prediction'])
            tpr_female = conf_matrix_female[1, 1] / (conf_matrix_female[1, :].sum() if conf_matrix_female[1, :].sum() != 0 else 1)
            tpr_male = conf_matrix_male[1, 1] / (conf_matrix_male[1, :].sum() if conf_matrix_male[1, :].sum() != 0 else 1)
            EO = 1 - abs(tpr_male - tpr_female)

            # Print the metrics
            print(f'Overall Accuracy (A): {accuracy_A}')
            print(f'Overall Balanced Accuracy (BA): {balanced_accuracy_BA}')
            print(f'Female Accuracy: {accuracy_female}')
            print(f'Female Balanced Accuracy: {balanced_accuracy_female}')
            print(f'Male Accuracy: {accuracy_male}')
            print(f'Male Balanced Accuracy: {balanced_accuracy_male}')
            print(f'Equality of Opportunity (EO): {EO}')
        except Exception as e:
            raise CustomException(e)
