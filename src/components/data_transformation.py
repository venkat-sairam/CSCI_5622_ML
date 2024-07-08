from src.exception import CustomException
from src.logger import logging
from src.utils import read_from_csv_file
from dataclasses import dataclass
from src.constants import (
    TRANSFORMED_DIR_PATH,
    TRANSFORMED_TRAIN_FILE_PATH,
    TRANSFORMED_TEST_FILE_PATH,
    PREPROCESSED_OBJECT_DIR_PATH,
    PREPROCESSED_OBJECT_FILE_PATH,
)
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformtionArtifact
from src.utils import sys, read_from_csv_file
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src.utils import np, save_object, save_numpy_array


@dataclass
class DataTransformationConfig(object):

    transformed_directory_path = TRANSFORMED_DIR_PATH
    transformed_train_file_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_file_path = TRANSFORMED_TEST_FILE_PATH
    preprocessed_directory = PREPROCESSED_OBJECT_DIR_PATH
    preprocessed_object_file_path = PREPROCESSED_OBJECT_FILE_PATH


class Data_Transformation:
    def __init__(
        self,
        data_transform_config=DataTransformationConfig(),
        data_ingestion_artifact=DataIngestionArtifact,
    ):
        try:
            logging.info("Initializing data Transformation module")
            self.data_transform_config = data_transform_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logging.info("Initialized Data Transformation module successfully...")
        except Exception as e:
            raise CustomException(e, sys)

    def handle_duplicates(self, X):
        return X.drop_duplicates()

    def get_preprocessed_model_object(self, X_train):
        try:

            logging.info("Transforming the dataset using SMOTE")
            numerical_transformer = Pipeline(
                steps=[
                    ("drop_duplicates", FunctionTransformer(self.handle_duplicates)),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[("trf", numerical_transformer, X_train.columns)]
            )

            smote = SMOTE(random_state=42)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("smote", smote)])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def get_input_and_target_features_from_a_dataframe(self, df, columns_to_drop, target_feature):
        X = df.drop(columns=columns_to_drop)
        y = df[target_feature]
        return X, y

    def initiate_data_transformation(self):
        try:
            logging.info("Initializing data transformation process")
            train_df = read_from_csv_file(self.data_ingestion_artifact.train_file_path)
            test_df = read_from_csv_file(self.data_ingestion_artifact.test_file_path)

            train_input_features, train_target_feature = (
                self.get_input_and_target_features_from_a_dataframe(
                    df=train_df,
                    columns_to_drop=["depression", "gender", "participant_id"],
                    target_feature=["depression"],
                )
            )

            test_input_features, test_target_feature = (
                self.get_input_and_target_features_from_a_dataframe(
                    df=test_df,
                    columns_to_drop=["depression", "gender", "participant_id"],
                    target_feature=["depression"],
                )
            )

            participant_ids = train_df["participant_id"]
            gender = train_df["gender"]

            preprocessed_object = self.get_preprocessed_model_object(X_train=train_input_features)

            logging.info("Applyng SMOTE on the training dataset...")
            X_train_resampled, y_train_resampled = preprocessed_object["smote"].fit_resample(
                preprocessed_object["preprocessor"].fit_transform(train_input_features),
                train_target_feature,
            )
            logging.info(
                f"transformed size of the train dataset after applying SMOTE: {X_train_resampled.shape, y_train_resampled.shape}"
            )
            logging.info("Applyng SMOTE on the testing dataset")
            X_test_resampled, y_test_resampled = preprocessed_object["smote"].fit_resample(
                preprocessed_object["preprocessor"].transform(test_input_features),
                test_target_feature,
            )

            logging.info(
                f"transformed size of the test dataset after applying SMOTE: {X_test_resampled.shape, y_test_resampled.shape}"
            )

            train_array = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_array = np.c_[X_test_resampled, np.array(y_test_resampled)]

            logging.info(
                f"Saving the transformed train dataset to {self.data_transform_config.transformed_train_file_path}"
            )

            save_object(
                file_path=self.data_transform_config.preprocessed_object_file_path,
                object=preprocessed_object,
            )
            save_numpy_array(
                file_path=self.data_transform_config.transformed_train_file_path, array=train_array
            )
            save_numpy_array(
                file_path=self.data_transform_config.transformed_test_file_path, array=test_array
            )

            transformed_directory_path = self.data_transform_config.transformed_directory_path
            transformed_train_file_path = self.data_transform_config.transformed_train_file_path
            transformed_test_file_path = self.data_transform_config.transformed_test_file_path
            preprocessed_object_file_path = (
                self.data_transform_config.preprocessed_object_file_path
            )

            data_transformation_artifact_details = DataTransformtionArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_directory_path=transformed_directory_path,
            )
            logging.info(
                f"DataTransformtionArtifact details : {data_transformation_artifact_details}"
            )
            logging.info("Data transformation completed successfully...")

            return data_transformation_artifact_details

        except Exception as e:
            raise CustomException(e, sys)
