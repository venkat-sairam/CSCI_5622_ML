from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import (
    os,
    sys,
    read_from_csv_file,
)
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.constants import (
    EXTERNAL_TRAIN_FILE_PATH,
    RAW_DATA_DIR_PATH,
    CURRENT_TIME_STAMP,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
)


@dataclass
class DataIngestionConfig(object):
    train_data_path = os.path.join(
        RAW_DATA_DIR_PATH,
        CURRENT_TIME_STAMP,
        TRAIN_FILE_NAME,
    )
    test_data_path = os.path.join(RAW_DATA_DIR_PATH, CURRENT_TIME_STAMP, TEST_FILE_NAME)


class Data_Ingestion:

    def __init__(self) -> None:
        self.config_info = DataIngestionConfig()

    def _initiate_data_ingestion(self):
        logging.info("Initializing data ingestion process...")
        try:
            df = read_from_csv_file(EXTERNAL_TRAIN_FILE_PATH)

            train_df, test_df = train_test_split(df, test_size=0.30)
            print(train_df.shape, test_df.shape)

            os.makedirs(os.path.dirname(self.config_info.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config_info.test_data_path), exist_ok=True)

            train_df.to_csv(self.config_info.train_data_path)
            test_df.to_csv(self.config_info.test_data_path)
            logging.info("Data ingestion completed successfully.")

            data_ingestion_artifact_details = DataIngestionArtifact(
                train_file_path=self.config_info.train_data_path,
                test_file_path=self.config_info.test_data_path,
            )
            print(f"Data ingestion artifact details: {data_ingestion_artifact_details}")
        except Exception as e:
            raise CustomException(e, sys)
