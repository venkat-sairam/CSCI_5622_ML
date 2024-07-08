from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformtionArtifact


def print_before_execution(func):
    def wrapper(*args, **kwargs):
        logging.info(f"{'>>'*12} Executing function: {func.__name__}  {'<<' * 12}")
        result = func(*args, **kwargs)
        logging.info(f"{'>>'*12} Finished executing: {func.__name__}  {'<<' * 12}")
        return result

    return wrapper


class Pipeline:
    def __init__(self):
        print("Inside Pipeline....")

    @print_before_execution
    def start_data_ingestion(self):
        obj = Data_Ingestion()
        self.data_ingestion_artifact = obj._initiate_data_ingestion()
        return self.data_ingestion_artifact

    @print_before_execution
    def start_data_transformations(self):
        data_transformaton = Data_Transformation(
            data_ingestion_artifact=self.data_ingestion_artifact
        )
        self.data_transformtion_artifact = data_transformaton.initiate_data_transformation()
        return self.data_transformtion_artifact

    @print_before_execution
    def initiate_model_training(self):
        model_trainer = ModelTrainer(data_transformation_artifact=self.data_transformtion_artifact)
        model_trainer.initiate_model_trainer()
