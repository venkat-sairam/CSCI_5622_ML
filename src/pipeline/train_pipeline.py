from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformtionArtifact
from src.components.model_evaluator import ModelEvaluator

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
        self.model_trainer_artifact = model_trainer.initiate_model_trainer()

    @print_before_execution
    def initiate_model_evaluation(self):
        model_evaluator = ModelEvaluator(
            data_ingestion_artifact=self.data_ingestion_artifact,
            model_trainer_artifact=self.model_trainer_artifact,
            data_transformtion_artifact=self.data_transformtion_artifact,
        )
        # model_evaluator.evaluate_model(data=self.data_ingestion_artifact.test_file_path)
        model_evaluator.evaluate_model(data=r"D:\CSCI_5622_ML\test_combined_participants_data.csv")
