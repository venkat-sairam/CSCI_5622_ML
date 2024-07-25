from src.pipeline.train_pipeline import *


class TestPipeline:
    def __init__(self) -> None:
        pass

    def run_pipeline(self):
        obj = Pipeline()
        self.data_ingestion_artifact = obj.start_data_ingestion()
        self.data_transformtion_artifact = obj.start_data_transformations()
        self.model_trainer_artifact = obj.initiate_model_training()
        self.model_evaluation_artifact = obj.initiate_model_evaluation()

obj = TestPipeline()
obj.run_pipeline()
