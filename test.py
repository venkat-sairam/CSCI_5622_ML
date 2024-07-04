from src.pipeline.train_pipeline import *

class TestPipeline:
    def __init__(self) -> None:
        pass
    
    def run_pipeline(self):
        obj = Pipeline()
        self.data_ingestion_artifact = obj.start_data_ingestion()
        self.data_transformtion_artifact = obj.start_data_transformations()

obj  = TestPipeline()
obj.run_pipeline()