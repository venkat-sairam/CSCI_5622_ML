from src.components.data_ingestion import Data_Ingestion

class Pipeline:
    def __init__(self):
        print("Inside Pipeline....")
    
    def start_data_ingestion(self):
        obj = Data_Ingestion()
        obj._initiate_data_ingestion()
