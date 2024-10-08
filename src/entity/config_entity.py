from collections import namedtuple


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple(
    "DataIngestionConfig",
    [
        "raw_data_dir",
        "ingested_dir",
        "ingested_train_dir",
        "ingested_test_dir",
        "train_test_split_ratio",
    ],
)
