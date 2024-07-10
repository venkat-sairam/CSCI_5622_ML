from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact", ["train_file_path", "test_file_path"])

DataTransformtionArtifact = namedtuple(
    "DataTransformtionArtifact",
    [
        "transformed_directory_path",
        "transformed_train_file_path",
        "transformed_test_file_path",
        "preprocessed_object_file_path",
    ],
)

ModelTrainerArtifact = namedtuple(
    "ModelTrainerArtifact", ["is_model_trained", "trained_model_file_path", "model_metrics"]
)
