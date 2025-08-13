from enum import Enum


class DatasetStage(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    PREDICT = "predict"