from enum import Enum

class Variables(Enum):
    VAR_LR = 't2m'
    VAR_HR = 'T_2M'

class Format(Enum):
    EXT_IN = 'nc'
    EXT_OUT = 'png'

class Dir(Enum):
    LR_PATH = "/m100_work/IscrC_DLCC/data/era-interim-1979-2018-158x72-LR-075"
    LR_PATH_TEST = "/m100_work/IscrC_DLCC/data/era-interim-1979-2018-158x72-LR-075_TEST"
    HR_PATH = "/m100_work/IscrC_DLCC/data/era-interim-1979-2018-947x431-HR-0125"
    HR_PATH_TEST = "/m100_work/IscrC_DLCC/data/era-interim-1979-2018-947x431-HR-0125_TEST"

class Normalization(Enum):
    UPPER_LR = 1
    LOWER_LR = -1
    UPPER_HR = 1
    LOWER_HR = -1
    MAX_HR = 309.78953742151464
    MIN_HR = 211.29322814941406
    MAX_LR = 302.0277716354266
    MIN_LR = 239.30581206241132

class Config(Enum):
    TEST_TRAIN_RATIO = 0.95
    CHANNELS = 1
    CHECKPOINT_EPOCH = 5
    EPOCHS = 150
    BATCH_SIZE = 4
    LR_GEN = 0.0001
    LR_DIS = 0.0003
