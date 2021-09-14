"""Configuration"""

from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 60

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Evaluate before training.
_C.TRAIN.EVAL_FIRST = True

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Maximum number of panoptic segmentation proposals during testing.
_C.TEST.NUM_BOXES = 58

# Maximum number of noun phrases.
_C.TEST.MAX_NOUN_PHRASES = 28

# If true evaluate the oracle performance.
_C.TEST.ORACLE = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# MAximum sequence length.
_C.MODEL.MAX_SEQUENCE_LENGTH = 230

# Maximum number of panoptic segmentation proposals during training.
_C.MODEL.NUM_BOXES = 93

# Maximum number of plural noun phrases during testing.
_C.MODEL.MAX_PLURAL_REGIONS = 13

# If true freeze BERT model.
_C.MODEL.BERT_FREEZE = False

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# Dataset suffix if any.
_C.DATA.DATASET_SUFFIX = ""

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The path to the features directory.
_C.DATA.PATH_TO_FEATURES_DIR = ""

# Train split.
_C.DATA.TRAIN_SPLIT = "train2017"

# Validation split.
_C.DATA.VAL_SPLIT = "val2017"

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.0001

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adam"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 20

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


def _assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())