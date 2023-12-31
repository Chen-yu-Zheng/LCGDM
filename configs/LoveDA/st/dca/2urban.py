from configs.LoveDA.ToURBAN import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

# SNAPSHOT_DIR = './log/dca/dca_20k_seed2333/2urban'
# SNAPSHOT_DIR = './log/dca/dca_20k_ks4_cn7_seed2333/2urban'
SNAPSHOT_DIR = './log/dca/dca_20k_ent_seed2333/2urban'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 20000  # for learning rate poly
NUM_STEPS_STOP = 20000  # Use damping instead of early stopping
FIRST_STAGE_STEP = 6000  # for first stage
PREHEAT_STEPS = int(NUM_STEPS / 20)  # for warm-up
POWER = 0.9  # lr poly power
EVAL_EVERY = 2000
GENERATE_PSEDO_EVERY = 1000
MULTI_LAYER = False
IGNORE_BG = True
PSEUDO_SELECT = True

PENALTY_LOSS_WEIGHT = 0.25


TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG = SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG = TARGET_DATA_CONFIG
EVAL_DATA_CONFIG = EVAL_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG

KERNEL_SIZE = 4
CROP_NUM = 7
WEIGHT_LOCAL = 0.5
WEIGHT_GLOBAL = 0.5