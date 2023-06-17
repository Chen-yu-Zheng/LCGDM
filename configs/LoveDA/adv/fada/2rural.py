from configs.LoveDA.ToRURAL import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET


MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 2000
#SNAPSHOT_DIR = './log/fada/20k_ks4_cn7_seed2333_w6k_l1e-1/2rural'
# SNAPSHOT_DIR = './log/fada/20k_seed2333/2rural'
SNAPSHOT_DIR = './log/fada/20k_ent_e-2/2rural'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 5e-3
LEARNING_RATE_D = 1e-4
NUM_STEPS = 20 * 1000
NUM_STEPS_STOP = 20 * 1000  # Use damping instead of early stopping
WARMUP_STEP = 6000
ITER_SIZE=1
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
LAMBDA_SEG = 0.1
LAMBDA_ADV = 0.001
PENALTY_LOSS_WEIGHT = 0.01


TARGET_SET = TARGET_SET

EVAL_EVERY=2000
TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG=TARGET_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG

KERNEL_SIZE = 4 
CROP_NUM = 7