BATCH_SIZE=60
LOG_INTERVAL=100
LR=0.00001
MAX_EPOCH=0
OPTIMIZER="adam" #"sgd" "adamax"
ALPHA=0.9
NUM_WORKERS=2

DATA_PATH="/data/langvis/data/coco/"
FEATURES_PATH="/data/langvis/experiments/detectron2/panoptic_fpn_R_101_3x/inference/"
DATASET_SUFFIX="_noun_phrases_curation_final_corrected"

CHECKPOINT_PATH="/data/langvis/experiments/phrase-grounding/noun-phrases-curation-corrected-avg-pool-preds/model_final.pth"

EXPERIMENT_NAME="release_test" #"noun-phrases-curation-corrected-avg-pool-preds" #"bert-pretrained-full-plurals-final-lr-5-e-5" #"R-101-FPN-3x-lr-1e-6-com-extension-masks-stuff-nouns-hierarchy-panoptic-bert-pretrained-full" #
OUTPUT_DIR="/data/langvis/experiments/phrase-grounding/"$EXPERIMENT_NAME
mkdir -p $OUTPUT_DIR

NUM_GPUS=1
CUDA_VISIBLE_DEVICES=2 taskset -c 20-29 python -W ignore -m main \
--init_method "tcp://localhost:8082" \
NUM_GPUS $NUM_GPUS \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
DATA.PATH_TO_FEATURES_DIR $FEATURES_PATH \
DATA.DATASET_SUFFIX $DATASET_SUFFIX \
SOLVER.MAX_EPOCH $MAX_EPOCH \
SOLVER.BASE_LR $LR \
SOLVER.OPTIMIZING_METHOD $OPTIMIZER \
TRAIN.BATCH_SIZE $BATCH_SIZE \
LOG_PERIOD $LOG_INTERVAL \
DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
TEST.ENABLE 'False' \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT_PATH
OUTPUT_DIR $OUTPUT_DIR #>> $OUTPUT_DIR"/train_log.txt"

# DATA.DATASET_SUFFIX $DATASET_SUFFIX \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT_PATH \
# MODEL.BERT_FREEZE 'True' \
# TRAIN.EVAL_FIRST 'False' \