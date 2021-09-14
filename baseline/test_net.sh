BATCH_SIZE=60
LOG_INTERVAL=100
NUM_WORKERS=0

DATA_PATH="/data/langvis/data/coco/"
FEATURES_PATH="/data/langvis/experiments/detectron2/panoptic_fpn_R_101_3x/inference/"
DATASET_SUFFIX="_noun_phrases_curation_final_corrected"

EXPERIMENT_NAME="noun-phrases-curation-corrected-avg-pool-preds"
OUTPUT_DIR="/data/langvis/experiments/phrase-grounding/"$EXPERIMENT_NAME
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=5 taskset -c 50-59 python -W ignore -m main \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
DATA.PATH_TO_FEATURES_DIR $FEATURES_PATH \
DATA.DATASET_SUFFIX $DATASET_SUFFIX \
TRAIN.BATCH_SIZE $BATCH_SIZE \
TRAIN.ENABLE 'False' \
LOG_PERIOD $LOG_INTERVAL \
DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
OUTPUT_DIR $OUTPUT_DIR #>> $OUTPUT_DIR"/eval_log.txt"

# TEST.ORACLE 'True' \
# TEST.VISUALIZE_PREDICTIONS 'True' \