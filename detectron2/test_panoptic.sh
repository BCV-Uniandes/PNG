NUM_GPUS=1
EXP_DIR="/data/langvis/experiments/detectron2"
CONFIG_PATH="configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
NUM_WORKERS=0

# for epoch in {0000337..0014195..0000338}
# do
WEIGHTS=$EXP_DIR"/panoptic_fpn_R_101_3x"
CUDA_VISIBLE_DEVICES=7 python tools/train_net.py --num-gpus $NUM_GPUS \
    --eval-only \
    --config-file $CONFIG_PATH \
    --dist-url tcp://0.0.0.0:12340 OUTPUT_DIR "../data/panoptic_narrative_grounding/features/val2017" \
    MODEL.WEIGHTS $WEIGHTS"/panoptic_fpn_R_101_3x.pkl" \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# done