#!/bin/bash

SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
RESULT_DIR="logs"

for SCENE in $SCENE_LIST;
do
    # if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
    #     DATA_FACTOR=2
    # else
    #     DATA_FACTOR=4
    # fi

    echo "Running: $SCENE, Configuration: $CONFIG"

    # train without eval
    nvidia-smi > $RESULT_DIR/train_$SCENE.log
    CUDA_VISIBLE_DEVICES=4 python render.py -m models/$SCENE >> $RESULT_DIR/train_$SCENE.log

      CUDA_VISIBLE_DEVICES=3  python analysis/render_knn.py \
    -m models/bicycle \
    --iteration 30000 \
    --knn_k 10 \
    --metrics long_axis_consistency \
    --percentile_min 1 \
    --percentile_max 99.5
done
