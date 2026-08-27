#!/bin/bash

SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
RESULT_DIR="logs"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"

    # train without eval
    nvidia-smi > $RESULT_DIR/train_$SCENE.log
    CUDA_VISIBLE_DEVICES=3 python analysis/knn.py -m models/$SCENE --iteration 30000 --k 5 10 20 >> $RESULT_DIR/knn_$SCENE.log
done
