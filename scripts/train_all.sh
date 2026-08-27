#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bonsai counter flowers garden kitchen room stump treehill"
RESULT_DIR="logs"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"

#    CUDA_VISIBLE_DEVICES=3 \
#     python train.py \
#         -s data/mipnerf360/$SCENE \
#         -m models/view_dropout_llff4/$SCENE \
#         --iteration 30000 \
#         --eval

    CUDA_VISIBLE_DEVICES=3 python render.py \
        -m models/view_dropout_llff4/$SCENE \
        -s data/mipnerf360/$SCENE \
        --eval
done