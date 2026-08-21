#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"
RESULT_DIR="logs"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"

#    CUDA_VISIBLE_DEVICES=3 \
    python analysis/covariance.py \
        -m models/bicycle \
        --iteration 30000 \
        --k 5 10 20

done