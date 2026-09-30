#!/bin/bash

SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
RESULT_DIR="logs"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"
    python train.py -s ./data/mipnerf360/$SCENE
done
