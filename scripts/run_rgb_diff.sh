#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"

for SCENE in $SCENE_LIST;
do

    python3 visualize_rgb_difference.py \
    models/$SCENE/train/ours_30000/renders \
    models/$SCENE/train/ours_30000/gt \
    -o models/$SCENE/train/ours_30000/rgb_l1_difference/renders \
    --percentile-max 99 \
    --workers 4
done
