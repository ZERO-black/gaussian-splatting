#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"
RESULT_DIR="logs"
K="20"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"

   CUDA_VISIBLE_DEVICES=3 \
    python analysis/covariance.py \
        -m models/$SCENE \
        --iteration 30000 \
        --k $K

   CUDA_VISIBLE_DEVICES=3 python analysis/render_knn.py \
  -m models/$SCENE \
  --iteration 30000 \
  --knn_k $K \
  --metrics long_axis_consistency short_axis_consistency \
  --percentile_min 1 \
  --percentile_max 99.5
done
