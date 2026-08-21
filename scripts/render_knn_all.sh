#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"
RESULT_DIR="logs"
METRICS=(
    kth
    mean
    # kth_over_max_scale
    # mean_over_max_scale
    # kth_over_mean_scale
    # mean_over_mean_scale
)

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"
    # train without eval
    nvidia-smi > $RESULT_DIR/train_$SCENE.log
    CUDA_VISIBLE_DEVICES=3 python render_knn.py \
    -m models/$SCENE \
    --iteration 30000 \
    --knn_k 10 \
    --knn_percentile_min 1 \
    --knn_percentile_max 99.5 \
    --metrics "${METRICS[@]}" \
    >> $RESULT_DIR/render_knn_$SCENE.log

    python3 combine_render_folders.py \
    models/$SCENE/train/ours_30000/renders \
    models/$SCENE/train/ours_30000/ssim_dissimilarity/renders \
    models/$SCENE/train/ours_30000/knn_kth_k10/renders \
    models/$SCENE/train/ours_30000/knn_mean_k10/renders \
    -o models/$SCENE/train/ours_30000/combined \
    --labels RGB SSIM 10th mean \
    --rows 2 \
    --cols 2
done
