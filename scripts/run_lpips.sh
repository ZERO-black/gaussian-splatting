#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"

for SCENE in $SCENE_LIST;
do

CUDA_VISIBLE_DEVICES=3 python3 visualize_lpips.py \
  models/$SCENE/train/ours_30000/renders \
  models/$SCENE/train/ours_30000/gt \
  -o models/$SCENE/train/ours_30000/lpips_difference/renders \
  --net-type vgg \
  --percentile-max 99 \
  --device cuda \
  --normalization per-image
done
