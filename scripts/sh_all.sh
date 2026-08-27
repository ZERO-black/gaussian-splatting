#!/bin/bash

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"

for SCENE in $SCENE_LIST;
do
    echo "Running: $SCENE"

   CUDA_VISIBLE_DEVICES=3 python analysis/sh_analysis.py \
  -m models/$SCENE \
  --iteration 30000 \
  --num-directions 256 \
  --clip-percentiles 0.5 99.5
done
