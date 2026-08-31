CUDA_VISIBLE_DEVICES=2 python analysis/rendered_knn_distribution.py \
  --config configs/trajectory_train.yaml \
  --camera-split all \
  --quantiles 0.1 0.25 0.5 0.75 0.9 0.95 0.99