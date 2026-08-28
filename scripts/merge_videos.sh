CUDA_VISIBLE_DEVICES=2 python scripts/merge_videos.py \
  video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
  --rows 2 \
  --cols 2 \
  -o merged.mp4