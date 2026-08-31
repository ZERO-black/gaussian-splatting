mkdir -p output/trajectory_batch_logs

START_INDEX=1   END_INDEX=34  GPU=0 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu0_0001-0034.log 2>&1 &

START_INDEX=35  END_INDEX=68  GPU=1 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu1_0035-0068.log 2>&1 &

START_INDEX=69  END_INDEX=101 GPU=2 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu2_0069-0101.log 2>&1 &

START_INDEX=102 END_INDEX=134 GPU=3 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu3_0102-0134.log 2>&1 &

START_INDEX=135 END_INDEX=167 GPU=4 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu4_0135-0167.log 2>&1 &

START_INDEX=168 END_INDEX=200 GPU=5 OUTPUT_ROOT=output/trajectory_batches_200 \
  nohup scripts/optimize_lookat_range.sh > output/trajectory_batch_logs/gpu5_0168-0200.log 2>&1 &

wait