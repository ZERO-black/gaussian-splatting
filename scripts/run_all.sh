#!/bin/bash

set -euo pipefail

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="bicycle"
ITERATION=30000
K=10
GPU=2
REFERENCE_ROOT="models"
TARGET_ROOT="models/view_dropout_llff4"

# These metrics can all be evaluated from point_cloud_knn.ply. The projected
# and view-perpendicular metrics reuse xyz/covariance from that same PLY.
KNN_METRICS=(
    kth
    mean
    # kth_over_max_scale
    # mean_over_max_scale
    # kth_over_mean_scale
    # mean_over_mean_scale
    kth_over_camera_depth
    mean_over_camera_depth
    max_projected_gap_over_footprint
    mean_projected_gap_over_footprint
    max_projected_gap_over_major_axis
    mean_projected_gap_over_major_axis
    max_projected_gap_over_minor_axis
    mean_projected_gap_over_minor_axis
    max_view_perp_over_support
    mean_view_perp_over_support
)

# covariance.py stores these properties in a separate annotated PLY.
COVARIANCE_METRICS=(
    long_axis_consistency
    short_axis_consistency
)

for SCENE in $SCENE_LIST; do
    REFERENCE_MODEL="$REFERENCE_ROOT/$SCENE"
    TARGET_MODEL="$TARGET_ROOT/$SCENE"

    echo "Running all metrics: $SCENE"
    echo "Reference: $REFERENCE_MODEL"
    echo "Target:    $TARGET_MODEL"

    for MODEL_PATH in "$REFERENCE_MODEL" "$TARGET_MODEL"; do
        KNN_PLY="$MODEL_PATH/knn_analysis/iteration_$ITERATION/point_cloud_knn.ply"
        COVARIANCE_PLY="$MODEL_PATH/covariance_analysis/iteration_$ITERATION/point_cloud_covariance.ply"

        python analysis/knn.py \
            -m "$MODEL_PATH" \
            --iteration "$ITERATION" \
            --k "$K"

        # python analysis/covariance.py \
        #     -m "$MODEL_PATH" \
        #     --iteration "$ITERATION" \
        #     --k "$K"

        CUDA_VISIBLE_DEVICES="$GPU" python analysis/render_knn.py \
            -m "$MODEL_PATH" \
            --iteration "$ITERATION" \
            --analysis_ply "$KNN_PLY" \
            --knn_k "$K" \
            --percentile_min 1 \
            --percentile_max 99.5 \
            --metrics "${KNN_METRICS[@]}"

        # CUDA_VISIBLE_DEVICES="$GPU" python analysis/render_knn.py \
        #     -m "$MODEL_PATH" \
        #     --iteration "$ITERATION" \
        #     --analysis_ply "$COVARIANCE_PLY" \
        #     --knn_k "$K" \
        #     --percentile_min 1 \
        #     --percentile_max 99.5 \
        #     --metrics "${COVARIANCE_METRICS[@]}"
    done

    for METRIC in "${KNN_METRICS[@]}"; do
        CUDA_VISIBLE_DEVICES="$GPU" python analysis/validate_knn_metric.py \
            --reference_model "$REFERENCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --iteration "$ITERATION" \
            --knn_k "$K" \
            --metric "$METRIC" \
            --eval \
            --eval_split test \
            --percentile_min 1 \
            --percentile_max 99.5
    done
done
