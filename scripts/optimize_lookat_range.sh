#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
START_INDEX=${START_INDEX:-150}
END_INDEX=${END_INDEX:-200}
SCENE=${SCENE:-bicycle}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/trajectory_train.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/trajectory_batches}
NEXT_SCRIPT=${NEXT_SCRIPT:-}
DRY_RUN=${DRY_RUN:-0}
VQA_SCRIPT=${VQA_SCRIPT:-/home/jiyoung.park/Code/FAST-VQA-and-FasterVQA/vqa_score.sh}
VQA_MODEL=${VQA_MODEL:-FAST-VQA}

if (( START_INDEX < 0 || END_INDEX < START_INDEX )); then
    echo "Invalid range: START_INDEX=$START_INDEX END_INDEX=$END_INDEX" >&2
    exit 2
fi
if [[ "$NEXT_SCRIPT" == *$'\n'* ]]; then
    echo "NEXT_SCRIPT must be one executable path" >&2
    exit 2
fi

cd "$REPO_ROOT"

CONFIG_PATH=$(realpath "$CONFIG")
CAMERA_ROOT=$(realpath "cameras/$SCENE")
OUTPUT_ROOT_PATH=$(realpath -m "$OUTPUT_ROOT")
BATCH_NAME=$(date -u +'%Y-%m-%d_%H-%M-%S')_$(printf '%04d-%04d' "$START_INDEX" "$END_INDEX")
BATCH_DIR="$OUTPUT_ROOT_PATH/$BATCH_NAME"
MANIFEST_PATH="$BATCH_DIR/manifest.tsv"
VQA_CSV_PATH="$BATCH_DIR/vqa_scores.csv"

VQA_SCRIPT=$(realpath "$VQA_SCRIPT")
if [[ ! -x "$VQA_SCRIPT" ]]; then
    echo "VQA_SCRIPT is not executable: $VQA_SCRIPT" >&2
    exit 2
fi

if [[ -n "$NEXT_SCRIPT" ]]; then
    NEXT_SCRIPT=$(realpath "$NEXT_SCRIPT")
    if [[ ! -x "$NEXT_SCRIPT" ]]; then
        echo "NEXT_SCRIPT is not executable: $NEXT_SCRIPT" >&2
        exit 2
    fi
fi

mkdir -p "$BATCH_DIR"
printf 'index\tinput_path\toutput_path\tfinal_trajectory\n' > "$MANIFEST_PATH"
printf 'index,score_0,score_200\n' > "$VQA_CSV_PATH"
printf '%s\n' "$BATCH_DIR" > "$OUTPUT_ROOT_PATH/latest_batch.txt"

echo "BATCH_OUTPUT_PATH=$BATCH_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "VQA_CSV_PATH=$VQA_CSV_PATH"

for (( index=START_INDEX; index<=END_INDEX; index++ )); do
    camera_id=$(printf '%04d' "$index")
    camera_path="$CAMERA_ROOT/$camera_id.lookat"
    output_path="$BATCH_DIR/$camera_id"
    log_path="$BATCH_DIR/$camera_id.log"

    if [[ ! -f "$camera_path" ]]; then
        echo "Missing camera file: $camera_path" >&2
        exit 1
    fi

    echo "[$camera_id] INPUT_PATH=$camera_path"
    echo "[$camera_id] OUTPUT_PATH=$output_path"

    command=(
        "$PYTHON_BIN" optimize_trajectory.py
        --config "$CONFIG_PATH"
        "trajectory.path=$camera_path"
        "output.directory=$BATCH_DIR"
        "output.run_name=\"$camera_id\""
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[%s] COMMAND=' "$camera_id"
        printf ' %q' "${command[@]}"
        printf '\n'
        for iteration in 0 200; do
            iteration_id=$(printf '%06d' "$iteration")
            video_path="$output_path/previews/iteration_$iteration_id/trajectory_rgb.mp4"
            printf '[%s] VQA_COMMAND=' "$camera_id"
            printf ' %q' \
                "$VQA_SCRIPT" -v "$video_path" -m "$VQA_MODEL"
            printf '\n'
        done
        continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$log_path"

    final_trajectory=$(find "$output_path" -maxdepth 1 -type f -name 'trajectory_*.npz' -print | sort | tail -n 1)
    if [[ -z "$final_trajectory" ]]; then
        echo "No final trajectory was produced in $output_path" >&2
        exit 1
    fi
    final_trajectory=$(realpath "$final_trajectory")
    printf '%s\t%s\t%s\t%s\n' \
        "$camera_id" "$camera_path" "$output_path" "$final_trajectory" \
        >> "$MANIFEST_PATH"
    printf '%s\n' "$output_path" > "$OUTPUT_ROOT_PATH/latest_output.txt"

    echo "[$camera_id] FINAL_TRAJECTORY=$final_trajectory"

    score_0=
    score_200=
    for iteration in 0 200; do
        iteration_id=$(printf '%06d' "$iteration")
        video_path="$output_path/previews/iteration_$iteration_id/trajectory_rgb.mp4"
        vqa_log_path="$BATCH_DIR/${camera_id}_iteration_${iteration_id}_vqa.log"
        if [[ ! -f "$video_path" ]]; then
            echo "Missing VQA input video: $video_path" >&2
            exit 1
        fi
        video_path=$(realpath "$video_path")

        echo "[$camera_id] VQA_ITERATION=$iteration VIDEO_PATH=$video_path"
        vqa_output=$(
            CUDA_VISIBLE_DEVICES="$GPU" \
                "$VQA_SCRIPT" -v "$video_path" -m "$VQA_MODEL" \
                2>&1 | tee "$vqa_log_path"
        )
        vqa_score=$(printf '%s\n' "$vqa_output" | tail -n 1)
        if [[ ! "$vqa_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "Could not parse VQA score from $vqa_log_path: $vqa_score" >&2
            exit 1
        fi
        if [[ "$iteration" == "0" ]]; then
            score_0=$vqa_score
        else
            score_200=$vqa_score
        fi
        echo "[$camera_id] VQA_SCORE iteration=$iteration score=$vqa_score"
    done
    printf '%s,%s,%s\n' "$camera_id" "$score_0" "$score_200" \
        >> "$VQA_CSV_PATH"

    if [[ -n "$NEXT_SCRIPT" ]]; then
        "$NEXT_SCRIPT" "$output_path"
    fi
done

echo "BATCH_COMPLETE=$BATCH_DIR"
