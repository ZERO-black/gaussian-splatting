#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arrange videos in a row-major grid with ffmpeg."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged.mp4"),
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS; default uses the first input video's FPS.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ),
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def require_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable was not found: {name}")
    return executable


def probe_video(path, ffprobe):
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    if fps <= 0:
        raise ValueError(f"Invalid FPS in {path}: {stream['avg_frame_rate']}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def even(value):
    value = int(value)
    return value if value % 2 == 0 else value - 1


def grid_filter(metadata, fps, rows, cols):
    cell_width = even(min(item["width"] for item in metadata))
    cell_height = even(min(item["height"] for item in metadata))
    filters = []
    labels = []
    for index in range(len(metadata)):
        label = f"v{index}"
        filters.append(
            f"[{index}:v]fps={fps:.12g},"
            f"scale={cell_width}:{cell_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    layout = "|".join(
        f"{(index % cols) * cell_width}_{(index // cols) * cell_height}"
        for index in range(len(metadata))
    )
    canvas_width = cols * cell_width
    canvas_height = rows * cell_height
    filters.append(
        f"{''.join(labels)}xstack=inputs={len(labels)}:layout={layout}:"
        "shortest=1:fill=black[stacked]"
    )
    filters.append(
        f"[stacked]pad={canvas_width}:{canvas_height}:0:0:black[outv]"
    )
    return ";".join(filters)


def main():
    args = parse_args()
    if len(args.inputs) < 2:
        raise ValueError("At least two input videos are required")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if len(args.inputs) > args.rows * args.cols:
        raise ValueError(
            f"{len(args.inputs)} videos do not fit in a "
            f"{args.rows}x{args.cols} grid"
        )
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be in [0, 51]")

    inputs = [path.expanduser().resolve() for path in args.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")

    output = args.output.expanduser().resolve()
    if output in inputs:
        raise ValueError("Output path must differ from every input path")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")
    metadata = [probe_video(path, ffprobe) for path in inputs]
    fps = args.fps or metadata[0]["fps"]

    filter_graph = grid_filter(metadata, fps, args.rows, args.cols)

    command = [ffmpeg, "-hide_banner", "-loglevel", "warning"]
    command.append("-y" if args.overwrite else "-n")
    for path in inputs:
        command.extend(("-i", str(path)))
    command.extend((
        "-filter_complex", filter_graph,
        "-map", "[outv]",
        "-an",
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ))

    print(
        f"Merging {len(inputs)} videos into "
        f"a {args.rows}x{args.cols} grid -> {output}"
    )
    subprocess.run(command, check=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
