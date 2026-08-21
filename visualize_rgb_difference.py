"""Visualize per-pixel L1 RGB differences with a dataset-wide color scale."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_RGB_ERROR_SUM = 3 * 255


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize mean absolute RGB error between matching render and GT images."
        )
    )
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument(
        "--percentile-min", default=0.0, type=float,
        help="Dataset-wide lower normalization percentile (default: 0).",
    )
    parser.add_argument(
        "--percentile-max", default=99.0, type=float,
        help="Dataset-wide upper normalization percentile (default: 99).",
    )
    parser.add_argument(
        "--value-min", type=float,
        help="Optional fixed lower RGB-error bound in [0, 1].",
    )
    parser.add_argument(
        "--value-max", type=float,
        help="Optional fixed upper RGB-error bound in [0, 1].",
    )
    parser.add_argument(
        "--workers", default=4, type=int,
        help="Parallel image loading/saving workers (default: 4).",
    )
    return parser.parse_args()


def image_files(folder):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError("Image folder not found: {}".format(folder))
    return {
        path.name: path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def load_rgb(path):
    with Image.open(str(path)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def rgb_error_sum(render_path, gt_path):
    rendered = load_rgb(render_path)
    gt = load_rgb(gt_path)
    if rendered.shape != gt.shape:
        raise ValueError(
            "Image shape mismatch for {} and {}: {} versus {}".format(
                render_path, gt_path, rendered.shape, gt.shape
            )
        )
    difference = np.abs(rendered.astype(np.int16) - gt.astype(np.int16))
    return difference.sum(axis=2, dtype=np.int16)


def analyze_pair(pair):
    filename, render_path, gt_path = pair
    error_sum = rgb_error_sum(render_path, gt_path)
    histogram = np.bincount(
        error_sum.reshape(-1), minlength=MAX_RGB_ERROR_SUM + 1
    ).astype(np.int64, copy=False)
    return filename, histogram, int(error_sum.sum(dtype=np.int64)), error_sum.size


def histogram_percentile(histogram, percentile):
    sample_count = int(histogram.sum())
    if sample_count == 0:
        raise ValueError("Cannot compute a percentile from an empty histogram")
    rank = percentile / 100.0 * (sample_count - 1)
    index = int(np.searchsorted(np.cumsum(histogram), rank, side="right"))
    return index / float(MAX_RGB_ERROR_SUM)


def heatmap(values):
    """Apply the blue-cyan-yellow-red colormap used by KNN visualization."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=2)


def save_heatmap(image, output_path):
    array = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(str(output_path))


def render_pair(task):
    pair, output_dir, lower, upper = task
    filename, render_path, gt_path = pair
    error = rgb_error_sum(render_path, gt_path).astype(np.float32)
    error /= float(MAX_RGB_ERROR_SUM)
    normalized = np.clip((error - lower) / (upper - lower), 0.0, 1.0)
    save_heatmap(heatmap(normalized), output_dir / filename)
    return filename


def save_colorbar(output_path):
    gradient = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    gradient = np.repeat(gradient[np.newaxis, :], 32, axis=0)
    save_heatmap(heatmap(gradient), output_path)


def validate_args(args):
    if not 0 <= args.percentile_min < args.percentile_max <= 100:
        raise ValueError(
            "Expected 0 <= --percentile-min < --percentile-max <= 100"
        )
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    for name, value in (("--value-min", args.value_min), ("--value-max", args.value_max)):
        if value is not None and not 0 <= value <= 1:
            raise ValueError("{} must be in [0, 1]".format(name))
    if (
        args.value_min is not None and args.value_max is not None
        and args.value_max <= args.value_min
    ):
        raise ValueError("--value-max must be greater than --value-min")


def main():
    args = parse_args()
    validate_args(args)
    render_files = image_files(args.render_dir)
    gt_files = image_files(args.gt_dir)
    filenames = sorted(set(render_files) & set(gt_files))
    skipped_filenames = sorted((set(render_files) | set(gt_files)) - set(filenames))
    if not filenames:
        raise ValueError("Render and GT folders have no matching image filenames")
    if skipped_filenames:
        print("Skipping {} unmatched files".format(len(skipped_filenames)))
    pairs = [
        (filename, render_files[filename], gt_files[filename])
        for filename in filenames
    ]

    histogram = np.zeros(MAX_RGB_ERROR_SUM + 1, dtype=np.int64)
    total_error_sum = 0
    total_pixels = 0
    per_image_l1 = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (filename, image_histogram, error_sum, pixel_count) in enumerate(
            executor.map(analyze_pair, pairs), 1
        ):
            histogram += image_histogram
            total_error_sum += error_sum
            total_pixels += pixel_count
            per_image_l1[filename] = error_sum / float(
                pixel_count * MAX_RGB_ERROR_SUM
            )
            print(
                "Analyzing RGB difference: {}/{}".format(index, len(pairs)),
                end="\r", flush=True,
            )
    print()

    lower = (
        args.value_min if args.value_min is not None
        else histogram_percentile(histogram, args.percentile_min)
    )
    upper = (
        args.value_max if args.value_max is not None
        else histogram_percentile(histogram, args.percentile_max)
    )
    if upper <= lower:
        raise ValueError(
            "RGB visualization maximum ({}) must be greater than minimum ({})".format(
                upper, lower
            )
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(pair, output_dir, lower, upper) for pair in pairs]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(render_pair, tasks), 1):
            print(
                "Saving RGB difference: {}/{}".format(index, len(tasks)),
                end="\r", flush=True,
            )
    print()

    visualization_dir = output_dir.parent
    save_colorbar(visualization_dir / "colormap.png")
    metadata = {
        "render_dir": str(args.render_dir.expanduser().resolve()),
        "gt_dir": str(args.gt_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "metric": "mean(abs(render_rgb - gt_rgb))",
        "rgb_range": [0.0, 1.0],
        "normalization_min": lower,
        "normalization_max": upper,
        "percentile_min": None if args.value_min is not None else args.percentile_min,
        "percentile_max": None if args.value_max is not None else args.percentile_max,
        "colormap": "blue-cyan-yellow-red",
        "low_color": "blue",
        "high_color": "red",
        "mean_rgb_l1": total_error_sum / float(total_pixels * MAX_RGB_ERROR_SUM),
        "image_count": len(filenames),
        "skipped_filenames": skipped_filenames,
        "per_image_rgb_l1": per_image_l1,
    }
    metadata_path = visualization_dir / "rgb_difference_metadata.json"
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(
        "RGB L1 range [{:.6g}, {:.6g}], mean {:.6g}".format(
            lower, upper, metadata["mean_rgb_l1"]
        )
    )
    print("RGB difference visualization written to {}".format(output_dir))


if __name__ == "__main__":
    main()
