"""Visualize spatial LPIPS differences with a dataset-wide color scale."""

import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lpipsPyTorch.modules.lpips import LPIPS


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize spatial LPIPS maps for matching render and GT images."
    )
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument(
        "--net-type", choices=("alex", "squeeze", "vgg"), default="vgg",
        help="LPIPS backbone; vgg matches this repository's metrics.py (default: vgg).",
    )
    parser.add_argument(
        "--normalization", choices=("per-image", "global", "fixed"),
        default="per-image",
        help=(
            "Color normalization: save each map immediately (per-image), use one "
            "dataset range (global), or use --value-min/max (fixed)."
        ),
    )
    parser.add_argument("--percentile-min", default=0.0, type=float)
    parser.add_argument("--percentile-max", default=99.0, type=float)
    parser.add_argument("--value-min", type=float)
    parser.add_argument("--value-max", type=float)
    parser.add_argument("--histogram-bins", default=16384, type=int)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:<index> (default: auto).",
    )
    parser.add_argument(
        "--workers", default=4, type=int,
        help="Parallel map colorization workers (default: 4).",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Parent for temporary float16 maps; defaults to the output parent.",
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


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_rgb_tensor(path, device):
    with Image.open(str(path)) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


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


def save_colorbar(output_path):
    gradient = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    gradient = np.repeat(gradient[np.newaxis, :], 32, axis=0)
    save_heatmap(heatmap(gradient), output_path)


def histogram_percentile(histogram, value_min, value_max, percentile):
    sample_count = int(histogram.sum())
    rank = percentile / 100.0 * (sample_count - 1)
    index = int(np.searchsorted(np.cumsum(histogram), rank, side="right"))
    if percentile == 0:
        return value_min
    if percentile == 100:
        return value_max
    bin_width = (value_max - value_min) / len(histogram)
    return value_min + (index + 0.5) * bin_width


def colorize_cached_map(task):
    cache_path, output_path, lower, upper = task
    values = np.load(str(cache_path), mmap_mode="r").astype(np.float32)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    save_heatmap(heatmap(normalized), output_path)
    return output_path.name


def validate_args(args):
    if not 0 <= args.percentile_min < args.percentile_max <= 100:
        raise ValueError(
            "Expected 0 <= --percentile-min < --percentile-max <= 100"
        )
    if args.histogram_bins <= 1 or args.workers <= 0:
        raise ValueError("--histogram-bins must exceed one and --workers must be positive")
    if (
        args.value_min is not None and args.value_max is not None
        and args.value_max <= args.value_min
    ):
        raise ValueError("--value-max must be greater than --value-min")
    if args.normalization == "fixed" and (
        args.value_min is None or args.value_max is None
    ):
        raise ValueError("fixed normalization requires --value-min and --value-max")


def compute_lpips_map(criterion, render_path, gt_path, device, filename):
    rendered = load_rgb_tensor(render_path, device)
    gt = load_rgb_tensor(gt_path, device)
    if rendered.shape != gt.shape:
        raise ValueError(
            "Image shape mismatch for {}: render {}, GT {}".format(
                filename, tuple(rendered.shape), tuple(gt.shape)
            )
        )
    scalar, spatial = criterion.forward_with_map(rendered, gt)
    spatial_array = spatial.squeeze(0).squeeze(0).float().cpu().numpy()
    if not np.isfinite(spatial_array).all():
        raise ValueError("LPIPS map for {} contains non-finite values".format(filename))
    return float(scalar.item()), spatial_array


def map_bounds(values, args):
    lower = (
        float(args.value_min) if args.value_min is not None
        else float(np.percentile(values, args.percentile_min))
    )
    upper = (
        float(args.value_max) if args.value_max is not None
        else float(np.percentile(values, args.percentile_max))
    )
    return lower, upper


def normalize_map(values, lower, upper):
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


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

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print("Loading {} LPIPS network on {}".format(args.net_type, device))
    criterion = LPIPS(net_type=args.net_type).eval().to(device)

    per_image_lpips = {}
    per_image_normalization = {}
    observed_min = float("inf")
    observed_max = float("-inf")

    lower = None
    upper = None
    if args.normalization in ("per-image", "fixed"):
        with torch.no_grad():
            for index, filename in enumerate(filenames, 1):
                scalar, spatial_array = compute_lpips_map(
                    criterion, render_files[filename], gt_files[filename],
                    device, filename,
                )
                observed_min = min(observed_min, float(spatial_array.min()))
                observed_max = max(observed_max, float(spatial_array.max()))
                per_image_lpips[filename] = scalar
                image_lower, image_upper = map_bounds(spatial_array, args)
                per_image_normalization[filename] = {
                    "min": image_lower, "max": image_upper,
                }
                save_heatmap(
                    heatmap(normalize_map(spatial_array, image_lower, image_upper)),
                    output_dir / filename,
                )
                print(
                    "Computing and saving LPIPS maps: {}/{}".format(
                        index, len(filenames)
                    ),
                    end="\r", flush=True,
                )
        print()
        if args.normalization == "fixed":
            lower, upper = float(args.value_min), float(args.value_max)
    else:
        cache_parent = (
            output_dir.parent if args.cache_dir is None
            else args.cache_dir.expanduser().resolve()
        )
        cache_parent.mkdir(parents=True, exist_ok=True)
        cached_maps = []
        with tempfile.TemporaryDirectory(
            prefix=".lpips_maps_", dir=str(cache_parent)
        ) as temp:
            temp_dir = Path(temp)
            with torch.no_grad():
                for index, filename in enumerate(filenames, 1):
                    scalar, spatial_array = compute_lpips_map(
                        criterion, render_files[filename], gt_files[filename],
                        device, filename,
                    )
                    cached_array = spatial_array.astype(np.float16)
                    if not np.isfinite(cached_array).all():
                        raise ValueError(
                            "LPIPS map for {} exceeds the float16 cache range".format(
                                filename
                            )
                        )
                    observed_min = min(observed_min, float(cached_array.min()))
                    observed_max = max(observed_max, float(cached_array.max()))
                    per_image_lpips[filename] = scalar
                    cache_path = temp_dir / "{:06d}.npy".format(index - 1)
                    np.save(str(cache_path), cached_array)
                    cached_maps.append((filename, cache_path))
                    print(
                        "Computing LPIPS maps: {}/{}".format(index, len(filenames)),
                        end="\r", flush=True,
                    )
            print()

            if observed_max <= observed_min:
                raise ValueError(
                    "All LPIPS map values are identical: {}".format(observed_min)
                )
            histogram = np.zeros(args.histogram_bins, dtype=np.int64)
            for _, cache_path in cached_maps:
                # np.histogram cannot represent many narrow bins with float16 edges.
                values = np.load(str(cache_path), mmap_mode="r").astype(np.float32)
                image_histogram, _ = np.histogram(
                    values, bins=args.histogram_bins,
                    range=(observed_min, observed_max),
                )
                histogram += image_histogram

            lower = (
                float(args.value_min) if args.value_min is not None
                else histogram_percentile(
                    histogram, observed_min, observed_max, args.percentile_min
                )
            )
            upper = (
                float(args.value_max) if args.value_max is not None
                else histogram_percentile(
                    histogram, observed_min, observed_max, args.percentile_max
                )
            )
            if upper <= lower:
                raise ValueError(
                    "LPIPS visualization maximum ({}) must exceed minimum ({})".format(
                        upper, lower
                    )
                )

            tasks = [
                (cache_path, output_dir / filename, lower, upper)
                for filename, cache_path in cached_maps
            ]
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for index, _ in enumerate(executor.map(colorize_cached_map, tasks), 1):
                    print(
                        "Saving LPIPS maps: {}/{}".format(index, len(tasks)),
                        end="\r", flush=True,
                    )
            print()

    visualization_dir = output_dir.parent
    save_colorbar(visualization_dir / "colormap.png")
    metadata = {
        "render_dir": str(args.render_dir.expanduser().resolve()),
        "gt_dir": str(args.gt_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "metric": "spatial LPIPS",
        "net_type": args.net_type,
        "normalization": args.normalization,
        "normalization_min": lower,
        "normalization_max": upper,
        "percentile_min": None if args.value_min is not None else args.percentile_min,
        "percentile_max": None if args.value_max is not None else args.percentile_max,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "colormap": "blue-cyan-yellow-red",
        "low_color": "blue",
        "high_color": "red",
        "mean_lpips": float(np.mean(list(per_image_lpips.values()))),
        "image_count": len(filenames),
        "skipped_filenames": skipped_filenames,
        "per_image_lpips": per_image_lpips,
        "per_image_normalization": per_image_normalization,
    }
    metadata_path = visualization_dir / "lpips_metadata.json"
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    if lower is not None:
        print("LPIPS normalization range [{:.6g}, {:.6g}]".format(lower, upper))
    print("Mean scalar LPIPS: {:.6g}".format(metadata["mean_lpips"]))
    print("LPIPS visualization written to {}".format(output_dir))


if __name__ == "__main__":
    main()
