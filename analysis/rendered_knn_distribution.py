from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from analysis.knn import (
    distribution_quantile_label,
    ply_property_name,
    validate_distribution_quantiles,
)
from analysis.metric_core import colorize, metric_quantiles
from analysis.metric_histogram import (
    build_histogram_spec,
    save_single_histogram,
    subsample_values,
)
from gaussian_renderer import GaussianModel, render_knn_times_splat_radius
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.system_utils import searchForMaxIteration


DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render KNN times CUDA splat-radius maps for every scene "
            "camera and summarize their pixel distribution."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trajectory_train.yaml"),
        help="Trajectory optimizer config whose KNN settings should be reproduced.",
    )
    parser.add_argument(
        "--camera-split",
        choices=("train", "test", "all"),
        default=None,
        help="Default: trajectory.camera_split from the config.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUANTILES),
        help="Distribution probabilities in [0, 1], e.g. 0.9 for the 90%% quantile.",
    )
    parser.add_argument("--histogram-bins", type=int, default=200)
    parser.add_argument(
        "--histogram-quantile-min",
        type=float,
        default=0.001,
        help="Lower display-range quantile in [0, 1].",
    )
    parser.add_argument(
        "--histogram-quantile-max",
        type=float,
        default=0.999,
        help="Upper display-range quantile in [0, 1].",
    )
    parser.add_argument(
        "--max-samples-per-camera",
        type=int,
        default=100_000,
        help="Pixel sample cap per camera; 0 keeps every rendered pixel.",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=5_000_000,
        help="Final pixel sample cap; 0 keeps every collected sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Also save each rendered KNN x splat-radius map as a risk-color PNG.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <model>/knn_analysis/iteration_<n>/"
            "rendered_<metric>_k<k>_times_splat_radius."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args):
    args.quantiles = validate_distribution_quantiles(args.quantiles)
    if not args.quantiles:
        raise ValueError("At least one --quantiles value is required")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive")
    if args.max_samples_per_camera < 0 or args.max_total_samples < 0:
        raise ValueError("Sample caps must be non-negative")
    if not (
        0.0
        <= args.histogram_quantile_min
        < args.histogram_quantile_max
        <= 1.0
    ):
        raise ValueError("Histogram quantiles must satisfy 0 <= min < max <= 1")


def resolve_iteration(config):
    iteration = int(config.model.iteration)
    if iteration == -1:
        iteration = searchForMaxIteration(
            str(Path(config.model.model_path) / "point_cloud")
        )
    return iteration


def resolve_knn_ply(config, iteration):
    configured = config.knn.ply_path
    if configured is not None:
        path = Path(configured).expanduser()
    else:
        path = (
            Path(config.model.model_path)
            / "knn_analysis"
            / f"iteration_{iteration}"
            / "point_cloud_knn.ply"
        )
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KNN-annotated PLY not found: {path}")
    return path


def resolve_tail_threshold(config, knn_ply, iteration):
    configured = config.knn.tail_threshold
    if configured is not None:
        return float(configured), "config"

    summary_path = knn_ply.parent / "summary.json"
    if not summary_path.is_file():
        fallback = (
            Path(config.model.model_path)
            / "knn_analysis"
            / f"iteration_{iteration}"
            / "summary.json"
        )
        summary_path = fallback.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(
            "KNN summary.json was not found and knn.tail_threshold is null"
        )

    with summary_path.open() as stream:
        summary = json.load(stream)
    threshold = summary["metrics"][str(config.knn.k)][config.knn.metric][
        "tail_threshold"
    ]
    return float(threshold), str(summary_path)


def metadata_to_camera(camera, device):
    world_view = torch.as_tensor(
        getWorld2View2(camera.R, camera.T), dtype=torch.float32, device=device
    ).transpose(0, 1).contiguous()
    projection = getProjectionMatrix(
        znear=camera.znear,
        zfar=camera.zfar,
        fovX=camera.FoVx,
        fovY=camera.FoVy,
    ).transpose(0, 1).to(device)
    return MiniCam(
        camera.image_width,
        camera.image_height,
        camera.FoVy,
        camera.FoVx,
        camera.znear,
        camera.zfar,
        world_view,
        world_view @ projection,
    )


def selected_cameras(scene, split):
    if split == "train":
        return [("train", camera) for camera in scene.getTrainCameras()]
    if split == "test":
        return [("test", camera) for camera in scene.getTestCameras()]
    return [
        *(("train", camera) for camera in scene.getTrainCameras()),
        *(("test", camera) for camera in scene.getTestCameras()),
    ]


def camera_sample_caps(args, cameras):
    """Allocate the sample budget in proportion to each camera's pixel count."""
    pixel_counts = [
        int(camera.image_width) * int(camera.image_height)
        for _, camera in cameras
    ]
    total_pixels = sum(pixel_counts)
    caps = []
    for pixel_count in pixel_counts:
        cap = pixel_count
        if args.max_samples_per_camera > 0:
            cap = min(cap, args.max_samples_per_camera)
        if args.max_total_samples > 0:
            proportional_cap = math.ceil(
                args.max_total_samples * pixel_count / total_pixels
            )
            cap = min(cap, proportional_cap)
        caps.append(cap)
    return caps


def scalar_map_to_risk_rgb(knn_map, coverage):
    """Colorize foreground values and reserve pure red for uncovered pixels."""
    if knn_map.ndim != 3 or knn_map.shape[0] != 1:
        raise ValueError(
            f"Expected scalar map [1, H, W], got {tuple(knn_map.shape)}"
        )
    if coverage.shape != knn_map.shape:
        raise ValueError("coverage must have the same shape as knn_map")
    height, width = knn_map.shape[1:]
    values = knn_map.reshape(-1)
    covered = coverage.reshape(-1) > 0.0
    if covered.any():
        lower, upper = metric_quantiles(values[covered], [0.01, 0.99])
        if upper <= lower:
            upper = lower + torch.finfo(values.dtype).eps
    else:
        lower, upper = 0.0, 1.0
    rgb = colorize(values, lower, upper).transpose(0, 1).reshape(3, height, width)
    uncovered = (~covered).reshape(height, width)
    rgb[0][uncovered] = 1.0
    rgb[1][uncovered] = 0.0
    rgb[2][uncovered] = 0.0
    return rgb


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to rasterize the KNN/distance maps")

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = OmegaConf.load(config_path)
    safe_state(args.quiet)
    device = torch.device(config.runtime.device)
    iteration = resolve_iteration(config)
    knn_ply = resolve_knn_ply(config, iteration)
    tail_threshold, threshold_source = resolve_tail_threshold(
        config, knn_ply, iteration
    )

    gaussians = GaussianModel(config.model.sh_degree)
    scene = Scene(
        config.model,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        load_ply_path=str(knn_ply),
        load_camera_images=False,
    )
    property_name = ply_property_name(config.knn.metric, config.knn.k)
    raw_knn = gaussians.get_knn_metric(property_name)
    normalized_knn = torch.clamp(raw_knn / tail_threshold, 0.0, 1.0)
    normalized_knn.requires_grad_(False)

    split = args.camera_split or config.trajectory.camera_split
    cameras = selected_cameras(scene, split)
    if not cameras:
        raise ValueError(f"Camera split {split!r} is empty")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(config.model.model_path)
            / "knn_analysis"
            / f"iteration_{iteration}"
            / f"rendered_{config.knn.metric}_k{config.knn.k}_times_splat_radius"
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    if args.save_maps:
        map_dir.mkdir(exist_ok=True)

    background = torch.full(
        (3,), float(config.knn.background), dtype=torch.float32, device=device
    )
    sample_caps = camera_sample_caps(args, cameras)
    sample_chunks = []
    camera_stats = []
    full_pixel_count = 0

    with torch.no_grad():
        for index, (camera_split, metadata) in enumerate(
            tqdm(cameras, desc="Rendering KNN x splat-radius maps")
        ):
            camera = metadata_to_camera(metadata, device)
            render_output = render_knn_times_splat_radius(
                camera,
                gaussians,
                config.pipeline,
                background,
                normalized_knn,
            )
            knn_map = render_output["knn"]
            coverage = render_output["knn_coverage"]
            visible_radii = render_output["knn_splat_radii"]
            visible_radii = visible_radii[visible_radii > 0]
            flat = knn_map.reshape(-1)
            if not torch.isfinite(flat).all():
                raise ValueError(f"Rendered map {index} contains non-finite values")

            full_pixel_count += flat.numel()
            sample_chunks.append(
                subsample_values(flat, sample_caps[index], seed=args.seed + index)
            )
            camera_stats.append({
                "index": index,
                "split": camera_split,
                "camera": metadata.image_name,
                "pixel_count": flat.numel(),
                "min": float(flat.min().item()),
                "mean": float(flat.mean().item()),
                "max": float(flat.max().item()),
                "visible_gaussian_count": int(visible_radii.numel()),
                "mean_visible_splat_radius_px": (
                    float(visible_radii.float().mean().item())
                    if visible_radii.numel()
                    else None
                ),
                "max_visible_splat_radius_px": (
                    int(visible_radii.max().item())
                    if visible_radii.numel()
                    else None
                ),
            })
            if args.save_maps:
                torchvision.utils.save_image(
                    scalar_map_to_risk_rgb(knn_map, coverage),
                    map_dir / f"{index:05d}.png",
                )
            del camera, render_output, knn_map, coverage, visible_radii, flat

    samples = torch.cat(sample_chunks)
    samples = subsample_values(
        samples, args.max_total_samples, seed=args.seed + len(cameras)
    )
    quantile_values = metric_quantiles(samples, args.quantiles)
    quantiles = {
        distribution_quantile_label(probability): float(value)
        for probability, value in zip(args.quantiles, quantile_values)
    }
    sample_statistics = {
        "min": float(samples.min().item()),
        "mean": float(samples.mean().item()),
        "std": float(samples.std(unbiased=False).item()),
        "max": float(samples.max().item()),
    }

    histogram_spec = build_histogram_spec(
        samples,
        bins=args.histogram_bins,
        percentile_min=100.0 * args.histogram_quantile_min,
        percentile_max=100.0 * args.histogram_quantile_max,
    )
    histogram_metadata = {
        "value_definition": (
            "rasterized clamp(raw_knn / tail_threshold, 0, 1) * "
            "CUDA forward.cu radii[idx] in pixels"
        ),
        "splat_radius_source": (
            "submodules/diff-gaussian-rasterization/"
            "cuda_rasterizer/forward.cu radii[idx]"
        ),
        "camera_distance_normalization": False,
        "background_included": True,
        "background_value": float(config.knn.background),
        "camera_split": split,
        "camera_count": len(cameras),
        "full_rendered_pixel_count": full_pixel_count,
        "sample_count": int(samples.numel()),
        "configured_max_samples_per_camera": args.max_samples_per_camera,
        "max_total_samples": args.max_total_samples,
        "sampling_strategy": "pixel-count-proportional sampling across all cameras",
        "sampling_seed": args.seed,
        "sample_statistics": sample_statistics,
        "quantiles": quantiles,
    }
    save_single_histogram(
        output_dir / "histogram.png",
        output_dir / "histogram.json",
        "rendered_knn_times_splat_radius",
        samples,
        histogram_spec,
        title=(
            f"Rendered KNN x splat radius ({split}, "
            f"{len(cameras)} cameras)"
        ),
        x_label="Rendered normalized KNN x projected splat radius (pixels)",
        metadata=histogram_metadata,
    )

    summary = {
        **histogram_metadata,
        "config": str(config_path),
        "model_iteration": iteration,
        "knn_ply": str(knn_ply),
        "knn_property": property_name,
        "knn_tail_threshold": tail_threshold,
        "knn_tail_threshold_source": threshold_source,
        "camera_statistics": camera_stats,
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print(f"\nRendered distribution written to {output_dir}")
    print(f"Cameras: {len(cameras):,}; rendered pixels: {full_pixel_count:,}")
    print(f"Percentile samples: {samples.numel():,}")
    for label, value in quantiles.items():
        print(f"  {label}: {value:.9g}")


if __name__ == "__main__":
    main()
