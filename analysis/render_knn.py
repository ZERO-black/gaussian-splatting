from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
from tqdm import tqdm

from analysis.knn import (
    STATIC_METRICS,
    distribution_quantile_label,
    validate_distribution_quantiles,
)
from analysis.metric_histogram import (
    build_histogram_spec,
    sample_values,
    save_single_histogram,
    subsample_values,
)
from analysis.metric_core import (
    ALL_METRICS,
    CAMERA_DEPTH_METRICS,
    COVARIANCE_METRICS,
    DEFAULT_RENDER_METRICS,
    METRIC_DESCRIPTIONS,
    VIEW_DEPENDENT_METRICS,
    colorize,
    compute_metric_values,
    metric_is_available,
    metric_quantiles,
    normalization_bounds,
    prepare_metric_context,
    valid_metric_mask,
    valid_metric_samples,
)
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except Exception:
    SEPARATE_SH = False


def default_input_ply(model_path, iteration, metrics):
    """Choose the PLY source implied by the requested metric set."""
    needs_knn = any(
        metric in STATIC_METRICS or metric in CAMERA_DEPTH_METRICS
        for metric in metrics
    )
    needs_covariance = any(metric in COVARIANCE_METRICS for metric in metrics)

    if needs_knn and needs_covariance:
        raise ValueError(
            "Requested metrics require both KNN and covariance annotations. "
            "Pass --analysis_ply pointing to a PLY that contains both property sets, "
            "or render the groups separately."
        )

    if needs_covariance:
        return os.path.join(
            model_path,
            "covariance_analysis",
            f"iteration_{iteration}",
            "point_cloud_covariance.ply",
        )

    if needs_knn:
        return os.path.join(
            model_path,
            "knn_analysis",
            f"iteration_{iteration}",
            "point_cloud_knn.ply",
        )

    # Pure pair metrics only need xyz/covariance and can use the trained PLY directly.
    return os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )


def save_colormap(output_dir, dtype, device):
    values = torch.linspace(0.0, 1.0, 512, dtype=dtype, device=device)
    image = colorize(values, 0.0, 1.0)
    image = image.transpose(0, 1).unsqueeze(1).repeat(1, 32, 1)
    torchvision.utils.save_image(image, os.path.join(output_dir, "colormap.png"))


def metric_bounds(values, valid, args):
    return normalization_bounds(
        values,
        valid,
        args.percentile_min,
        args.percentile_max,
        args.value_min,
        args.value_max,
    )


def render_metric_split(
    model_path,
    split_name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    metric,
    k,
    context,
    args,
):
    output_dir = os.path.join(
        model_path,
        split_name,
        f"ours_{iteration}",
        f"metric_{metric}_k{k}",
    )
    render_dir = os.path.join(output_dir, "renders")
    makedirs(render_dir, exist_ok=True)
    save_colormap(output_dir, background.dtype, background.device)

    is_view_dependent = metric in VIEW_DEPENDENT_METRICS
    metadata = {
        "metric": metric,
        "description": METRIC_DESCRIPTIONS[metric],
        "k": k,
        "input_ply": args.analysis_ply,
        "colormap": "blue-cyan-yellow-red",
        "value_transform": "linear",
        "normalization_population": "valid and finite Gaussians",
        "normalization_scope": "per_view" if is_view_dependent else "global_scene",
        "percentile_min": args.percentile_min if args.value_min is None else None,
        "percentile_max": args.percentile_max if args.value_max is None else None,
        "per_view_normalization": {},
    }

    static_values = static_valid = static_lower = static_upper = None
    threshold_variants = []
    histogram_chunks = []
    if not is_view_dependent:
        static_values, static_valid = compute_metric_values(
            gaussians, None, metric, k, context
        )
        static_lower, static_upper = metric_bounds(static_values, static_valid, args)
        metadata["normalization_min"] = static_lower
        metadata["normalization_max"] = static_upper
        metadata["valid_gaussians"] = int(
            valid_metric_mask(static_values, static_valid).sum().item()
        )
        histogram_chunks.append(
            sample_values(
                static_values,
                static_valid,
                args.histogram_max_samples,
                seed=args.histogram_seed,
            )
        )

        if args.threshold_quantiles:
            samples = valid_metric_samples(static_values, static_valid)
            threshold_values = metric_quantiles(samples, args.threshold_quantiles)
            for quantile, threshold in zip(
                args.threshold_quantiles, threshold_values
            ):
                label = distribution_quantile_label(quantile)
                variant_dir = os.path.join(output_dir, f"threshold_{label}")
                variant_render_dir = os.path.join(variant_dir, "renders")
                makedirs(variant_render_dir, exist_ok=True)
                save_colormap(variant_dir, background.dtype, background.device)
                threshold_variants.append({
                    "quantile": quantile,
                    "label": label,
                    "threshold": float(threshold),
                    "output_dir": variant_dir,
                    "render_dir": variant_render_dir,
                })
                plateau_count = int(
                    (samples <= float(threshold)).sum().item()
                )
                threshold_variants[-1]["plateau_count"] = plateau_count
                threshold_variants[-1]["plateau_fraction"] = (
                    plateau_count / samples.numel()
                )
                print(
                    f"  {label}: threshold={float(threshold):.9g}, "
                    f"plateau={plateau_count:,}/{samples.numel():,} "
                    f"({plateau_count / samples.numel():.2%})"
                )
            del samples

            metadata["lower_threshold_plateaus"] = {
                variant["label"]: {
                    "quantile": variant["quantile"],
                    "threshold": variant["threshold"],
                    "plateau_count": variant["plateau_count"],
                    "plateau_fraction": variant["plateau_fraction"],
                    "directory": os.path.relpath(variant["output_dir"], output_dir),
                }
                for variant in threshold_variants
            }
            metadata["threshold_rule"] = "max(metric_value, quantile_value)"

    for index, view in enumerate(tqdm(views, desc=f"Rendering {metric} ({split_name})")):
        if is_view_dependent:
            values, valid = compute_metric_values(gaussians, view, metric, k, context)
            lower, upper = metric_bounds(values, valid, args)
            metadata["per_view_normalization"][f"{index:05d}.png"] = {
                "normalization_min": lower,
                "normalization_max": upper,
                "valid_gaussians": int(
                    valid_metric_mask(values, valid).sum().item()
                ),
                "camera": getattr(view, "image_name", str(index)),
            }
            histogram_chunks.append(
                sample_values(
                    values,
                    valid,
                    args.histogram_max_samples,
                    seed=args.histogram_seed + index,
                )
            )
        else:
            values, valid = static_values, static_valid
            lower, upper = static_lower, static_upper

        colors = colorize(values, lower, upper)
        image = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=SEPARATE_SH,
            override_color=colors,
        )["render"]

        if train_test_exp:
            image = image[..., image.shape[-1] // 2 :]

        torchvision.utils.save_image(
            image, os.path.join(render_dir, f"{index:05d}.png")
        )
        del colors, image

        for variant in threshold_variants:
            thresholded_values = torch.clamp_min(values, variant["threshold"])
            thresholded_colors = colorize(thresholded_values, lower, upper)
            thresholded_image = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=False,
                separate_sh=SEPARATE_SH,
                override_color=thresholded_colors,
            )["render"]
            if train_test_exp:
                thresholded_image = thresholded_image[
                    ..., thresholded_image.shape[-1] // 2 :
                ]
            torchvision.utils.save_image(
                thresholded_image,
                os.path.join(variant["render_dir"], f"{index:05d}.png"),
            )
            del thresholded_values, thresholded_colors, thresholded_image

    with open(os.path.join(output_dir, "normalization.json"), "w") as stream:
        json.dump(metadata, stream, indent=2)

    for variant in threshold_variants:
        variant_metadata = {
            "metric": metric,
            "description": METRIC_DESCRIPTIONS[metric],
            "k": k,
            "input_ply": args.analysis_ply,
            "threshold_quantile": variant["quantile"],
            "threshold_value": variant["threshold"],
            "threshold_rule": "max(metric_value, threshold_value)",
            "plateau_count": variant["plateau_count"],
            "plateau_fraction": variant["plateau_fraction"],
            "normalization_min": static_lower,
            "normalization_max": static_upper,
            "normalization_source": "original unthresholded scene distribution",
            "colormap": "blue-cyan-yellow-red",
        }
        with open(
            os.path.join(variant["output_dir"], "normalization.json"), "w"
        ) as stream:
            json.dump(variant_metadata, stream, indent=2)

    histogram_chunks = [chunk for chunk in histogram_chunks if chunk.numel()]
    if histogram_chunks:
        histogram_samples = subsample_values(
            torch.cat(histogram_chunks),
            max_samples=args.histogram_max_total_samples,
            seed=args.histogram_seed + len(views),
        )
        histogram_spec = build_histogram_spec(
            histogram_samples,
            bins=args.histogram_bins,
            percentile_min=args.histogram_percentile_min,
            percentile_max=args.histogram_percentile_max,
        )
        save_single_histogram(
            os.path.join(output_dir, "value_histogram.png"),
            os.path.join(output_dir, "value_histogram.json"),
            metric,
            histogram_samples,
            histogram_spec,
            title=f"{metric} distribution ({split_name}, K={k})",
            metadata={
                "split": split_name,
                "k": k,
                "sampling": "valid finite Gaussians, random without replacement",
                "sampling_seed": args.histogram_seed,
                "maximum_samples_per_view": args.histogram_max_samples,
                "maximum_total_samples": args.histogram_max_total_samples,
            },
        )


def render_sets(dataset, iteration, pipeline, args):
    with torch.no_grad():
        loaded_iteration = iteration
        if loaded_iteration == -1:
            loaded_iteration = searchForMaxIteration(
                os.path.join(dataset.model_path, "point_cloud")
            )

        metrics = list(args.metrics or DEFAULT_RENDER_METRICS)
        input_ply = args.analysis_ply
        if input_ply is None:
            input_ply = default_input_ply(dataset.model_path, loaded_iteration, metrics)
        input_ply = os.path.abspath(input_ply)
        if not os.path.isfile(input_ply):
            raise FileNotFoundError(f"Metric input PLY not found: {input_ply}")
        args.analysis_ply = input_ply

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=loaded_iteration,
            shuffle=False,
            load_ply_path=input_ply,
        )

        unavailable = [
            metric for metric in metrics
            if not metric_is_available(gaussians, metric, args.knn_k)
        ]
        if unavailable:
            raise ValueError(
                f"Metrics unavailable from {input_ply} for K={args.knn_k}: {unavailable}"
            )

        context = prepare_metric_context(gaussians, metrics, args.knn_k)
        # Undefined metric pixels must be visually unmistakable high risk.
        background = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
        )

        for metric in metrics:
            print(f"\n{metric}: {METRIC_DESCRIPTIONS[metric]}")

            if not args.skip_train:
                render_metric_split(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    scene.getTrainCameras(),
                    gaussians,
                    pipeline,
                    background,
                    dataset.train_test_exp,
                    metric,
                    args.knn_k,
                    context,
                    args,
                )

            if not args.skip_test:
                render_metric_split(
                    dataset.model_path,
                    "test",
                    scene.loaded_iter,
                    scene.getTestCameras(),
                    gaussians,
                    pipeline,
                    background,
                    dataset.train_test_exp,
                    metric,
                    args.knn_k,
                    context,
                    args,
                )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render exploratory Gaussian quality metrics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--analysis_ply",
        default=None,
        help="Metric-annotated PLY.",
    )
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=None)
    parser.add_argument(
        "--threshold_quantiles",
        "--threshold-quantiles",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Render extra lower-plateau variants. Values are distribution "
            "probabilities in [0, 1]; for example 0.9 uses the 90%% quantile."
        ),
    )
    parser.add_argument(
        "--percentile_min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--percentile_max",
        type=float,
        default=99.5,
    )
    parser.add_argument("--value_min", type=float, default=None)
    parser.add_argument("--value_max", type=float, default=None)
    parser.add_argument("--histogram_bins", type=int, default=160)
    parser.add_argument("--histogram_percentile_min", type=float, default=0.1)
    parser.add_argument("--histogram_percentile_max", type=float, default=99.9)
    parser.add_argument(
        "--histogram_max_samples",
        type=int,
        default=200_000,
        help="Maximum valid samples per view; 0 keeps every valid finite value.",
    )
    parser.add_argument(
        "--histogram_seed",
        type=int,
        default=0,
        help="Base seed for reproducible random histogram sampling.",
    )
    parser.add_argument(
        "--histogram_max_total_samples",
        type=int,
        default=1_000_000,
        help="Final aggregate histogram sample cap; 0 keeps all per-view samples.",
    )

    args = get_combined_args(parser)
    args.threshold_quantiles = validate_distribution_quantiles(
        args.threshold_quantiles
    )
    if args.knn_k <= 0:
        raise ValueError("--knn_k must be positive")
    selected_metrics = list(args.metrics or DEFAULT_RENDER_METRICS)
    if args.threshold_quantiles:
        unsupported = [
            metric for metric in selected_metrics if metric not in STATIC_METRICS
        ]
        if unsupported:
            raise ValueError(
                "--threshold_quantiles currently requires static KNN metrics; "
                f"unsupported metrics: {unsupported}"
            )
    if not 0 <= args.percentile_min < args.percentile_max <= 100:
        raise ValueError("Expected 0 <= percentile_min < percentile_max <= 100")
    if args.value_min is not None and args.value_max is not None:
        if args.value_max <= args.value_min:
            raise ValueError("--value_max must be greater than --value_min")
    if (
        args.histogram_bins <= 0
        or args.histogram_max_samples < 0
        or args.histogram_max_total_samples < 0
    ):
        raise ValueError("Histogram bins must be positive and sample caps non-negative")
    if not (
        0
        <= args.histogram_percentile_min
        < args.histogram_percentile_max
        <= 100
    ):
        raise ValueError("Invalid histogram percentile range")

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)
