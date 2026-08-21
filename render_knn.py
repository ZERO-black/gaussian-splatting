#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, knn_colormap
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_analysis_set(
    model_path, name, iteration, views, gaussians, pipeline, background,
    train_test_exp, separate_sh, analysis_colors, render_label,
    visualization_metadata,
):
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base_path, render_label, "renders")
    makedirs(render_path, exist_ok=True)
    if visualization_metadata is not None:
        visualization_path = os.path.dirname(render_path)
        metadata_path = os.path.join(visualization_path, "normalization.json")
        with open(metadata_path, "w") as stream:
            json.dump(visualization_metadata, stream, indent=2)
        gradient_values = torch.linspace(
            0.0, 1.0, 512, dtype=background.dtype, device=background.device
        )
        colorbar = knn_colormap(gradient_values, 0.0, 1.0)
        colorbar = colorbar.transpose(0, 1).unsqueeze(1).repeat(1, 32, 1)
        torchvision.utils.save_image(
            colorbar, os.path.join(visualization_path, "colormap.png")
        )

    for idx, view in enumerate(tqdm(views, desc="KNN analysis rendering progress")):
        analysis_rendering = render(
            view, gaussians, pipeline, background,
            use_trained_exp=False,
            separate_sh=separate_sh,
            override_color=analysis_colors,
        )["render"]

        if train_test_exp:
            analysis_rendering = analysis_rendering[
                ..., analysis_rendering.shape[-1] // 2:
            ]

        torchvision.utils.save_image(
            analysis_rendering, os.path.join(render_path, "{0:05d}.png".format(idx))
        )


def _kth_distance(gaussians, k):
    return gaussians.get_knn(k).reshape(-1)


def _mean_distance(gaussians, k):
    return gaussians.get_knn_metric("knn_mean_k{}".format(k)).reshape(-1)


def _max_scale(gaussians):
    return torch.max(gaussians.get_scaling, dim=1).values


def _mean_scale(gaussians):
    return torch.mean(gaussians.get_scaling, dim=1)


def _mean_over_max_scale(gaussians, k):
    return _mean_distance(gaussians, k) / _max_scale(gaussians)


def _kth_over_max_scale(gaussians, k):
    return _kth_distance(gaussians, k) / _max_scale(gaussians)


def _mean_over_mean_scale(gaussians, k):
    return _mean_distance(gaussians, k) / _mean_scale(gaussians)


def _kth_over_mean_scale(gaussians, k):
    return _kth_distance(gaussians, k) / _mean_scale(gaussians)


# Add a metric by defining one function with this signature and registering it here.
METRIC_FUNCTIONS = {
    "kth": _kth_distance,
    "mean": _mean_distance,
    "kth_over_max_scale": _kth_over_max_scale,
    "mean_over_max_scale": _mean_over_max_scale,
    "kth_over_mean_scale": _kth_over_mean_scale,
    "mean_over_mean_scale": _mean_over_mean_scale,
}

METRIC_DESCRIPTIONS = {
    "kth": "raw Euclidean distance to the K-th nearest Gaussian",
    "mean": "mean raw Euclidean distance to the K nearest Gaussians",
    "kth_over_max_scale": "K-th distance divided by the longest Gaussian scale axis",
    "mean_over_max_scale": "mean KNN distance divided by the longest Gaussian scale axis",
    "kth_over_mean_scale": "K-th distance divided by the mean Gaussian scale axis",
    "mean_over_mean_scale": "mean KNN distance divided by the mean Gaussian scale axis",
}


def prepare_analysis_colors(
    gaussians, metric, k, percentile_min, percentile_max,
    value_min=None, value_max=None,
):
    values = METRIC_FUNCTIONS[metric](gaussians, k)
    if not torch.isfinite(values).all():
        raise ValueError("{} at K={} contains non-finite values".format(metric, k))

    requested_percentiles = []
    if value_min is None:
        requested_percentiles.append(percentile_min / 100.0)
    if value_max is None:
        requested_percentiles.append(percentile_max / 100.0)
    percentile_values = []
    if requested_percentiles:
        quantiles = torch.tensor(
            requested_percentiles, dtype=values.dtype, device=values.device
        )
        percentile_values = torch.quantile(values, quantiles).tolist()
    percentile_index = 0
    if value_min is None:
        lower = float(percentile_values[percentile_index])
        percentile_index += 1
    else:
        lower = float(value_min)
    if value_max is None:
        upper = float(percentile_values[percentile_index])
    else:
        upper = float(value_max)
    if upper <= lower:
        raise ValueError(
            "KNN visualization maximum ({}) must be greater than minimum ({})".format(
                upper, lower
            )
        )
    colors = knn_colormap(values, lower, upper)
    metadata = {
        "metric": metric,
        "k": k,
        "statistic": METRIC_DESCRIPTIONS[metric],
        "normalization_min": lower,
        "normalization_max": upper,
        "percentile_min": None if value_min is not None else percentile_min,
        "percentile_max": None if value_max is not None else percentile_max,
        "colormap": "blue-cyan-yellow-red",
        "low_color": "blue",
        "high_color": "red",
    }
    return colors, metadata


def render_sets(
    dataset : ModelParams, iteration : int, pipeline : PipelineParams,
    skip_train : bool, skip_test : bool, separate_sh: bool,
    knn_ply=None, knn_k=5, metrics=None, output_label=None,
    knn_percentile_min=1.0, knn_percentile_max=99.5,
    knn_min=None, knn_max=None,
):
    with torch.no_grad():
        loaded_iteration = iteration
        if loaded_iteration == -1:
            loaded_iteration = searchForMaxIteration(
                os.path.join(dataset.model_path, "point_cloud")
            )
        if knn_ply is None:
            knn_ply = os.path.join(
                dataset.model_path, "knn_analysis",
                "iteration_{}".format(loaded_iteration), "point_cloud_knn.ply",
            )
        knn_ply = os.path.abspath(knn_ply)
        if not os.path.isfile(knn_ply):
            raise FileNotFoundError(
                "KNN-annotated PLY not found: {}. Run analysis/knn.py first.".format(
                    knn_ply
                )
            )

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset, gaussians, load_iteration=loaded_iteration, shuffle=False,
            load_ply_path=knn_ply,
        )

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        metrics = list(METRIC_FUNCTIONS) if metrics is None else metrics
        for metric in metrics:
            analysis_colors, visualization_metadata = prepare_analysis_colors(
                gaussians, metric, knn_k,
                knn_percentile_min, knn_percentile_max, knn_min, knn_max,
            )
            render_label = output_label or "knn_{}_k{}".format(metric, knn_k)
            visualization_metadata["input_ply"] = knn_ply
            visualization_metadata["render_folder"] = render_label
            print(
                "KNN analysis: {} (K={}), range [{:.6g}, {:.6g}], blue -> red".format(
                    metric, knn_k,
                    visualization_metadata["normalization_min"],
                    visualization_metadata["normalization_max"],
                )
            )

            if not skip_train:
                render_analysis_set(
                    dataset.model_path, "train", scene.loaded_iter,
                    scene.getTrainCameras(), gaussians, pipeline, background,
                    dataset.train_test_exp, separate_sh, analysis_colors,
                    render_label, visualization_metadata,
                )

            if not skip_test:
                render_analysis_set(
                    dataset.model_path, "test", scene.loaded_iter,
                    scene.getTestCameras(), gaussians, pipeline, background,
                    dataset.train_test_exp, separate_sh, analysis_colors,
                    render_label, visualization_metadata,
                )
            del analysis_colors

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--knn_ply", type=str, default=None,
        help="Annotated PLY; defaults to the analysis output for this iteration.",
    )
    parser.add_argument("--knn_k", type=int, default=5, help="KNN property K to render.")
    parser.add_argument(
        "--metrics", nargs="+", choices=sorted(METRIC_FUNCTIONS), default=None,
        help="Metrics to render in one run; omitted means all metrics.",
    )
    parser.add_argument(
        "--metric", choices=sorted(METRIC_FUNCTIONS), default=None,
        help="Deprecated single-metric alias retained for compatibility.",
    )
    parser.add_argument(
        "--output_label", type=str, default=None,
        help="Optional output folder name; defaults to knn_<metric>_k<K>.",
    )
    parser.add_argument(
        "--knn_percentile_min", type=float, default=1.0,
        help="Lower normalization percentile (default: 1).",
    )
    parser.add_argument(
        "--knn_percentile_max", type=float, default=99.5,
        help="Upper normalization percentile (default: 99.5).",
    )
    parser.add_argument(
        "--knn_min", type=float, default=None,
        help="Optional fixed selected-metric normalization minimum.",
    )
    parser.add_argument(
        "--knn_max", type=float, default=None,
        help="Optional fixed selected-metric normalization maximum.",
    )
    args = get_combined_args(parser)
    knn_ply = getattr(args, "knn_ply", None)
    knn_min = getattr(args, "knn_min", None)
    knn_max = getattr(args, "knn_max", None)
    output_label = getattr(args, "output_label", None)
    metrics = getattr(args, "metrics", None)
    legacy_metric = getattr(args, "metric", None)
    if metrics is not None and legacy_metric is not None:
        raise ValueError("Use either --metrics or --metric, not both")
    if metrics is None:
        metrics = [legacy_metric] if legacy_metric is not None else list(METRIC_FUNCTIONS)
    metrics = list(dict.fromkeys(metrics))
    if output_label is not None and len(metrics) != 1:
        raise ValueError("--output_label can only be used when rendering one metric")
    print("Rendering KNN analysis for " + args.model_path)
    if args.knn_k <= 0:
        raise ValueError("--knn_k must be positive")
    if not 0 <= args.knn_percentile_min < args.knn_percentile_max <= 100:
        raise ValueError(
            "Expected 0 <= --knn_percentile_min < --knn_percentile_max <= 100"
        )
    if knn_min is not None and knn_max is not None:
        if knn_max <= knn_min:
            raise ValueError("--knn_max must be greater than --knn_min")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args),
        args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
        knn_ply, args.knn_k, metrics, output_label,
        args.knn_percentile_min, args.knn_percentile_max,
        knn_min, knn_max,
    )
