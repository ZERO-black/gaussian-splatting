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
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_comparison_set(
    model_path, name, iteration, views, gaussians, pipeline, background,
    train_test_exp, separate_sh, knn_colors, render_label,
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

    for idx, view in enumerate(tqdm(views, desc="RGB + KNN rendering progress")):
        rgb_rendering = render(
            view, gaussians, pipeline, background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )["render"]
        knn_rendering = render(
            view, gaussians, pipeline, background,
            use_trained_exp=False,
            separate_sh=separate_sh,
            override_color=knn_colors,
        )["render"]

        if train_test_exp:
            rgb_rendering = rgb_rendering[..., rgb_rendering.shape[-1] // 2:]
            knn_rendering = knn_rendering[..., knn_rendering.shape[-1] // 2:]

        comparison = torch.cat((rgb_rendering, knn_rendering), dim=-1)
        torchvision.utils.save_image(
            comparison, os.path.join(render_path, "{0:05d}.png".format(idx))
        )


def prepare_knn_colors(
    gaussians, k, percentile_min, percentile_max, value_min=None, value_max=None
):
    values = gaussians.get_knn(k).reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("knn_k{} contains non-finite values".format(k))

    lower = (
        float(value_min)
        if value_min is not None
        else float(torch.quantile(values, percentile_min / 100.0).item())
    )
    upper = (
        float(value_max)
        if value_max is not None
        else float(torch.quantile(values, percentile_max / 100.0).item())
    )
    if upper <= lower:
        raise ValueError(
            "KNN visualization maximum ({}) must be greater than minimum ({})".format(
                upper, lower
            )
        )
    colors = knn_colormap(values, lower, upper)
    metadata = {
        "property": "knn_k{}".format(k),
        "statistic": "raw Euclidean distance to the K-th nearest Gaussian",
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
    knn_ply=None, knn_k=5,
    knn_percentile_min=1.0, knn_percentile_max=99.5,
    knn_min=None, knn_max=None,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if knn_ply is None:
            knn_ply = os.path.join(
                dataset.model_path, "knn_analysis",
                "iteration_{}".format(scene.loaded_iter), "point_cloud_knn.ply",
            )
        knn_ply = os.path.abspath(knn_ply)
        if not os.path.isfile(knn_ply):
            raise FileNotFoundError(
                "KNN-annotated PLY not found: {}. Run analysis/knn.py first.".format(
                    knn_ply
                )
            )
        gaussians.load_ply(knn_ply, dataset.train_test_exp)
        knn_colors, visualization_metadata = prepare_knn_colors(
            gaussians, knn_k, knn_percentile_min, knn_percentile_max,
            knn_min, knn_max,
        )
        render_label = "knn_k{}_comparison".format(knn_k)
        visualization_metadata["input_ply"] = knn_ply
        visualization_metadata["layout"] = "left: RGB, right: KNN"
        print(
            "RGB + KNN comparison: knn_k{}, range [{:.6g}, {:.6g}], blue -> red".format(
                knn_k,
                visualization_metadata["normalization_min"],
                visualization_metadata["normalization_max"],
            )
        )

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_comparison_set(
                dataset.model_path, "train", scene.loaded_iter,
                scene.getTrainCameras(), gaussians, pipeline, background,
                dataset.train_test_exp, separate_sh, knn_colors, render_label,
                visualization_metadata,
            )

        if not skip_test:
            render_comparison_set(
                dataset.model_path, "test", scene.loaded_iter,
                scene.getTestCameras(), gaussians, pipeline, background,
                dataset.train_test_exp, separate_sh, knn_colors, render_label,
                visualization_metadata,
            )

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
        "--knn_percentile_min", type=float, default=1.0,
        help="Lower normalization percentile (default: 1).",
    )
    parser.add_argument(
        "--knn_percentile_max", type=float, default=99.5,
        help="Upper normalization percentile (default: 99.5).",
    )
    parser.add_argument(
        "--knn_min", type=float, default=None,
        help="Optional fixed raw-distance normalization minimum.",
    )
    parser.add_argument(
        "--knn_max", type=float, default=None,
        help="Optional fixed raw-distance normalization maximum.",
    )
    args = get_combined_args(parser)
    knn_ply = getattr(args, "knn_ply", None)
    knn_min = getattr(args, "knn_min", None)
    knn_max = getattr(args, "knn_max", None)
    print("Rendering RGB + KNN comparison for " + args.model_path)
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
        knn_ply, args.knn_k,
        args.knn_percentile_min, args.knn_percentile_max,
        knn_min, knn_max,
    )
