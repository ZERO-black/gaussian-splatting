from __future__ import annotations

import ast
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from os import makedirs
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
from tqdm import tqdm

from analysis.knn import STATIC_METRICS
from analysis.metric_histogram import (
    build_histogram_spec,
    sample_values,
    save_comparison_histogram,
    subsample_values,
)
from analysis.metric_core import (
    ALL_METRICS,
    CAMERA_DEPTH_METRICS,
    COVARIANCE_METRICS,
    METRIC_DESCRIPTIONS,
    VIEW_DEPENDENT_METRICS,
    colorize,
    compute_metric_values,
    metric_quantiles,
    metric_is_available,
    normalization_bounds,
    prepare_metric_context,
    valid_metric_samples,
)
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except Exception:
    SEPARATE_SH = False


def default_input_ply(model_path, iteration, metric):
    if metric in COVARIANCE_METRICS:
        return os.path.join(
            model_path,
            "covariance_analysis",
            f"iteration_{iteration}",
            "point_cloud_covariance.ply",
        )

    if metric in STATIC_METRICS or metric in CAMERA_DEPTH_METRICS:
        return os.path.join(
            model_path,
            "knn_analysis",
            f"iteration_{iteration}",
            "point_cloud_knn.ply",
        )

    return os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )


def read_model_config(model_path):
    """Read the model's saved Namespace without executing arbitrary Python."""
    config_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(config_path):
        return {}

    with open(config_path) as stream:
        expression = ast.parse(stream.read().strip(), mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "Namespace"
        or expression.args
    ):
        raise ValueError(f"Unsupported cfg_args format: {config_path}")
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in expression.keywords
        if keyword.arg is not None
    }


def build_dataset_config(model_params, args, model_path):
    """Merge one model's cfg_args with explicit shared CLI overrides."""
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    defaults = {
        key.lstrip("_"): value
        for key, value in vars(model_params).items()
    }
    config = {**defaults, **read_model_config(model_path)}
    for name in defaults:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            config[name] = cli_value
    config["model_path"] = model_path
    if not config.get("source_path"):
        raise ValueError(
            f"Dataset source_path is missing from {model_path}/cfg_args; "
            "provide a shared --source_path/-s override"
        )

    dataset = model_params.extract(Namespace(**config))
    if not os.path.isdir(dataset.source_path):
        raise FileNotFoundError(
            f"Dataset source_path for {model_path} does not exist: {dataset.source_path}"
        )
    return dataset


def build_shared_pipeline_config(pipeline_params, args, reference_model, target_model):
    """Load and verify the renderer settings saved with both models."""
    defaults = dict(vars(pipeline_params))

    def merged_model_pipeline(model_path):
        saved = read_model_config(os.path.abspath(model_path))
        return {
            name: saved.get(name, default)
            for name, default in defaults.items()
        }

    reference = merged_model_pipeline(reference_model)
    target = merged_model_pipeline(target_model)

    # PipelineParams currently contains store_true booleans. A true CLI flag is
    # therefore an explicit shared override; false means the flag was omitted.
    for name in defaults:
        if getattr(args, name, defaults[name]) != defaults[name]:
            reference[name] = getattr(args, name)
            target[name] = getattr(args, name)

    differences = {
        name: (reference[name], target[name])
        for name in defaults
        if reference[name] != target[name]
    }
    if differences:
        details = ", ".join(
            f"{name}: {values[0]!r} != {values[1]!r}"
            for name, values in differences.items()
        )
        raise ValueError(f"Reference and target pipeline settings differ: {details}")

    return pipeline_params.extract(Namespace(**reference))


def load_scene(dataset, iteration, input_ply, metric):
    model_path = dataset.model_path

    loaded_iteration = iteration
    if loaded_iteration == -1:
        loaded_iteration = searchForMaxIteration(
            os.path.join(model_path, "point_cloud")
        )

    if input_ply is None:
        input_ply = default_input_ply(model_path, loaded_iteration, metric)
    input_ply = os.path.abspath(input_ply)
    if not os.path.isfile(input_ply):
        raise FileNotFoundError(f"Metric input PLY not found: {input_ply}")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=loaded_iteration,
        shuffle=False,
        load_ply_path=input_ply,
    )
    return dataset, scene, gaussians, input_ply


def choose_views(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    raise ValueError(f"Unknown split: {split}")


def collect_metric_samples(
    gaussians,
    views,
    metric,
    k,
    context,
    max_samples_per_view,
    max_total_samples,
    histogram_seed,
    label,
):
    if metric not in VIEW_DEPENDENT_METRICS:
        values, valid = compute_metric_values(gaussians, None, metric, k, context)
        samples = sample_values(
            values, valid, max_samples_per_view, seed=histogram_seed
        )
        if samples.numel() == 0:
            raise RuntimeError(f"No valid {label} metric samples were found")
        return subsample_values(
            samples, max_samples=max_total_samples, seed=histogram_seed
        )

    chunks = []
    for index, view in enumerate(
        tqdm(views, desc=f"Collecting {label} distribution")
    ):
        values, valid = compute_metric_values(gaussians, view, metric, k, context)
        chunk = sample_values(
            values,
            valid,
            max_samples_per_view,
            seed=histogram_seed + index,
        )
        if chunk.numel() > 0:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(f"No valid {label} metric samples were found")
    return subsample_values(
        torch.cat(chunks),
        max_samples=max_total_samples,
        seed=histogram_seed + len(views),
    )


def validate_shared_cameras(reference_views, target_views):
    """Ensure reference camera objects are valid for rendering the target model."""
    if len(reference_views) != len(target_views):
        raise ValueError(
            "Reference and target evaluation splits have different camera counts: "
            f"{len(reference_views)} != {len(target_views)}"
        )
    if not reference_views:
        raise ValueError("The selected evaluation split has no cameras")

    for index, (reference, target) in enumerate(zip(reference_views, target_views)):
        if reference.image_name != target.image_name:
            raise ValueError(
                f"Camera {index} image mismatch: "
                f"{reference.image_name!r} != {target.image_name!r}"
            )
        scalar_fields = ("image_width", "image_height", "FoVx", "FoVy")
        for field in scalar_fields:
            reference_value = getattr(reference, field)
            target_value = getattr(target, field)
            if abs(float(reference_value) - float(target_value)) > 1e-6:
                raise ValueError(
                    f"Camera {reference.image_name!r} differs in {field}: "
                    f"{reference_value} != {target_value}"
                )
        if not torch.allclose(
            reference.world_view_transform,
            target.world_view_transform,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                f"Camera pose mismatch for {reference.image_name!r}"
            )


def compute_summary(values, valid, thresholds):
    samples = valid_metric_samples(values, valid)
    if samples.numel() == 0:
        return {"count": 0}
    p90, p95, p99 = metric_quantiles(samples, [0.90, 0.95, 0.99])

    summary = {
        "count": int(samples.numel()),
        "mean": float(samples.mean().item()),
        "median": float(samples.median().item()),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(samples.max().item()),
    }
    for name, threshold in thresholds.items():
        summary[f"fraction_above_{name}"] = float(
            (samples > threshold).float().mean().item()
        )
    return summary


def render_model_with_fixed_range(
    label,
    output_root,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    metric,
    k,
    context,
    lower,
    upper,
    thresholds,
):
    model_output = os.path.join(output_root, label)
    render_dir = os.path.join(model_output, "renders")
    makedirs(render_dir, exist_ok=True)

    per_view = {}
    static_values = static_valid = None
    if metric not in VIEW_DEPENDENT_METRICS:
        static_values, static_valid = compute_metric_values(
            gaussians, None, metric, k, context
        )

    for index, view in enumerate(tqdm(views, desc=f"Rendering {label}")):
        if metric in VIEW_DEPENDENT_METRICS:
            values, valid = compute_metric_values(gaussians, view, metric, k, context)
        else:
            values, valid = static_values, static_valid

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

        filename = f"{index:05d}.png"
        torchvision.utils.save_image(image, os.path.join(render_dir, filename))

        stats = compute_summary(values, valid, thresholds)
        stats["camera"] = getattr(view, "image_name", str(index))
        per_view[filename] = stats

    with open(os.path.join(model_output, "stats.json"), "w") as stream:
        json.dump(per_view, stream, indent=2)
    return per_view


def aggregate_view_stats(per_view, keys):
    result = {}
    for key in keys:
        values = [stats[key] for stats in per_view.values() if key in stats]
        result[key] = None if not values else float(sum(values) / len(values))
    return result


def main(args, model_params, pipeline_params):
    ref_dataset = build_dataset_config(
        model_params, args, args.reference_model
    )
    target_dataset = build_dataset_config(
        model_params, args, args.target_model
    )
    ref_dataset, ref_scene, ref_gaussians, ref_ply = load_scene(
        ref_dataset,
        args.iteration,
        args.reference_ply,
        args.metric,
    )
    target_dataset, target_scene, target_gaussians, target_ply = load_scene(
        target_dataset,
        args.iteration,
        args.target_ply,
        args.metric,
    )

    if not metric_is_available(ref_gaussians, args.metric, args.knn_k):
        raise ValueError(
            f"Reference PLY does not provide inputs for {args.metric}, K={args.knn_k}"
        )
    if not metric_is_available(target_gaussians, args.metric, args.knn_k):
        raise ValueError(
            f"Target PLY does not provide inputs for {args.metric}, K={args.knn_k}"
        )

    eval_views = choose_views(ref_scene, args.eval_split)
    target_views = choose_views(target_scene, args.eval_split)
    if ref_dataset.white_background != target_dataset.white_background:
        raise ValueError("Reference and target white_background settings differ")
    if ref_dataset.train_test_exp != target_dataset.train_test_exp:
        raise ValueError("Reference and target train_test_exp settings differ")
    validate_shared_cameras(eval_views, target_views)
    pipeline = build_shared_pipeline_config(
        pipeline_params,
        args,
        args.reference_model,
        args.target_model,
    )

    ref_context = prepare_metric_context(
        ref_gaussians, args.metric, args.knn_k
    )
    target_context = prepare_metric_context(
        target_gaussians, args.metric, args.knn_k
    )

    reference_samples = collect_metric_samples(
        ref_gaussians,
        eval_views,
        args.metric,
        args.knn_k,
        ref_context,
        args.max_samples_per_view,
        args.max_total_samples,
        args.histogram_seed,
        "reference",
    )
    target_samples = collect_metric_samples(
        target_gaussians,
        eval_views,
        args.metric,
        args.knn_k,
        target_context,
        args.max_samples_per_view,
        args.max_total_samples,
        args.histogram_seed,
        "target",
    )

    lower, upper = normalization_bounds(
        reference_samples,
        torch.ones_like(reference_samples, dtype=torch.bool),
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
    )
    threshold_values = metric_quantiles(
        reference_samples,
        [p / 100.0 for p in args.threshold_percentiles],
    )
    thresholds = {
        f"p{p:g}": float(value)
        for p, value in zip(args.threshold_percentiles, threshold_values)
    }

    output_root = args.output_dir or os.path.join(
        args.target_model,
        "metric_validation",
        f"{args.metric}_k{args.knn_k}",
    )
    makedirs(output_root, exist_ok=True)

    histogram_spec = build_histogram_spec(
        reference_samples,
        bins=args.histogram_bins,
        percentile_min=args.histogram_percentile_min,
        percentile_max=args.histogram_percentile_max,
    )
    save_comparison_histogram(
        os.path.join(output_root, "distribution_comparison.png"),
        os.path.join(output_root, "distribution_comparison.json"),
        args.metric,
        reference_samples,
        target_samples,
        histogram_spec,
        title=f"Reference vs target: {args.metric} (K={args.knn_k})",
        metadata={
            "k": args.knn_k,
            "eval_split": args.eval_split,
            "range_source": "reference valid samples only",
            "sampling": "random without replacement",
            "sampling_seed": args.histogram_seed,
            "maximum_samples_per_view": args.max_samples_per_view,
            "maximum_total_samples": args.max_total_samples,
        },
    )

    background = torch.tensor(
        [1, 1, 1] if ref_dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    reference_stats = render_model_with_fixed_range(
        "reference",
        output_root,
        eval_views,
        ref_gaussians,
        pipeline,
        background,
        ref_dataset.train_test_exp,
        args.metric,
        args.knn_k,
        ref_context,
        lower,
        upper,
        thresholds,
    )
    target_stats = render_model_with_fixed_range(
        "target",
        output_root,
        eval_views,
        target_gaussians,
        pipeline,
        background,
        target_dataset.train_test_exp,
        args.metric,
        args.knn_k,
        target_context,
        lower,
        upper,
        thresholds,
    )

    comparison_keys = ["mean", "median"] + [
        f"fraction_above_{name}" for name in thresholds
    ]
    reference_aggregate = aggregate_view_stats(reference_stats, comparison_keys)
    target_aggregate = aggregate_view_stats(target_stats, comparison_keys)
    differences = {
        key: (
            None
            if reference_aggregate[key] is None or target_aggregate[key] is None
            else target_aggregate[key] - reference_aggregate[key]
        )
        for key in comparison_keys
    }

    metadata = {
        "metric": args.metric,
        "description": METRIC_DESCRIPTIONS[args.metric],
        "k": args.knn_k,
        "eval_split": args.eval_split,
        "camera_source": "reference_model",
        "camera_configuration_check": "passed",
        "pipeline_configuration_check": "passed",
        "pipeline": vars(pipeline),
        "reference_model": args.reference_model,
        "target_model": args.target_model,
        "reference_iteration": ref_scene.loaded_iter,
        "target_iteration": target_scene.loaded_iter,
        "reference_source_path": ref_dataset.source_path,
        "target_source_path": target_dataset.source_path,
        "reference_ply": ref_ply,
        "target_ply": target_ply,
        "normalization_source": "reference_model_only",
        "normalization_min": lower,
        "normalization_max": upper,
        "percentile_min": args.percentile_min,
        "percentile_max": args.percentile_max,
        "reference_thresholds": thresholds,
        "reference_sample_count": int(reference_samples.numel()),
        "target_sample_count": int(target_samples.numel()),
        "sampling": "random without replacement",
        "sampling_seed": args.histogram_seed,
        "maximum_samples_per_view": args.max_samples_per_view,
        "maximum_total_samples": args.max_total_samples,
        "aggregate": {
            "reference": reference_aggregate,
            "target": target_aggregate,
            "target_minus_reference": differences,
        },
    }

    with open(os.path.join(output_root, "validation.json"), "w") as stream:
        json.dump(metadata, stream, indent=2)

    print("\nValidation complete")
    print(f"Output: {output_root}")
    print(f"Fixed reference range: [{lower:.6g}, {upper:.6g}]")
    for name, value in thresholds.items():
        print(f"Reference {name}: {value:.6g}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare one Gaussian metric using reference-model normalization"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--reference_model", required=True)
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--reference_ply", default=None)
    parser.add_argument("--target_ply", default=None)
    parser.add_argument("--metric", required=True, choices=ALL_METRICS)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--eval_split", choices=["train", "test"], default="test")
    parser.add_argument("--percentile_min", type=float, default=1.0)
    parser.add_argument("--percentile_max", type=float, default=99.5)
    parser.add_argument(
        "--threshold_percentiles", nargs="+", type=float, default=[95.0, 99.0]
    )
    parser.add_argument(
        "--max_samples_per_view",
        type=int,
        default=200_000,
        help="0 uses all valid values from every reference view.",
    )
    parser.add_argument(
        "--histogram_seed",
        type=int,
        default=0,
        help="Base seed for reproducible random distribution sampling.",
    )
    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=1_000_000,
        help="Final reference/target distribution sample cap; 0 keeps all samples.",
    )
    parser.add_argument("--histogram_bins", type=int, default=160)
    parser.add_argument("--histogram_percentile_min", type=float, default=0.1)
    parser.add_argument("--histogram_percentile_max", type=float, default=99.9)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.model_path is not None:
        parser.error(
            "--model_path/-m is not used by validation; use "
            "--reference_model and --target_model"
        )
    if args.knn_k <= 0:
        raise ValueError("--knn_k must be positive")
    if not 0 <= args.percentile_min < args.percentile_max <= 100:
        raise ValueError("Expected 0 <= percentile_min < percentile_max <= 100")
    if any(p <= 0 or p >= 100 for p in args.threshold_percentiles):
        raise ValueError("threshold percentiles must be between 0 and 100")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram_bins must be positive")
    if args.max_samples_per_view < 0:
        raise ValueError("--max_samples_per_view must be non-negative")
    if args.max_total_samples < 0:
        raise ValueError("--max_total_samples must be non-negative")
    if not (
        0
        <= args.histogram_percentile_min
        < args.histogram_percentile_max
        <= 100
    ):
        raise ValueError("Invalid histogram percentile range")

    safe_state(args.quiet)
    main(args, model, pipeline)
