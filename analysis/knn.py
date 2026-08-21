"""Analyze KNN distances between trained Gaussian centers.

PLY vertex order is preserved, so every reported index identifies the same
Gaussian in the trained model. Distances can also be normalized by either the
longest Gaussian scale axis or the mean axis length.
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)

METRIC_LABELS = {
    "mean": "Mean distance to K nearest neighbors",
    "kth": "Distance to the K-th nearest neighbor",
    "mean_over_max_scale": "Mean KNN distance / longest scale axis",
    "kth_over_max_scale": "K-th distance / longest scale axis",
    "mean_over_mean_scale": "Mean KNN distance / mean scale axis",
    "kth_over_mean_scale": "K-th distance / mean scale axis",
}


def ply_property_name(metric, k):
    """Return stable PLY property names, retaining the old K-th-distance name."""
    if metric == "kth":
        return "knn_k{}".format(k)
    return "knn_{}_k{}".format(metric, k)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Euclidean KNN distances in a trained 3DGS point cloud."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-m", "--model-path", type=Path,
        help="Model directory containing point_cloud/iteration_*/point_cloud.ply.",
    )
    source.add_argument(
        "--ply-path", type=Path,
        help="A trained point_cloud.ply (indices follow its vertex order).",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Iteration under --model-path; -1 selects the latest (default: -1).",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[5, 10, 20, 50],
        help="One or more neighbor counts (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--tail-percentile", type=float, default=99,
        help="Flag points by either raw-distance metric at this percentile (default: 99).",
    )
    parser.add_argument(
        "--percentiles", type=float, nargs="+", default=list(DEFAULT_PERCENTILES),
        help="Percentiles included in summary.json.",
    )
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins (default: 100).")
    parser.add_argument(
        "--hist-max-percentile", type=float, default=99.5,
        help=(
            "Upper x-axis percentile for histograms; 100 shows the full range "
            "(default: 99.5). This only affects visualization."
        ),
    )
    parser.add_argument(
        "--query-batch-size", type=int, default=100_000,
        help="Centers queried at once to bound temporary memory (default: 100000).",
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="cKDTree workers; -1 uses all CPU cores (default: -1).",
    )
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Regenerate histograms from existing knn_metrics.npz without recomputing KNN.",
    )
    parser.add_argument(
        "--export-ply-only", action="store_true",
        help="Add saved KNN metrics to a PLY without recomputing KNN.",
    )
    parser.add_argument(
        "--annotated-ply", type=Path,
        help="Annotated PLY path (default: <output-dir>/point_cloud_knn.ply).",
    )
    parser.add_argument(
        "--skip-annotated-ply", action="store_true",
        help="Do not write the Gaussian PLY containing KNN metric properties.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Output directory (default: <model>/knn_analysis/iteration_<n>).",
    )
    return parser.parse_args()


def _latest_iteration(point_cloud_root):
    iterations = []
    if point_cloud_root.is_dir():
        for path in point_cloud_root.iterdir():
            if path.is_dir() and path.name.startswith("iteration_"):
                try:
                    iterations.append(int(path.name.rsplit("_", 1)[1]))
                except ValueError:
                    pass
    if not iterations:
        raise FileNotFoundError("No iteration_* directories found in {}".format(point_cloud_root))
    return max(iterations)


def _iteration_from_path(ply_path):
    if ply_path.parent.name.startswith("iteration_"):
        try:
            return int(ply_path.parent.name.rsplit("_", 1)[1])
        except ValueError:
            pass
    return -1


def resolve_input(args, analysis_directory="knn_analysis"):
    if args.ply_path is not None:
        ply_path = args.ply_path.expanduser().resolve()
        iteration = _iteration_from_path(ply_path)
        default_output = ply_path.parent / analysis_directory
    else:
        model_path = args.model_path.expanduser().resolve()
        point_cloud_root = model_path / "point_cloud"
        iteration = args.iteration
        if iteration == -1:
            iteration = _latest_iteration(point_cloud_root)
        ply_path = point_cloud_root / ("iteration_{}".format(iteration)) / "point_cloud.ply"
        default_output = model_path / analysis_directory / ("iteration_{}".format(iteration))
    if not ply_path.is_file():
        raise FileNotFoundError("Trained Gaussian PLY not found: {}".format(ply_path))
    return ply_path, (args.output_dir or default_output).expanduser().resolve(), iteration


def load_gaussian_geometry(ply_path, include_rotations=False):
    """Read xyz, activated scales, and optionally quaternions without CUDA."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required; install the project environment first.") from exc

    # GaussianModel.load_ply reads this format too, but materializes every
    # appearance/covariance property on CUDA. This read-only analysis only needs xyz.
    ply = PlyData.read(str(ply_path), mmap="r")
    if not ply.elements or ply.elements[0].name != "vertex":
        raise ValueError("PLY has no leading vertex element: {}".format(ply_path))
    vertex = ply.elements[0]
    available = {prop.name for prop in vertex.properties}
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if include_rotations:
        required.update({"rot_0", "rot_1", "rot_2", "rot_3"})
    if not required.issubset(available):
        missing = sorted(required - available)
        raise ValueError("PLY vertex element is missing {}: {}".format(missing, ply_path))
    centers = np.ascontiguousarray(
        np.column_stack((vertex["x"], vertex["y"], vertex["z"])), dtype=np.float64
    )
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("Expected a non-empty Nx3 center array, got {}".format(centers.shape))
    if not np.isfinite(centers).all():
        bad_count = int(np.count_nonzero(~np.isfinite(centers).all(axis=1)))
        raise ValueError("Found {} centers with non-finite coordinates".format(bad_count))
    # 3DGS stores log-scale in PLY; get_scaling applies exp during rendering.
    log_scales = np.column_stack(
        (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])
    ).astype(np.float64, copy=False)
    scales = np.exp(log_scales)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Found non-finite or non-positive activated Gaussian scales")
    if not include_rotations:
        return centers, scales
    rotations = np.column_stack(
        (vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])
    ).astype(np.float64, copy=False)
    rotation_norms = np.linalg.norm(rotations, axis=1)
    if not np.isfinite(rotations).all() or np.any(rotation_norms == 0):
        raise ValueError("Found non-finite or zero-length Gaussian rotations")
    rotations = rotations / rotation_norms[:, None]
    return centers, scales, rotations


def iter_knn_batches(centers, max_k, batch_size, workers):
    """Yield the exact neighborhoods used by all center-based KNN analyses."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for exact configurable-K queries; update the project environment."
        ) from exc

    point_count = centers.shape[0]
    if max_k >= point_count:
        raise ValueError(
            "Largest K ({}) must be smaller than Gaussian count ({})".format(
                max_k, point_count
            )
        )
    tree = cKDTree(centers)
    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        try:
            distances, indices = tree.query(
                centers[start:end], k=max_k + 1, workers=workers
            )
        except TypeError:  # scipy < 1.6
            distances, indices = tree.query(
                centers[start:end], k=max_k + 1, n_jobs=workers
            )
        query_indices = np.arange(start, end)[:, None]
        is_self = indices == query_indices
        # Usually self is column zero. Explicit removal also handles coincident
        # centers, for which cKDTree does not promise an ordering among ties.
        rank = np.broadcast_to(np.arange(max_k + 1), indices.shape)
        non_self_order = np.argsort(
            np.where(is_self, max_k + 1, rank), axis=1, kind="stable"
        )[:, :max_k]
        yield (
            start,
            end,
            np.take_along_axis(distances, non_self_order, axis=1),
            np.take_along_axis(indices, non_self_order, axis=1),
        )


def compute_knn_metrics(centers, scales, k_values, batch_size, workers):
    """Compute exact KNN metrics with a tree and bounded query memory."""
    k_values = sorted(set(k_values))
    point_count, max_k = centers.shape[0], max(k_values)
    metrics = {k: {
        name: np.empty(point_count, dtype=np.float32) for name in METRIC_LABELS
    } for k in k_values}
    max_scale = np.max(scales, axis=1)
    mean_scale = np.mean(scales, axis=1)
    for start, end, neighbor_distances, _ in iter_knn_batches(
        centers, max_k, batch_size, workers
    ):
        cumulative = np.cumsum(neighbor_distances, axis=1, dtype=np.float64)
        for k, values in metrics.items():
            mean_distance = cumulative[:, k - 1] / k
            kth_distance = neighbor_distances[:, k - 1]
            values["mean"][start:end] = mean_distance
            values["kth"][start:end] = kth_distance
            values["mean_over_max_scale"][start:end] = mean_distance / max_scale[start:end]
            values["kth_over_max_scale"][start:end] = kth_distance / max_scale[start:end]
            values["mean_over_mean_scale"][start:end] = mean_distance / mean_scale[start:end]
            values["kth_over_mean_scale"][start:end] = kth_distance / mean_scale[start:end]
        print(
            "KNN query: {:,}/{:,} centers".format(end, point_count),
            end="\r", flush=True,
        )
    print()
    return metrics


def _percentile_label(percentile):
    return "p{:g}".format(percentile)


def describe(values, percentiles):
    percentile_values = np.percentile(values, percentiles)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            _percentile_label(p): float(value)
            for p, value in zip(percentiles, percentile_values)
        },
    }


def save_histogram(output_path, k, metrics, thresholds, bins, hist_max_percentile):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save histogram PNGs.") from exc

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for axis, metric in zip(axes.flat, METRIC_LABELS):
        values = metrics[metric]
        threshold = thresholds[metric]
        title = (
            METRIC_LABELS[metric]
            .replace("K-th", "{}-th".format(k))
            .replace("K nearest", "{} nearest".format(k))
        )
        x_max = float(np.percentile(values, hist_max_percentile))
        clipped_count = int(np.count_nonzero(values > x_max))
        axis.hist(values, bins=bins, range=(0, x_max), color="#4472C4", alpha=0.85)
        axis.axvline(
            threshold, color="#C00000", linestyle="--", linewidth=1.5,
            label="tail threshold = {:.6g}".format(threshold),
        )
        axis.set_title(title)
        axis.set_xlabel("Distance" if "_over_" not in metric else "Dimensionless ratio")
        axis.set_ylabel("Gaussian count")
        axis.set_xlim(0, x_max)
        axis.grid(alpha=0.2)
        axis.legend()
        if clipped_count:
            axis.text(
                0.98, 0.95,
                "x-axis: p{:g} ({} above range)".format(
                    hist_max_percentile, format(clipped_count, ",")
                ),
                transform=axis.transAxes, ha="right", va="top", fontsize=9,
            )
    fig.suptitle("Gaussian-center KNN metrics (K={})".format(k))
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def save_outliers(output_path, centers, metrics, thresholds):
    tails = {
        metric: values >= thresholds[metric] for metric, values in metrics.items()
    }
    combined_tail = np.logical_or.reduce(list(tails.values()))
    indices = np.flatnonzero(combined_tail)
    tiny = np.finfo(np.float32).tiny
    scores = np.maximum.reduce([
        metrics[metric][indices] / max(thresholds[metric], tiny)
        for metric in metrics
    ])
    indices = indices[np.argsort(scores)[::-1]]
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "index", "x", "y", "z",
            *metrics.keys(),
            *("{}_tail".format(metric) for metric in metrics),
            "tail_score",
        ))
        for index in indices:
            writer.writerow((
                int(index), *centers[index].tolist(),
                *(float(values[index]) for values in metrics.values()),
                *(bool(tail[index]) for tail in tails.values()),
                float(max(
                    metrics[metric][index] / max(thresholds[metric], tiny)
                    for metric in metrics
                )),
            ))
    return int(indices.size)


def annotated_ply_path(args, output_dir):
    return (args.annotated_ply or (output_dir / "point_cloud_knn.ply")).expanduser().resolve()


def write_annotated_ply(input_path, output_path, properties, chunk_size=100_000):
    """Copy a 3DGS PLY and append aligned float properties to every vertex."""
    from plyfile import PlyData, PlyElement

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Annotated PLY must not overwrite the trained input PLY")
    ply = PlyData.read(str(input_path), mmap="r")
    if not ply.elements or ply.elements[0].name != "vertex":
        raise ValueError("PLY has no leading vertex element: {}".format(input_path))
    vertex = ply.elements[0]
    source = vertex.data
    point_count = len(source)
    existing = set(source.dtype.names)
    for name, values in properties.items():
        if name in existing:
            raise ValueError("PLY already contains property {!r}".format(name))
        if values.shape != (point_count,):
            raise ValueError(
                "{} has shape {}; expected ({},)".format(name, values.shape, point_count)
            )
        if not np.isfinite(values).all():
            raise ValueError("{} contains non-finite values".format(name))

    scalar_dtype = source.dtype.fields["x"][0]
    annotated_dtype = np.dtype(
        source.dtype.descr + [(name, scalar_dtype) for name in properties]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        prefix=".knn_ply_", suffix=".tmp", dir=str(output_path.parent), delete=False
    )
    temp_path = Path(temp.name)
    temp.close()
    annotated = None
    try:
        annotated = np.memmap(
            str(temp_path), dtype=annotated_dtype, mode="w+", shape=(point_count,)
        )
        source_bytes = source.view(np.uint8).reshape(point_count, source.dtype.itemsize)
        annotated_bytes = annotated.view(np.uint8).reshape(
            point_count, annotated_dtype.itemsize
        )
        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            annotated_bytes[start:end, :source.dtype.itemsize] = source_bytes[start:end]
            for name, values in properties.items():
                annotated[name][start:end] = values[start:end]
            print(
                "PLY export: {:,}/{:,} Gaussians".format(end, point_count),
                end="\r", flush=True,
            )
        print()
        annotated.flush()
        annotated_vertex = PlyElement.describe(annotated, vertex.name)
        comments = list(ply.comments)
        for name in properties:
            comments.append("{}: Gaussian-center KNN analysis metric".format(name))
        PlyData(
            [annotated_vertex] + list(ply.elements[1:]),
            text=ply.text,
            byte_order=ply.byte_order,
            comments=comments,
            obj_info=list(ply.obj_info),
        ).write(str(output_path))
    finally:
        if annotated is not None:
            del annotated
        if temp_path.exists():
            os.unlink(str(temp_path))
    print("Annotated Gaussian PLY written to {}".format(output_path))


def export_saved_metrics(ply_path, output_dir, args):
    metrics_path = output_dir / "knn_metrics.npz"
    summary_path = output_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "--export-ply-only requires knn_metrics.npz and summary.json in {}".format(
                output_dir
            )
        )
    with summary_path.open() as stream:
        summary = json.load(stream)
    with np.load(str(metrics_path)) as arrays:
        properties = {}
        for k in summary["k_values"]:
            for metric in METRIC_LABELS:
                key = "{}_k{}".format(metric, k)
                # Support archives produced by the original two-metric script.
                legacy_key = {
                    "mean": "mean_distance_k{}".format(k),
                    "kth": "kth_distance_k{}".format(k),
                }.get(metric)
                if key in arrays:
                    properties[ply_property_name(metric, k)] = arrays[key]
                elif legacy_key is not None and legacy_key in arrays:
                    properties[ply_property_name(metric, k)] = arrays[legacy_key]
        write_annotated_ply(
            ply_path, annotated_ply_path(args, output_dir), properties,
            args.query_batch_size,
        )


def validate_args(args):
    k_values = sorted(set(args.k))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("Every K must be a positive integer")
    if not 0 <= args.tail_percentile <= 100:
        raise ValueError("--tail-percentile must be in [0, 100]")
    if not args.percentiles or any(p < 0 or p > 100 for p in args.percentiles):
        raise ValueError("Every --percentiles value must be in [0, 100]")
    if not 0 < args.hist_max_percentile <= 100:
        raise ValueError("--hist-max-percentile must be in (0, 100]")
    if args.bins <= 0 or args.query_batch_size <= 0:
        raise ValueError("--bins and --query-batch-size must be positive")
    if args.workers == 0 or args.workers < -1:
        raise ValueError("--workers must be -1 or a positive integer")
    if args.replot_only and args.export_ply_only:
        raise ValueError("--replot-only and --export-ply-only are mutually exclusive")
    if args.skip_annotated_ply and args.export_ply_only:
        raise ValueError("--skip-annotated-ply cannot be used with --export-ply-only")
    return k_values


def replot_existing(output_dir, args):
    metrics_path = output_dir / "knn_metrics.npz"
    summary_path = output_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "--replot-only requires knn_metrics.npz and summary.json in {}".format(output_dir)
        )
    with summary_path.open() as stream:
        summary = json.load(stream)
    with np.load(str(metrics_path)) as arrays:
        for k in summary["k_values"]:
            metric_summary = summary["metrics"][str(k)]
            values = {}
            thresholds = {}
            for metric in METRIC_LABELS:
                key = "{}_k{}".format(metric, k)
                if key not in arrays:
                    raise ValueError(
                        "Existing archive lacks {}; recompute analysis to plot all metrics".format(
                            key
                        )
                    )
                values[metric] = arrays[key]
                thresholds[metric] = metric_summary[metric]["tail_threshold"]
            save_histogram(
                output_dir / "histogram_k{}.png".format(k), k,
                values, thresholds,
                args.bins, args.hist_max_percentile,
            )
            print("Regenerated histogram for K={}".format(k))


def main():
    args = parse_args()
    k_values = validate_args(args)
    ply_path, output_dir, iteration = resolve_input(args)
    if args.replot_only:
        replot_existing(output_dir, args)
        return
    if args.export_ply_only:
        export_saved_metrics(ply_path, output_dir, args)
        return
    print("Loading Gaussian centers from {}".format(ply_path))
    centers, scales = load_gaussian_geometry(ply_path)
    print("Loaded {:,} centers; computing exact KNN for K={}".format(
        centers.shape[0], k_values
    ))
    metrics = compute_knn_metrics(
        centers, scales, k_values, args.query_batch_size, args.workers
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_ply": str(ply_path),
        "iteration": iteration,
        "gaussian_count": int(centers.shape[0]),
        "k_values": k_values,
        "distance": "Euclidean center distance",
        "scale": "exp(scale_0..2) from the trained 3DGS PLY",
        "self_neighbor_excluded": True,
        "tail_percentile": args.tail_percentile,
        "hist_max_percentile": args.hist_max_percentile,
        "ply_properties": {
            str(k): {
                metric: ply_property_name(metric, k) for metric in METRIC_LABELS
            }
            for k in k_values
        },
        "metrics": {},
    }
    arrays = {"index": np.arange(centers.shape[0], dtype=np.int64)}
    for k in k_values:
        values = metrics[k]
        thresholds = {
            metric: float(np.percentile(metric_values, args.tail_percentile))
            for metric, metric_values in values.items()
        }
        outlier_count = save_outliers(
            output_dir / "outliers_k{}.csv".format(k), centers,
            values, thresholds,
        )
        save_histogram(
            output_dir / "histogram_k{}.png".format(k), k,
            values, thresholds, args.bins, args.hist_max_percentile,
        )
        for metric, metric_values in values.items():
            arrays["{}_k{}".format(metric, k)] = metric_values
        summary["metrics"][str(k)] = {
            metric: dict(
                describe(metric_values, args.percentiles),
                tail_threshold=thresholds[metric],
            )
            for metric, metric_values in values.items()
        }
        summary["metrics"][str(k)].update({
            "outlier_rule": (
                "any of the six metrics >= its tail threshold"
            ),
            "outlier_count": outlier_count,
        })
        print("K={}: saved {:,} tail candidates".format(k, outlier_count))

    np.savez_compressed(str(output_dir / "knn_metrics.npz"), **arrays)
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    if not args.skip_annotated_ply:
        write_annotated_ply(
            ply_path,
            annotated_ply_path(args, output_dir),
            {
                ply_property_name(metric, k): metrics[k][metric]
                for k in k_values for metric in METRIC_LABELS
            },
            args.query_batch_size,
        )
    print("Analysis written to {}".format(output_dir))


if __name__ == "__main__":
    main()
