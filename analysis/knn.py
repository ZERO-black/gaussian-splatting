
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_K_VALUES = (5, 10, 20, 50)
DEFAULT_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)

# These names are part of the file-format contract with metric_core.py.
STATIC_METRIC_LABELS = {
    "mean": "Mean distance to K nearest neighbors",
    "kth": "Distance to the K-th nearest neighbor",
    "mean_over_max_scale": "Mean KNN distance / longest 3D scale axis",
    "kth_over_max_scale": "K-th KNN distance / longest 3D scale axis",
    "mean_over_mean_scale": "Mean KNN distance / mean 3D scale axis",
    "kth_over_mean_scale": "K-th KNN distance / mean 3D scale axis",
}
STATIC_METRICS = tuple(STATIC_METRIC_LABELS)


def ply_property_name(metric: str, k: int) -> str:
    """Return the uniform property name stored in an annotated Gaussian PLY."""
    return f"knn_{metric}_k{k}"


def archive_key(metric: str, k: int) -> str:
    return f"{metric}_k{k}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze exact Euclidean KNN distances in a trained 3DGS PLY."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-m", "--model-path", type=Path,
        help="Model directory containing point_cloud/iteration_*/point_cloud.ply.",
    )
    source.add_argument(
        "--ply-path", type=Path,
        help="Explicit trained point_cloud.ply path.",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Iteration under --model-path; -1 selects the latest.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
        help="One or more neighbor counts.",
    )
    parser.add_argument(
        "--tail-percentile", type=float, default=99.0,
        help="Outlier threshold percentile applied independently to each metric.",
    )
    parser.add_argument(
        "--percentiles", type=float, nargs="+", default=list(DEFAULT_PERCENTILES),
        help="Percentiles stored in summary.json.",
    )
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--hist-max-percentile", type=float, default=99.5,
        help="Histogram x-axis maximum percentile; affects visualization only.",
    )
    parser.add_argument(
        "--query-batch-size", type=int, default=100_000,
        help="Number of centers queried per cKDTree batch.",
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="cKDTree workers; -1 uses all CPU cores.",
    )
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Regenerate histograms from existing knn_metrics.npz + summary.json.",
    )
    parser.add_argument(
        "--export-ply-only", action="store_true",
        help="Regenerate the annotated PLY from existing knn_metrics.npz.",
    )
    parser.add_argument(
        "--annotated-ply", type=Path,
        help="Output PLY path; default: <output-dir>/point_cloud_knn.ply.",
    )
    parser.add_argument(
        "--skip-annotated-ply", action="store_true",
        help="Do not write the annotated PLY.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Default: <model>/knn_analysis/iteration_<n>.",
    )
    return parser.parse_args()


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


def latest_iteration(point_cloud_root: Path) -> int:
    iterations = []
    if point_cloud_root.is_dir():
        for path in point_cloud_root.iterdir():
            if not path.is_dir() or not path.name.startswith("iteration_"):
                continue
            try:
                iterations.append(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                pass
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories found in {point_cloud_root}")
    return max(iterations)


def iteration_from_path(ply_path: Path) -> int:
    if ply_path.parent.name.startswith("iteration_"):
        try:
            return int(ply_path.parent.name.rsplit("_", 1)[1])
        except ValueError:
            pass
    return -1


def resolve_input(args, analysis_directory="knn_analysis"):
    if args.ply_path is not None:
        ply_path = args.ply_path.expanduser().resolve()
        iteration = iteration_from_path(ply_path)
        default_output = ply_path.parent / analysis_directory
    else:
        model_path = args.model_path.expanduser().resolve()
        point_cloud_root = model_path / "point_cloud"
        iteration = args.iteration
        if iteration == -1:
            iteration = latest_iteration(point_cloud_root)
        ply_path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
        default_output = model_path / analysis_directory / f"iteration_{iteration}"

    if not ply_path.is_file():
        raise FileNotFoundError(f"Trained Gaussian PLY not found: {ply_path}")

    output_dir = (args.output_dir or default_output).expanduser().resolve()
    return ply_path, output_dir, iteration


def load_gaussian_geometry(ply_path: Path, include_rotations=False):
    """Load centers, activated scales, and optional normalized rotations."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required in the project environment") from exc

    ply = PlyData.read(str(ply_path), mmap="r")
    if not ply.elements or ply.elements[0].name != "vertex":
        raise ValueError(f"PLY has no leading vertex element: {ply_path}")

    vertex = ply.elements[0]
    available = {prop.name for prop in vertex.properties}
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if include_rotations:
        required.update({"rot_0", "rot_1", "rot_2", "rot_3"})
    missing = required - available
    if missing:
        raise ValueError(f"PLY is missing properties {sorted(missing)}: {ply_path}")

    centers = np.ascontiguousarray(
        np.column_stack((vertex["x"], vertex["y"], vertex["z"])),
        dtype=np.float64,
    )
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError(f"Expected non-empty Nx3 centers, got {centers.shape}")
    if not np.isfinite(centers).all():
        raise ValueError("Gaussian centers contain non-finite coordinates")

    log_scales = np.column_stack(
        (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])
    ).astype(np.float64, copy=False)
    scales = np.exp(log_scales)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Activated Gaussian scales contain invalid values")

    if not include_rotations:
        return centers, scales

    rotations = np.column_stack(
        (vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])
    ).astype(np.float64, copy=False)
    rotation_norms = np.linalg.norm(rotations, axis=1)
    if not np.isfinite(rotations).all() or np.any(rotation_norms == 0):
        raise ValueError("Gaussian rotations contain non-finite or zero-length values")
    rotations = rotations / rotation_norms[:, None]
    return centers, scales, rotations


def iter_knn_batches(centers, max_k, batch_size=100_000, workers=-1):
    """Yield exact KNN distances/indices while explicitly removing self matches."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for exact KNN analysis") from exc

    point_count = centers.shape[0]
    if max_k >= point_count:
        raise ValueError(
            f"Largest K ({max_k}) must be smaller than Gaussian count ({point_count})"
        )

    tree = cKDTree(centers)
    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        distances, indices = tree.query(
            centers[start:end], k=max_k + 1, workers=workers
        )

        query_indices = np.arange(start, end)[:, None]
        is_self = indices == query_indices
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


def compute_knn_metrics(centers, scales, k_values, batch_size=100_000, workers=-1):
    """Compute all view-independent metrics in one exact KNN pass."""
    k_values = sorted(set(k_values))
    point_count = centers.shape[0]
    max_k = max(k_values)

    metrics = {
        k: {
            metric: np.empty(point_count, dtype=np.float32)
            for metric in STATIC_METRICS
        }
        for k in k_values
    }

    max_scale = np.max(scales, axis=1)
    mean_scale = np.mean(scales, axis=1)

    for start, end, neighbor_distances, _ in iter_knn_batches(
        centers, max_k, batch_size, workers
    ):
        cumulative = np.cumsum(neighbor_distances, axis=1, dtype=np.float64)

        for k in k_values:
            mean_distance = cumulative[:, k - 1] / k
            kth_distance = neighbor_distances[:, k - 1]
            values = metrics[k]

            values["mean"][start:end] = mean_distance
            values["kth"][start:end] = kth_distance
            values["mean_over_max_scale"][start:end] = (
                mean_distance / max_scale[start:end]
            )
            values["kth_over_max_scale"][start:end] = (
                kth_distance / max_scale[start:end]
            )
            values["mean_over_mean_scale"][start:end] = (
                mean_distance / mean_scale[start:end]
            )
            values["kth_over_mean_scale"][start:end] = (
                kth_distance / mean_scale[start:end]
            )

        print(f"KNN query: {end:,}/{point_count:,} centers", end="\r", flush=True)

    print()
    return metrics


def percentile_label(percentile):
    return f"p{percentile:g}"


def validate_distribution_quantiles(quantiles):
    """Validate and canonicalize probabilities used as distribution thresholds."""
    if quantiles is None:
        return []

    quantiles = sorted(set(float(quantile) for quantile in quantiles))
    if any(not np.isfinite(quantile) or quantile < 0.0 or quantile > 1.0
           for quantile in quantiles):
        raise ValueError("Distribution quantiles must be finite values in [0, 1]")
    return quantiles


def distribution_quantile_label(quantile):
    """Return a filesystem-friendly label such as q0p9 for probability 0.9."""
    value = f"{float(quantile):.6g}"
    return f"q{value.replace('.', 'p')}"


def distribution_thresholds(values, quantiles):
    """Resolve quantile probabilities to values from the finite distribution."""
    quantiles = validate_distribution_quantiles(quantiles)
    values = np.asarray(values).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot resolve thresholds from an empty distribution")
    if not quantiles:
        return {}

    resolved = np.quantile(finite_values, quantiles)
    return {
        distribution_quantile_label(quantile): float(threshold)
        for quantile, threshold in zip(quantiles, resolved)
    }


def apply_lower_threshold_plateau(values, threshold):
    """Make every finite value below threshold share the threshold value."""
    values = np.asarray(values)
    result = values.copy()
    finite = np.isfinite(result)
    result[finite] = np.maximum(result[finite], threshold)
    return result


def describe(values, percentiles):
    percentile_values = np.percentile(values, percentiles)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            percentile_label(p): float(value)
            for p, value in zip(percentiles, percentile_values)
        },
    }


def metric_thresholds(metrics, tail_percentile):
    return {
        metric: float(np.percentile(values, tail_percentile))
        for metric, values in metrics.items()
    }


def save_histogram(output_path, k, metrics, thresholds, bins, hist_max_percentile):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save histograms") from exc

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for axis, metric in zip(axes.flat, STATIC_METRICS):
        values = metrics[metric]
        threshold = thresholds[metric]
        x_max = float(np.percentile(values, hist_max_percentile))
        clipped_count = int(np.count_nonzero(values > x_max))

        axis.hist(values, bins=bins, range=(0, x_max), color="#4472C4", alpha=0.85)
        axis.axvline(
            threshold,
            color="#C00000",
            linestyle="--",
            linewidth=1.5,
            label=f"tail threshold = {threshold:.6g}",
        )
        axis.set_title(
            STATIC_METRIC_LABELS[metric]
            .replace("K-th", f"{k}-th")
            .replace("K nearest", f"{k} nearest")
        )
        axis.set_xlabel("Distance" if "_over_" not in metric else "Dimensionless ratio")
        axis.set_ylabel("Gaussian count")
        axis.set_xlim(0, x_max)
        axis.grid(alpha=0.2)
        axis.legend()

        if clipped_count:
            axis.text(
                0.98,
                0.95,
                f"x-axis: p{hist_max_percentile:g} ({clipped_count:,} above range)",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

    fig.suptitle(f"Gaussian-center KNN metrics (K={k})")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def save_outliers(output_path, centers, metrics, thresholds):
    """Save Gaussians in the upper tail of any static metric."""
    tails = {
        metric: values >= thresholds[metric]
        for metric, values in metrics.items()
    }
    combined_tail = np.logical_or.reduce(list(tails.values()))
    indices = np.flatnonzero(combined_tail)

    tiny = np.finfo(np.float32).tiny
    if indices.size:
        scores = np.maximum.reduce([
            metrics[metric][indices] / max(thresholds[metric], tiny)
            for metric in STATIC_METRICS
        ])
        indices = indices[np.argsort(scores)[::-1]]

    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "index", "x", "y", "z",
            *STATIC_METRICS,
            *(f"{metric}_tail" for metric in STATIC_METRICS),
            "tail_score",
        ))
        for index in indices:
            tail_score = max(
                metrics[metric][index] / max(thresholds[metric], tiny)
                for metric in STATIC_METRICS
            )
            writer.writerow((
                int(index),
                *centers[index].tolist(),
                *(float(metrics[metric][index]) for metric in STATIC_METRICS),
                *(bool(tails[metric][index]) for metric in STATIC_METRICS),
                float(tail_score),
            ))

    return int(indices.size)


def annotated_ply_path(args, output_dir):
    path = args.annotated_ply or (output_dir / "point_cloud_knn.ply")
    return path.expanduser().resolve()


def write_annotated_ply(input_path, output_path, properties, chunk_size=100_000):
    """Copy the trained PLY and append aligned scalar metric properties."""
    from plyfile import PlyData, PlyElement

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Annotated PLY must not overwrite the trained input PLY")

    ply = PlyData.read(str(input_path), mmap="r")
    if not ply.elements or ply.elements[0].name != "vertex":
        raise ValueError(f"PLY has no leading vertex element: {input_path}")

    vertex = ply.elements[0]
    source = vertex.data
    point_count = len(source)
    existing = set(source.dtype.names)

    for name, values in properties.items():
        if name in existing:
            raise ValueError(f"PLY already contains property {name!r}")
        if values.shape != (point_count,):
            raise ValueError(f"{name} has shape {values.shape}; expected ({point_count},)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    scalar_dtype = source.dtype.fields["x"][0]
    annotated_dtype = np.dtype(
        source.dtype.descr + [(name, scalar_dtype) for name in properties]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        prefix=".knn_ply_",
        suffix=".tmp",
        dir=str(output_path.parent),
        delete=False,
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
            annotated_bytes[start:end, : source.dtype.itemsize] = source_bytes[start:end]
            for name, values in properties.items():
                annotated[name][start:end] = values[start:end]
            print(f"PLY export: {end:,}/{point_count:,} Gaussians", end="\r", flush=True)
        print()

        annotated.flush()
        annotated_vertex = PlyElement.describe(annotated, vertex.name)
        comments = list(ply.comments)
        comments.extend(
            f"{name}: Gaussian-center KNN analysis metric" for name in properties
        )
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

    print(f"Annotated Gaussian PLY written to {output_path}")


def save_analysis(output_dir, ply_path, iteration, centers, metrics, args, k_values):
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {"index": np.arange(centers.shape[0], dtype=np.int64)}
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
            str(k): {metric: ply_property_name(metric, k) for metric in STATIC_METRICS}
            for k in k_values
        },
        "metrics": {},
        "outliers": {},
    }

    for k in k_values:
        values = metrics[k]
        thresholds = metric_thresholds(values, args.tail_percentile)

        for metric in STATIC_METRICS:
            arrays[archive_key(metric, k)] = values[metric]

        summary["metrics"][str(k)] = {
            metric: {
                **describe(values[metric], args.percentiles),
                "tail_threshold": thresholds[metric],
            }
            for metric in STATIC_METRICS
        }

        outlier_count = save_outliers(
            output_dir / f"outliers_k{k}.csv",
            centers,
            values,
            thresholds,
        )
        summary["outliers"][str(k)] = {
            "rule": "any static metric >= its own tail threshold",
            "count": outlier_count,
        }

        save_histogram(
            output_dir / f"histogram_k{k}.png",
            k,
            values,
            thresholds,
            args.bins,
            args.hist_max_percentile,
        )
        print(f"K={k}: saved {outlier_count:,} tail candidates")

    np.savez_compressed(str(output_dir / "knn_metrics.npz"), **arrays)
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    if not args.skip_annotated_ply:
        properties = {
            ply_property_name(metric, k): metrics[k][metric]
            for k in k_values
            for metric in STATIC_METRICS
        }
        write_annotated_ply(
            ply_path,
            annotated_ply_path(args, output_dir),
            properties,
            args.query_batch_size,
        )


def load_saved_metrics(output_dir):
    metrics_path = output_dir / "knn_metrics.npz"
    summary_path = output_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Expected knn_metrics.npz and summary.json in {output_dir}"
        )

    with summary_path.open() as stream:
        summary = json.load(stream)

    arrays = np.load(str(metrics_path))
    return summary, arrays


def replot_existing(output_dir, args):
    summary, arrays = load_saved_metrics(output_dir)
    try:
        for k in summary["k_values"]:
            values = {
                metric: arrays[archive_key(metric, k)]
                for metric in STATIC_METRICS
            }
            thresholds = {
                metric: summary["metrics"][str(k)][metric]["tail_threshold"]
                for metric in STATIC_METRICS
            }
            save_histogram(
                output_dir / f"histogram_k{k}.png",
                k,
                values,
                thresholds,
                args.bins,
                args.hist_max_percentile,
            )
            print(f"Regenerated histogram for K={k}")
    finally:
        arrays.close()


def export_saved_metrics(ply_path, output_dir, args):
    summary, arrays = load_saved_metrics(output_dir)
    try:
        properties = {}
        for k in summary["k_values"]:
            for metric in STATIC_METRICS:
                key = archive_key(metric, k)
                if key not in arrays:
                    raise KeyError(f"Saved metric archive is missing {key}")
                properties[ply_property_name(metric, k)] = arrays[key]

        write_annotated_ply(
            ply_path,
            annotated_ply_path(args, output_dir),
            properties,
            args.query_batch_size,
        )
    finally:
        arrays.close()


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

    print(f"Loading Gaussian centers from {ply_path}")
    centers, scales = load_gaussian_geometry(ply_path)
    print(f"Loaded {centers.shape[0]:,} centers; computing exact KNN for K={k_values}")

    metrics = compute_knn_metrics(
        centers,
        scales,
        k_values,
        args.query_batch_size,
        args.workers,
    )
    save_analysis(output_dir, ply_path, iteration, centers, metrics, args, k_values)
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
