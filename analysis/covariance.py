"""Analyze local consistency of trained Gaussian covariance principal axes.

For each Gaussian, this script measures the mean absolute cosine similarity
between its longest/shortest covariance eigenvector and those of its K nearest
Gaussian centers. The PLY vertex order is preserved.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.knn import (
    DEFAULT_PERCENTILES,
    describe,
    iter_knn_batches,
    load_gaussian_geometry,
    ply_property_name,
    resolve_input,
    write_annotated_ply,
)


METRIC_LABELS = {
    "long_axis_consistency": "Longest covariance-axis consistency",
    "short_axis_consistency": "Shortest covariance-axis consistency",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze local consistency of 3DGS covariance eigenvectors."
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
        "--anomaly-percentile", type=float, default=1,
        help="Save low-consistency candidates at this percentile (default: 1).",
    )
    parser.add_argument(
        "--percentiles", type=float, nargs="+", default=list(DEFAULT_PERCENTILES),
        help="Percentiles included in summary.json.",
    )
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins.")
    parser.add_argument(
        "--query-batch-size", type=int, default=100_000,
        help="Centers queried at once to bound temporary memory (default: 100000).",
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="cKDTree workers; -1 uses all CPU cores (default: -1).",
    )
    parser.add_argument(
        "--export-ply-only", action="store_true",
        help="Add saved covariance metrics to a PLY without recomputing KNN.",
    )
    parser.add_argument(
        "--annotated-ply", type=Path,
        help="Annotated PLY path (default: <output-dir>/point_cloud_covariance.ply).",
    )
    parser.add_argument(
        "--skip-annotated-ply", action="store_true",
        help="Do not write the Gaussian PLY containing consistency properties.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Output directory (default: <model>/covariance_analysis/iteration_<n>).",
    )
    return parser.parse_args()


def validate_args(args):
    k_values = sorted(set(args.k))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("Every K must be a positive integer")
    if not 0 <= args.anomaly_percentile <= 100:
        raise ValueError("--anomaly-percentile must be in [0, 100]")
    if not args.percentiles or any(p < 0 or p > 100 for p in args.percentiles):
        raise ValueError("Every --percentiles value must be in [0, 100]")
    if args.bins <= 0 or args.query_batch_size <= 0:
        raise ValueError("--bins and --query-batch-size must be positive")
    if args.workers == 0 or args.workers < -1:
        raise ValueError("--workers must be -1 or a positive integer")
    if args.skip_annotated_ply and args.export_ply_only:
        raise ValueError("--skip-annotated-ply cannot be used with --export-ply-only")
    return k_values


def quaternion_rotation_matrices(rotations):
    """Convert normalized scalar-first quaternions to 3DGS rotation matrices."""
    w, x, y, z = rotations.T
    matrices = np.empty((rotations.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def extract_covariance_eigenvectors(scales, rotations):
    """Return all three covariance eigenvectors in ascending eigenvalue order.

    The columns of R are the covariance eigenvectors. Selecting columns by
    scale is therefore an analytic eigendecomposition that avoids materializing
    every 3x3 covariance matrix.
    """
    matrices = quaternion_rotation_matrices(rotations)
    scale_order = np.argsort(scales, axis=1, kind="stable")
    return np.ascontiguousarray(
        np.take_along_axis(matrices, scale_order[:, None, :], axis=2),
        dtype=np.float64,
    )


def extract_principal_axes(scales, rotations):
    """Extract longest and shortest eigenvectors of R diag(scale^2) R^T."""
    eigenvectors = extract_covariance_eigenvectors(scales, rotations)
    return (
        np.ascontiguousarray(eigenvectors[:, :, -1]),
        np.ascontiguousarray(eigenvectors[:, :, 0]),
    )


def compute_covariance_metrics(
    centers, scales, rotations, k_values, batch_size, workers
):
    """Compute mean sign-invariant axis cosine similarity per Gaussian."""
    k_values = sorted(set(k_values))
    point_count, max_k = centers.shape[0], max(k_values)
    longest, shortest = extract_principal_axes(scales, rotations)
    metrics = {
        k: {
            metric: np.empty(point_count, dtype=np.float32)
            for metric in METRIC_LABELS
        }
        for k in k_values
    }

    for start, end, _, neighbor_indices in iter_knn_batches(
        centers, max_k, batch_size, workers
    ):
        long_similarity = np.clip(np.abs(np.einsum(
            "bi,bki->bk", longest[start:end], longest[neighbor_indices]
        )), 0.0, 1.0)
        short_similarity = np.clip(np.abs(np.einsum(
            "bi,bki->bk", shortest[start:end], shortest[neighbor_indices]
        )), 0.0, 1.0)
        long_cumulative = np.cumsum(long_similarity, axis=1, dtype=np.float64)
        short_cumulative = np.cumsum(short_similarity, axis=1, dtype=np.float64)
        for k, values in metrics.items():
            values["long_axis_consistency"][start:end] = long_cumulative[:, k - 1] / k
            values["short_axis_consistency"][start:end] = short_cumulative[:, k - 1] / k
        print(
            "Covariance KNN query: {:,}/{:,} centers".format(end, point_count),
            end="\r", flush=True,
        )
    print()
    return metrics


def save_histogram(output_path, k, metrics, thresholds, bins):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Histogram rendering is unavailable because matplotlib or one of its "
            "binary dependencies could not be imported: {}".format(exc)
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric in zip(axes, METRIC_LABELS):
        values = metrics[metric]
        threshold = thresholds[metric]
        axis.hist(values, bins=bins, range=(0, 1), color="#4472C4", alpha=0.85)
        axis.axvline(
            threshold, color="#C00000", linestyle="--", linewidth=1.5,
            label="anomaly threshold = {:.6g}".format(threshold),
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Mean absolute cosine similarity")
        axis.set_ylabel("Gaussian count")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Local covariance-axis consistency (K={})".format(k))
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def save_anomalies(output_path, centers, metrics, thresholds):
    anomalies = {
        metric: values <= thresholds[metric] for metric, values in metrics.items()
    }
    indices = np.flatnonzero(np.logical_or.reduce(list(anomalies.values())))
    if indices.size:
        scores = np.minimum.reduce([
            metrics[metric][indices] for metric in metrics
        ])
        indices = indices[np.argsort(scores)]
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "index", "x", "y", "z", *metrics.keys(),
            *("{}_anomaly".format(metric) for metric in metrics),
        ))
        for index in indices:
            writer.writerow((
                int(index), *centers[index].tolist(),
                *(float(values[index]) for values in metrics.values()),
                *(bool(mask[index]) for mask in anomalies.values()),
            ))
    return int(indices.size)


def annotated_ply_path(args, output_dir):
    return (
        args.annotated_ply or (output_dir / "point_cloud_covariance.ply")
    ).expanduser().resolve()


def export_saved_metrics(ply_path, output_dir, args):
    metrics_path = output_dir / "covariance_metrics.npz"
    summary_path = output_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "--export-ply-only requires covariance_metrics.npz and summary.json in {}".format(
                output_dir
            )
        )
    with summary_path.open() as stream:
        summary = json.load(stream)
    with np.load(str(metrics_path)) as arrays:
        properties = {
            ply_property_name(metric, k): arrays["{}_k{}".format(metric, k)]
            for k in summary["k_values"] for metric in METRIC_LABELS
        }
        write_annotated_ply(
            ply_path, annotated_ply_path(args, output_dir), properties,
            args.query_batch_size,
        )


def main():
    args = parse_args()
    k_values = validate_args(args)
    ply_path, output_dir, iteration = resolve_input(args, "covariance_analysis")
    if args.export_ply_only:
        export_saved_metrics(ply_path, output_dir, args)
        return

    print("Loading Gaussian covariance geometry from {}".format(ply_path))
    centers, scales, rotations = load_gaussian_geometry(
        ply_path, include_rotations=True
    )
    print("Loaded {:,} Gaussians; computing consistency for K={}".format(
        centers.shape[0], k_values
    ))
    metrics = compute_covariance_metrics(
        centers, scales, rotations, k_values, args.query_batch_size, args.workers
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_ply": str(ply_path),
        "iteration": iteration,
        "gaussian_count": int(centers.shape[0]),
        "k_values": k_values,
        "neighborhood": "Euclidean center-distance KNN",
        "self_neighbor_excluded": True,
        "covariance": "R diag(exp(scale_0..2)^2) R^T",
        "similarity": "mean over KNN of abs(dot(center_axis, neighbor_axis))",
        "anomaly_percentile": args.anomaly_percentile,
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
            metric: float(np.percentile(metric_values, args.anomaly_percentile))
            for metric, metric_values in values.items()
        }
        anomaly_count = save_anomalies(
            output_dir / "anomalies_k{}.csv".format(k), centers, values, thresholds
        )
        for metric, metric_values in values.items():
            arrays["{}_k{}".format(metric, k)] = metric_values
        summary["metrics"][str(k)] = {
            metric: dict(
                describe(metric_values, args.percentiles),
                anomaly_threshold=thresholds[metric],
            )
            for metric, metric_values in values.items()
        }
        summary["metrics"][str(k)].update({
            "anomaly_rule": "either consistency metric <= its low-percentile threshold",
            "anomaly_count": anomaly_count,
        })
        print("K={}: saved {:,} local orientation candidates".format(k, anomaly_count))

    # Persist irreplaceable analysis results before optional visualization. A
    # broken matplotlib/Pillow installation must never discard a completed KNN.
    np.savez_compressed(str(output_dir / "covariance_metrics.npz"), **arrays)
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
    for k in k_values:
        thresholds = {
            metric: summary["metrics"][str(k)][metric]["anomaly_threshold"]
            for metric in METRIC_LABELS
        }
        save_histogram(
            output_dir / "histogram_k{}.png".format(k),
            k, metrics[k], thresholds, args.bins,
        )
    print("Analysis written to {}".format(output_dir))


if __name__ == "__main__":
    main()
