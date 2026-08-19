"""Analyze raw Euclidean KNN distances between trained Gaussian centers.

PLY vertex order is preserved, so every reported index identifies the same
Gaussian in the trained model.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)


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
        help="Flag points at or above this percentile for either metric (default: 99).",
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


def resolve_input(args):
    if args.ply_path is not None:
        ply_path = args.ply_path.expanduser().resolve()
        iteration = _iteration_from_path(ply_path)
        default_output = ply_path.parent / "knn_analysis"
    else:
        model_path = args.model_path.expanduser().resolve()
        point_cloud_root = model_path / "point_cloud"
        iteration = args.iteration
        if iteration == -1:
            iteration = _latest_iteration(point_cloud_root)
        ply_path = point_cloud_root / ("iteration_{}".format(iteration)) / "point_cloud.ply"
        default_output = model_path / "knn_analysis" / ("iteration_{}".format(iteration))
    if not ply_path.is_file():
        raise FileNotFoundError("Trained Gaussian PLY not found: {}".format(ply_path))
    return ply_path, (args.output_dir or default_output).expanduser().resolve(), iteration


def load_gaussian_centers(ply_path):
    """Read xyz from the project's standard PLY without allocating CUDA tensors."""
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
    if not {"x", "y", "z"}.issubset(available):
        raise ValueError("PLY vertex element does not contain x, y, z: {}".format(ply_path))
    centers = np.ascontiguousarray(
        np.column_stack((vertex["x"], vertex["y"], vertex["z"])), dtype=np.float64
    )
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("Expected a non-empty Nx3 center array, got {}".format(centers.shape))
    if not np.isfinite(centers).all():
        bad_count = int(np.count_nonzero(~np.isfinite(centers).all(axis=1)))
        raise ValueError("Found {} centers with non-finite coordinates".format(bad_count))
    return centers


def compute_knn_metrics(centers, k_values, batch_size, workers):
    """Compute exact KNN metrics with a tree and bounded query memory."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for exact configurable-K queries; update the project environment."
        ) from exc

    k_values = sorted(set(k_values))
    point_count, max_k = centers.shape[0], max(k_values)
    if max_k >= point_count:
        raise ValueError(
            "Largest K ({}) must be smaller than Gaussian count ({})".format(max_k, point_count)
        )
    tree = cKDTree(centers)
    metrics = {
        k: (np.empty(point_count, dtype=np.float32), np.empty(point_count, dtype=np.float32))
        for k in k_values
    }
    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        try:
            distances, _ = tree.query(centers[start:end], k=max_k + 1, workers=workers)
        except TypeError:  # scipy < 1.6
            distances, _ = tree.query(centers[start:end], k=max_k + 1, n_jobs=workers)
        neighbor_distances = distances[:, 1:]  # exclude the query center itself
        cumulative = np.cumsum(neighbor_distances, axis=1, dtype=np.float64)
        for k, (mean_distance, kth_distance) in metrics.items():
            mean_distance[start:end] = cumulative[:, k - 1] / k
            kth_distance[start:end] = neighbor_distances[:, k - 1]
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


def save_histogram(
    output_path, k, mean_distance, kth_distance, mean_threshold, kth_threshold,
    bins, hist_max_percentile
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save histogram PNGs.") from exc

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plots = (
        (axes[0], mean_distance, mean_threshold, "Mean distance to {} nearest neighbors".format(k)),
        (axes[1], kth_distance, kth_threshold, "Distance to neighbor {}".format(k)),
    )
    for axis, values, threshold, title in plots:
        x_max = float(np.percentile(values, hist_max_percentile))
        clipped_count = int(np.count_nonzero(values > x_max))
        axis.hist(values, bins=bins, range=(0, x_max), color="#4472C4", alpha=0.85)
        axis.axvline(
            threshold, color="#C00000", linestyle="--", linewidth=1.5,
            label="tail threshold = {:.6g}".format(threshold),
        )
        axis.set_title(title)
        axis.set_xlabel("Raw Euclidean distance")
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
    fig.suptitle("Gaussian-center KNN distance distribution (K={})".format(k))
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def save_outliers(
    output_path, centers, mean_distance, kth_distance, mean_threshold, kth_threshold
):
    mean_tail = mean_distance >= mean_threshold
    kth_tail = kth_distance >= kth_threshold
    indices = np.flatnonzero(mean_tail | kth_tail)
    tiny = np.finfo(np.float32).tiny
    mean_scale, kth_scale = max(mean_threshold, tiny), max(kth_threshold, tiny)
    scores = np.maximum(mean_distance[indices] / mean_scale, kth_distance[indices] / kth_scale)
    indices = indices[np.argsort(scores)[::-1]]
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "index", "x", "y", "z", "mean_knn_distance", "kth_neighbor_distance",
            "mean_tail", "kth_tail", "tail_score",
        ))
        for index in indices:
            writer.writerow((
                int(index), *centers[index].tolist(),
                float(mean_distance[index]), float(kth_distance[index]),
                bool(mean_tail[index]), bool(kth_tail[index]),
                float(max(mean_distance[index] / mean_scale, kth_distance[index] / kth_scale)),
            ))
    return int(indices.size)


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
            thresholds = metric_summary["tail_thresholds"]
            save_histogram(
                output_dir / "histogram_k{}.png".format(k), k,
                arrays["mean_distance_k{}".format(k)],
                arrays["kth_distance_k{}".format(k)],
                thresholds["mean_knn_distance"],
                thresholds["kth_neighbor_distance"],
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
    print("Loading Gaussian centers from {}".format(ply_path))
    centers = load_gaussian_centers(ply_path)
    print("Loaded {:,} centers; computing exact KNN for K={}".format(
        centers.shape[0], k_values
    ))
    metrics = compute_knn_metrics(
        centers, k_values, args.query_batch_size, args.workers
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_ply": str(ply_path),
        "iteration": iteration,
        "gaussian_count": int(centers.shape[0]),
        "k_values": k_values,
        "distance": "raw Euclidean center distance",
        "self_neighbor_excluded": True,
        "tail_percentile": args.tail_percentile,
        "hist_max_percentile": args.hist_max_percentile,
        "metrics": {},
    }
    arrays = {"index": np.arange(centers.shape[0], dtype=np.int64)}
    for k in k_values:
        mean_distance, kth_distance = metrics[k]
        mean_threshold = float(np.percentile(mean_distance, args.tail_percentile))
        kth_threshold = float(np.percentile(kth_distance, args.tail_percentile))
        outlier_count = save_outliers(
            output_dir / "outliers_k{}.csv".format(k), centers,
            mean_distance, kth_distance, mean_threshold, kth_threshold,
        )
        save_histogram(
            output_dir / "histogram_k{}.png".format(k), k,
            mean_distance, kth_distance, mean_threshold, kth_threshold, args.bins,
            args.hist_max_percentile,
        )
        arrays["mean_distance_k{}".format(k)] = mean_distance
        arrays["kth_distance_k{}".format(k)] = kth_distance
        summary["metrics"][str(k)] = {
            "mean_knn_distance": describe(mean_distance, args.percentiles),
            "kth_neighbor_distance": describe(kth_distance, args.percentiles),
            "tail_thresholds": {
                "mean_knn_distance": mean_threshold,
                "kth_neighbor_distance": kth_threshold,
            },
            "outlier_rule": (
                "mean_knn_distance >= threshold OR kth_neighbor_distance >= threshold"
            ),
            "outlier_count": outlier_count,
        }
        print("K={}: saved {:,} tail candidates".format(k, outlier_count))

    np.savez_compressed(str(output_dir / "knn_metrics.npz"), **arrays)
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("Analysis written to {}".format(output_dir))


if __name__ == "__main__":
    main()
