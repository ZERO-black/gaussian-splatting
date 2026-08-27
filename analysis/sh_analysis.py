"""Analyze covariance shape and view-dependent SH appearance in a 3DGS PLY.

The script preserves the PLY vertex order, writes one CSV row per Gaussian, and
creates three scatter plots with Pearson and Spearman correlation coefficients.
It uses the same activated-scale, scalar-first quaternion, SH property ordering,
and SH evaluator as the renderer/model implementation in this repository.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.covariance import quaternion_rotation_matrices
from analysis.knn import load_gaussian_geometry, resolve_input

from utils.sh_utils import eval_sh


PLOTS = (
    (
        "anisotropy_norm",
        "normalized_sh_energy",
        "anisotropy_vs_normalized_sh_energy.png",
        "Normalized anisotropy",
        "Normalized SH energy",
    ),
    (
        "anisotropy_norm",
        "sh_angular_variance",
        "anisotropy_vs_sh_angular_variance.png",
        "Normalized anisotropy",
        "SH angular variance",
    ),
    (
        "sh_energy",
        "sh_angular_variance",
        "sh_energy_vs_sh_angular_variance.png",
        "SH energy (l > 0)",
        "SH angular variance",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Gaussian anisotropy and SH appearance variation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-m", "--model-path", type=Path,
        help="Model directory containing point_cloud/iteration_*/point_cloud.ply.",
    )
    source.add_argument(
        "--ply-path", type=Path, help="A trained 3DGS point_cloud.ply.",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Iteration under --model-path; -1 selects the latest (default: -1).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Output directory (default: <model>/sh_analysis/iteration_<n>).",
    )
    parser.add_argument(
        "--num-directions", type=int, default=256,
        help="Number of deterministic uniform sphere samples (default: 256).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2_048,
        help="Gaussians evaluated per batch (default: 2048).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-12,
        help="Denominator epsilon for normalized SH energy (default: 1e-12).",
    )
    parser.add_argument(
        "--clip-percentiles", type=float, nargs=2, metavar=("LOW", "HIGH"),
        help=(
            "For each plot, show only points inside both axes' percentile range, "
            "for example 0.5 99.5. The CSV is never clipped."
        ),
    )
    parser.add_argument(
        "--log-scale", choices=("none", "x", "y", "both"), default="none",
        help="Use logarithmic x/y axes; non-positive plotted values are omitted.",
    )
    parser.add_argument(
        "--max-plot-points", type=int, default=200_000,
        help="Deterministically subsample scatter markers only (default: 200000).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed used only for plot subsampling (default: 0).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_directions <= 0 or args.batch_size <= 0:
        raise ValueError("--num-directions and --batch-size must be positive")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.max_plot_points <= 0:
        raise ValueError("--max-plot-points must be positive")
    if args.clip_percentiles is not None:
        low, high = args.clip_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("--clip-percentiles requires 0 <= LOW < HIGH <= 100")


def load_sh_coefficients(ply_path):
    """Load SH coefficients using GaussianModel.load_ply's property convention."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required; install the project environment first.") from exc

    ply = PlyData.read(str(ply_path), mmap="r")
    if not ply.elements or ply.elements[0].name != "vertex":
        raise ValueError("PLY has no leading vertex element: {}".format(ply_path))
    vertex = ply.elements[0]
    available = {prop.name for prop in vertex.properties}
    dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    missing_dc = sorted(set(dc_names) - available)
    if missing_dc:
        raise ValueError("PLY vertex element is missing {}".format(missing_dc))

    rest_names = sorted(
        (name for name in available if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError("SH f_rest property count must be divisible by 3")
    coefficient_count = 1 + len(rest_names) // 3
    sh_degree = math.isqrt(coefficient_count) - 1
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(
            "SH coefficient count {} is not a squared degree layout".format(
                coefficient_count
            )
        )
    if sh_degree > 4:
        raise ValueError(
            "SH degree {} is unsupported by utils.sh_utils.eval_sh (maximum: 4)".format(
                sh_degree
            )
        )

    point_count = len(vertex.data)
    coefficients = np.empty(
        (point_count, 3, coefficient_count), dtype=np.float32
    )
    for channel, name in enumerate(dc_names):
        coefficients[:, channel, 0] = np.asarray(vertex[name], dtype=np.float32)
    if rest_names:
        rest = np.column_stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in rest_names]
        )
        coefficients[:, :, 1:] = rest.reshape(point_count, 3, coefficient_count - 1)
    if not np.isfinite(coefficients).all():
        bad_count = int(np.count_nonzero(~np.isfinite(coefficients).all(axis=(1, 2))))
        raise ValueError("Found {} Gaussians with non-finite SH coefficients".format(bad_count))
    return coefficients, sh_degree


def fibonacci_sphere(sample_count):
    """Return deterministic, approximately equal-area unit directions."""
    indices = np.arange(sample_count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / sample_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def compute_anisotropy(scales, rotations, batch_size):
    """Compute eigenvalue-based shape features from R diag(scale^2) R^T."""
    point_count = scales.shape[0]
    eigenvalues = np.empty((point_count, 3), dtype=np.float64)
    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        rotation_matrices = quaternion_rotation_matrices(rotations[start:end])
        covariance = (
            rotation_matrices
            @ np.eye(3)[None, :, :] * np.square(scales[start:end])[:, None, :]
        ) @ np.swapaxes(rotation_matrices, 1, 2)
        eigenvalues[start:end] = np.linalg.eigvalsh(covariance)
    # eigvalsh is ascending. Activated scales are strictly positive, hence lambda_3 > 0.
    lambda_min, lambda_mid, lambda_max = eigenvalues.T
    anisotropy = lambda_max / lambda_min
    anisotropy_norm = (lambda_max - lambda_min) / (
        lambda_max + lambda_mid + lambda_min
    )
    return eigenvalues, anisotropy, anisotropy_norm


def compute_sh_features(coefficients, sh_degree, directions, batch_size, epsilon):
    """Compute coefficient energy and sampled angular color variance."""
    higher_order = coefficients[:, :, 1:].astype(np.float64, copy=False)
    sh_energy = np.einsum("nck,nck->n", higher_order, higher_order)
    dc = coefficients[:, :, 0].astype(np.float64, copy=False)
    dc_energy = np.einsum("nc,nc->n", dc, dc)
    normalized_energy = sh_energy / (dc_energy + epsilon)

    point_count = coefficients.shape[0]
    angular_variance = np.empty(point_count, dtype=np.float64)
    # Broadcast coefficients over samples and directions over Gaussians. eval_sh is
    # the repository renderer's real-SH polynomial implementation.
    eval_directions = directions[None, :, :]
    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        sh = coefficients[start:end, None, :, :].astype(np.float64, copy=False)
        colors = eval_sh(sh_degree, sh, eval_directions)
        centered = colors - colors.mean(axis=1, keepdims=True)
        angular_variance[start:end] = np.mean(
            np.sum(centered * centered, axis=2), axis=1
        )
        print(
            "SH direction evaluation: {:,}/{:,} Gaussians".format(end, point_count),
            end="\r", flush=True,
        )
    print()
    return sh_energy, normalized_energy, angular_variance


def _average_ranks(values):
    """Return one-based ranks with average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    starts, ends = boundaries[:-1], boundaries[1:]
    tied = ends - starts > 1
    for start, end in zip(starts[tied], ends[tied]):
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
    return ranks


def correlation(x, y):
    """Return Pearson and tie-aware Spearman correlations for finite pairs."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])
    return pearson, spearman


def save_csv(path, centers, eigenvalues, features):
    header = (
        "index", "x", "y", "z", "lambda_1", "lambda_2", "lambda_3",
        "anisotropy", "anisotropy_norm", "sh_energy",
        "normalized_sh_energy", "sh_angular_variance",
    )
    # eigenvalues is ascending internally; CSV follows lambda_1 >= lambda_2 >= lambda_3.
    columns = (
        np.arange(centers.shape[0]), centers[:, 0], centers[:, 1], centers[:, 2],
        eigenvalues[:, 2], eigenvalues[:, 1], eigenvalues[:, 0],
        features["anisotropy"], features["anisotropy_norm"],
        features["sh_energy"], features["normalized_sh_energy"],
        features["sh_angular_variance"],
    )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def _display_mask(x, y, clip_percentiles, log_scale):
    mask = np.isfinite(x) & np.isfinite(y)
    if clip_percentiles is not None and np.any(mask):
        low, high = clip_percentiles
        x_bounds = np.percentile(x[mask], (low, high))
        y_bounds = np.percentile(y[mask], (low, high))
        mask &= (x >= x_bounds[0]) & (x <= x_bounds[1])
        mask &= (y >= y_bounds[0]) & (y <= y_bounds[1])
    if log_scale in ("x", "both"):
        mask &= x > 0
    if log_scale in ("y", "both"):
        mask &= y > 0
    return mask


def save_plots(output_dir, features, clip_percentiles, log_scale, max_points, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save scatter plots.") from exc

    rng = np.random.default_rng(seed)
    for x_name, y_name, filename, x_label, y_label in PLOTS:
        x, y = features[x_name], features[y_name]
        mask = _display_mask(x, y, clip_percentiles, log_scale)
        displayed_indices = np.flatnonzero(mask)
        pearson, spearman = correlation(x[mask], y[mask])
        if displayed_indices.size > max_points:
            displayed_indices = np.sort(
                rng.choice(displayed_indices, size=max_points, replace=False)
            )

        fig, axis = plt.subplots(figsize=(7.5, 6.0))
        axis.scatter(
            x[displayed_indices], y[displayed_indices], s=3, alpha=0.25,
            color="#4472C4", edgecolors="none", rasterized=True,
        )
        if log_scale in ("x", "both"):
            axis.set_xscale("log")
        if log_scale in ("y", "both"):
            axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
        axis.text(
            0.03, 0.97,
            "Pearson r = {:.4f}\nSpearman ρ = {:.4f}\nn = {:,}".format(
                pearson, spearman, int(mask.sum())
            ),
            transform=axis.transAxes, va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        title = "{} vs. {}".format(y_label, x_label)
        if clip_percentiles is not None:
            title += " ({}-{} percentile clipped)".format(*clip_percentiles)
        axis.set_title(title)
        fig.tight_layout()
        fig.savefig(str(output_dir / filename), dpi=180)
        plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)
    ply_path, output_dir, iteration = resolve_input(args, "sh_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Gaussian geometry from {}".format(ply_path))
    centers, scales, rotations = load_gaussian_geometry(
        ply_path, include_rotations=True
    )
    coefficients, sh_degree = load_sh_coefficients(ply_path)
    if coefficients.shape[0] != centers.shape[0]:
        raise ValueError("Geometry and SH point counts do not match")
    print(
        "Loaded {:,} Gaussians (SH degree {}, iteration {})".format(
            centers.shape[0], sh_degree, iteration
        )
    )

    eigenvalues, anisotropy, anisotropy_norm = compute_anisotropy(
        scales, rotations, args.batch_size
    )
    directions = fibonacci_sphere(args.num_directions)
    sh_energy, normalized_sh_energy, angular_variance = compute_sh_features(
        coefficients, sh_degree, directions, args.batch_size, args.epsilon
    )
    features = {
        "anisotropy": anisotropy,
        "anisotropy_norm": anisotropy_norm,
        "sh_energy": sh_energy,
        "normalized_sh_energy": normalized_sh_energy,
        "sh_angular_variance": angular_variance,
    }

    csv_path = output_dir / "sh_features.csv"
    save_csv(csv_path, centers, eigenvalues, features)
    save_plots(
        output_dir, features, args.clip_percentiles, args.log_scale,
        args.max_plot_points, args.seed,
    )
    print("Saved features and plots to {}".format(output_dir))


if __name__ == "__main__":
    main()
