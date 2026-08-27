from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from analysis.knn import STATIC_METRICS, iter_knn_batches, ply_property_name


# -----------------------------------------------------------------------------
# Metric registry
# -----------------------------------------------------------------------------

CAMERA_DEPTH_METRICS = {
    "kth_over_camera_depth",
    "mean_over_camera_depth",
}

PROJECTED_FOOTPRINT_METRICS = {
    "max_projected_gap_over_footprint",
    "mean_projected_gap_over_footprint",
}

PROJECTED_AXIS_METRICS = {
    "max_projected_gap_over_major_axis",
    "mean_projected_gap_over_major_axis",
    "max_projected_gap_over_minor_axis",
    "mean_projected_gap_over_minor_axis",
}

VIEW_PERP_SUPPORT_METRICS = {
    "max_view_perp_over_support",
    "mean_view_perp_over_support",
}

COVARIANCE_METRICS = {
    "long_axis_consistency",
    "short_axis_consistency",
}

VIEW_DEPENDENT_PAIR_METRICS = (
    PROJECTED_FOOTPRINT_METRICS
    | PROJECTED_AXIS_METRICS
    | VIEW_PERP_SUPPORT_METRICS
)
VIEW_DEPENDENT_METRICS = CAMERA_DEPTH_METRICS | VIEW_DEPENDENT_PAIR_METRICS

METRIC_DESCRIPTIONS = {
    # Static metrics written by analysis.knn.
    "kth": "3D Euclidean distance to the K-th nearest Gaussian",
    "mean": "mean 3D Euclidean distance to the K nearest Gaussians",
    "kth_over_max_scale": "K-th 3D KNN distance / longest 3D Gaussian scale",
    "mean_over_max_scale": "mean 3D KNN distance / longest 3D Gaussian scale",
    "kth_over_mean_scale": "K-th 3D KNN distance / mean 3D Gaussian scale",
    "mean_over_mean_scale": "mean 3D KNN distance / mean 3D Gaussian scale",
    # View-aware versions of raw KNN distance.
    "kth_over_camera_depth": "K-th 3D KNN distance / camera-space forward depth",
    "mean_over_camera_depth": "mean 3D KNN distance / camera-space forward depth",
    # Experimental pair metrics.
    "max_projected_gap_over_footprint": (
        "maximum projected neighbor-center gap / paired directional 3-sigma footprints"
    ),
    "mean_projected_gap_over_footprint": (
        "mean projected neighbor-center gap / paired directional 3-sigma footprints"
    ),
    "max_projected_gap_over_major_axis": (
        "maximum projected neighbor-center gap / center projected major-axis 1-sigma"
    ),
    "mean_projected_gap_over_major_axis": (
        "mean projected neighbor-center gap / center projected major-axis 1-sigma"
    ),
    "max_projected_gap_over_minor_axis": (
        "maximum projected neighbor-center gap / center projected minor-axis 1-sigma"
    ),
    "mean_projected_gap_over_minor_axis": (
        "mean projected neighbor-center gap / center projected minor-axis 1-sigma"
    ),
    "max_view_perp_over_support": (
        "maximum view-perpendicular 3D neighbor spacing / paired directional 3-sigma support"
    ),
    "mean_view_perp_over_support": (
        "mean view-perpendicular 3D neighbor spacing / paired directional 3-sigma support"
    ),
    # Optional metrics from a separate covariance-analysis PLY.
    "long_axis_consistency": (
        "mean absolute cosine similarity of longest covariance axes over KNN"
    ),
    "short_axis_consistency": (
        "mean absolute cosine similarity of shortest covariance axes over KNN"
    ),
}

ALL_METRICS = tuple(sorted(METRIC_DESCRIPTIONS))
DEFAULT_RENDER_METRICS = (
    "mean",
    "kth",
    "mean_over_camera_depth",
    "kth_over_camera_depth",
)


@dataclass
class MetricContext:
    neighbor_indices: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------------
# Static PLY-backed metrics
# -----------------------------------------------------------------------------


def stored_metric_values(gaussians, metric, k):
    if metric not in STATIC_METRICS:
        raise KeyError(f"Not a stored static metric: {metric}")
    return gaussians.get_knn_metric(ply_property_name(metric, k)).reshape(-1)


def covariance_metric_values(gaussians, metric, k):
    if metric not in COVARIANCE_METRICS:
        raise KeyError(f"Not a covariance metric: {metric}")
    return gaussians.get_knn_metric(f"knn_{metric}_k{k}").reshape(-1)


def camera_depth_base_values(gaussians, metric, k):
    if metric == "kth_over_camera_depth":
        return stored_metric_values(gaussians, "kth", k)
    if metric == "mean_over_camera_depth":
        return stored_metric_values(gaussians, "mean", k)
    raise KeyError(metric)


def metric_is_available(gaussians, metric, k):
    """Return whether the loaded PLY contains the stored inputs required by metric."""
    if metric in STATIC_METRICS:
        return ply_property_name(metric, k) in gaussians.available_knn_metrics

    if metric in CAMERA_DEPTH_METRICS:
        base = "kth" if metric.startswith("kth") else "mean"
        return metric_is_available(gaussians, base, k)

    if metric in COVARIANCE_METRICS:
        return f"knn_{metric}_k{k}" in gaussians.available_knn_metrics

    # Pair metrics recompute their KNN indices directly from xyz, so no stored KNN
    # property is required beyond a normal trained Gaussian PLY.
    if metric in VIEW_DEPENDENT_PAIR_METRICS:
        return True

    return False


# -----------------------------------------------------------------------------
# Shared KNN context and normalization
# -----------------------------------------------------------------------------


def compute_knn_indices(gaussians, k, batch_size=100_000):
    centers = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    indices = np.empty((centers.shape[0], k), dtype=np.int32)

    for start, end, _, batch_indices in iter_knn_batches(
        centers, k, batch_size, -1
    ):
        indices[start:end] = batch_indices
        print(
            f"View-dependent KNN query: {end:,}/{centers.shape[0]:,} centers",
            end="\r",
            flush=True,
        )
    print()
    return torch.from_numpy(indices)


def prepare_metric_context(gaussians, metrics, k):
    if isinstance(metrics, str):
        metrics = [metrics]
    needs_pair_neighbors = any(
        metric in VIEW_DEPENDENT_PAIR_METRICS for metric in metrics
    )
    needs_view_geometry = any(metric in VIEW_DEPENDENT_METRICS for metric in metrics)
    if not needs_pair_neighbors and not needs_view_geometry:
        return MetricContext()

    return MetricContext(
        neighbor_indices=(
            compute_knn_indices(gaussians, k) if needs_pair_neighbors else None
        ),
        covariance=(gaussians.get_covariance().detach() if needs_view_geometry else None),
    )


def normalization_bounds(
    values,
    valid,
    percentile_min=1.0,
    percentile_max=99.5,
    value_min=None,
    value_max=None,
):
    """Compute linear clipping bounds from finite, valid metric values."""
    samples = valid_metric_samples(values, valid)
    if samples.numel() == 0:
        raise ValueError("No finite values are available for normalization")

    requested = []
    if value_min is None:
        requested.append(percentile_min / 100.0)
    if value_max is None:
        requested.append(percentile_max / 100.0)

    quantiles = []
    if requested:
        quantiles = metric_quantiles(samples, requested)

    q_index = 0
    lower = float(value_min) if value_min is not None else float(quantiles[q_index])
    if value_min is None:
        q_index += 1
    upper = float(value_max) if value_max is not None else float(quantiles[q_index])

    if upper <= lower:
        if value_min is None and value_max is None:
            padding = max(abs(lower) * 1e-6, torch.finfo(samples.dtype).eps)
            return lower - padding, upper + padding
        raise ValueError(
            f"Normalization maximum ({upper}) must be greater than minimum ({lower})"
        )
    return lower, upper


def metric_quantiles(samples, probabilities):
    """Compute quantiles, falling back when torch hits its tensor-size limit."""
    samples = torch.as_tensor(samples).reshape(-1)
    samples = samples[torch.isfinite(samples)]
    if samples.numel() == 0:
        raise ValueError("Cannot compute quantiles from an empty distribution")

    q = torch.as_tensor(
        probabilities,
        dtype=samples.dtype,
        device=samples.device,
    )
    try:
        return torch.quantile(samples, q).tolist()
    except RuntimeError as exc:
        if "input tensor is too large" not in str(exc):
            raise
        return np.quantile(
            samples.detach().cpu().numpy(),
            q.detach().cpu().numpy(),
        ).tolist()


def valid_metric_mask(values, valid):
    """Return the single mask used by every metric-statistics path."""
    if values.shape != valid.shape:
        raise ValueError(
            f"Metric values and valid mask must match: {values.shape} != {valid.shape}"
        )
    if valid.dtype != torch.bool:
        raise TypeError(f"Metric valid mask must be bool, got {valid.dtype}")
    return valid & torch.isfinite(values)


def valid_metric_samples(values, valid):
    """Select finite metric values for Gaussians where the metric is defined."""
    return values[valid_metric_mask(values, valid)]


def colorize(values, lower, upper):
    """Apply the shared linear blue-to-red map without metric-specific overrides."""
    values = values.reshape(-1)
    lower = torch.as_tensor(lower, dtype=values.dtype, device=values.device)
    upper = torch.as_tensor(upper, dtype=values.dtype, device=values.device)
    denominator = torch.clamp_min(upper - lower, torch.finfo(values.dtype).eps)
    normalized = torch.clamp((values - lower) / denominator, 0.0, 1.0)
    red = torch.clamp(1.5 - torch.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = torch.clamp(1.5 - torch.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = torch.clamp(1.5 - torch.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return torch.stack((red, green, blue), dim=1)


# -----------------------------------------------------------------------------
# Camera-space helpers
# -----------------------------------------------------------------------------


def camera_space_points(gaussians, view):
    xyz = gaussians.get_xyz
    matrix = view.world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
    return xyz @ matrix[:3, :3] + matrix[3, :3]


def camera_depth_values(gaussians, view, base_values, covariance=None):
    camera_points = camera_space_points(gaussians, view)
    depth = camera_points[:, 2]
    valid = center_gaussian_validity(
        gaussians, view, covariance=covariance
    )
    valid &= torch.isfinite(base_values) & torch.isfinite(depth)

    values = torch.zeros_like(base_values)
    values[valid] = base_values[valid] / depth[valid]
    valid &= torch.isfinite(values)
    return values, valid


# -----------------------------------------------------------------------------
# Projected Gaussian geometry
# -----------------------------------------------------------------------------


def projected_gaussian_geometry(
    gaussians, view, covariance=None, batch_size=100_000
):
    """Project centers/covariances with the same Jacobian model as 3DGS rasterization."""
    xyz = gaussians.get_xyz
    if covariance is None:
        covariance = gaussians.get_covariance()

    point_count = xyz.shape[0]
    device, dtype = xyz.device, xyz.dtype
    centers_2d = torch.empty((point_count, 2), dtype=dtype, device=device)
    covariance_2d = torch.empty((point_count, 3), dtype=dtype, device=device)
    positive_depth = torch.empty(point_count, dtype=torch.bool, device=device)
    in_view = torch.empty(point_count, dtype=torch.bool, device=device)

    view_matrix = view.world_view_transform.to(device=device, dtype=dtype)
    world_to_camera = view_matrix[:3, :3].transpose(0, 1)
    translation = view_matrix[3, :3]

    focal_x = view.image_width / (2.0 * math.tan(view.FoVx * 0.5))
    focal_y = view.image_height / (2.0 * math.tan(view.FoVy * 0.5))
    center_x = (view.image_width - 1.0) * 0.5
    center_y = (view.image_height - 1.0) * 0.5
    limit_x = 1.3 * math.tan(view.FoVx * 0.5)
    limit_y = 1.3 * math.tan(view.FoVy * 0.5)

    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        camera_points = xyz[start:end] @ view_matrix[:3, :3] + translation
        x, y, z = camera_points.unbind(dim=1)
        depth_valid = z > torch.finfo(dtype).eps
        safe_z = torch.where(depth_valid, z, torch.ones_like(z))
        positive_depth[start:end] = depth_valid

        centers_2d[start:end, 0] = focal_x * x / safe_z + center_x
        centers_2d[start:end, 1] = focal_y * y / safe_z + center_y

        clamped_x = torch.clamp(x / safe_z, -limit_x, limit_x) * safe_z
        clamped_y = torch.clamp(y / safe_z, -limit_y, limit_y) * safe_z

        jacobian = torch.zeros((end - start, 2, 3), dtype=dtype, device=device)
        jacobian[:, 0, 0] = focal_x / safe_z
        jacobian[:, 0, 2] = -focal_x * clamped_x / (safe_z * safe_z)
        jacobian[:, 1, 1] = focal_y / safe_z
        jacobian[:, 1, 2] = -focal_y * clamped_y / (safe_z * safe_z)
        world_jacobian = jacobian @ world_to_camera

        compact = covariance[start:end]
        world_cov = torch.empty((end - start, 3, 3), dtype=dtype, device=device)
        world_cov[:, 0, 0] = compact[:, 0]
        world_cov[:, 0, 1] = world_cov[:, 1, 0] = compact[:, 1]
        world_cov[:, 0, 2] = world_cov[:, 2, 0] = compact[:, 2]
        world_cov[:, 1, 1] = compact[:, 3]
        world_cov[:, 1, 2] = world_cov[:, 2, 1] = compact[:, 4]
        world_cov[:, 2, 2] = compact[:, 5]

        projected = world_jacobian @ world_cov @ world_jacobian.transpose(1, 2)
        projected[:, 0, 0] += 0.3
        projected[:, 1, 1] += 0.3

        covariance_2d[start:end, 0] = projected[:, 0, 0]
        covariance_2d[start:end, 1] = projected[:, 0, 1]
        covariance_2d[start:end, 2] = projected[:, 1, 1]

        midpoint = 0.5 * (projected[:, 0, 0] + projected[:, 1, 1])
        determinant = (
            projected[:, 0, 0] * projected[:, 1, 1]
            - projected[:, 0, 1] * projected[:, 1, 0]
        )
        largest_eigenvalue = midpoint + torch.sqrt(
            torch.clamp_min(midpoint * midpoint - determinant, 0.1)
        )
        radius = torch.ceil(3.0 * torch.sqrt(torch.clamp_min(largest_eigenvalue, 0.0)))

        projected_centers = centers_2d[start:end]
        in_view[start:end] = (
            depth_valid
            & (projected_centers[:, 0] + radius >= 0)
            & (projected_centers[:, 0] - radius < view.image_width)
            & (projected_centers[:, 1] + radius >= 0)
            & (projected_centers[:, 1] - radius < view.image_height)
        )

    return centers_2d, covariance_2d, positive_depth, in_view


def center_gaussian_validity(gaussians, view, covariance=None, batch_size=100_000):
    """Return the shared positive-depth, footprint-intersects-image mask."""
    _, _, _, in_view = projected_gaussian_geometry(
        gaussians,
        view,
        covariance=covariance,
        batch_size=batch_size,
    )
    return in_view


# -----------------------------------------------------------------------------
# View-dependent pair metrics
# -----------------------------------------------------------------------------


def reduce_pair_ratios(ratios, pair_valid, reduction):
    valid_count = pair_valid.sum(dim=1)
    if reduction == "mean":
        values = torch.where(
            valid_count > 0,
            torch.where(pair_valid, ratios, 0.0).sum(dim=1)
            / torch.clamp_min(valid_count, 1),
            0.0,
        )
    elif reduction == "max":
        values = torch.where(pair_valid, ratios, 0.0).max(dim=1).values
    else:
        raise ValueError(f"Unknown pair reduction: {reduction}")
    return values, valid_count > 0


def projected_footprint_values(
    gaussians, view, neighbor_indices, metric, covariance=None, batch_size=100_000
):
    centers_2d, covariance_2d, positive_depth, in_view = projected_gaussian_geometry(
        gaussians, view, covariance=covariance
    )
    reduction = "mean" if metric.startswith("mean_") else "max"
    point_count = centers_2d.shape[0]
    values = torch.zeros(point_count, dtype=centers_2d.dtype, device=centers_2d.device)
    metric_valid = torch.zeros(point_count, dtype=torch.bool, device=centers_2d.device)
    eps = torch.finfo(centers_2d.dtype).eps

    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        neighbors = neighbor_indices[start:end].to(device=centers_2d.device, dtype=torch.long)

        delta = centers_2d[neighbors] - centers_2d[start:end, None, :]
        center_distance = torch.linalg.vector_norm(delta, dim=2)
        direction = delta / torch.clamp_min(center_distance[..., None], eps)
        dx, dy = direction.unbind(dim=2)

        center_cov = covariance_2d[start:end, None, :]
        neighbor_cov = covariance_2d[neighbors]
        center_var = (
            center_cov[..., 0] * dx * dx
            + 2.0 * center_cov[..., 1] * dx * dy
            + center_cov[..., 2] * dy * dy
        )
        neighbor_var = (
            neighbor_cov[..., 0] * dx * dx
            + 2.0 * neighbor_cov[..., 1] * dx * dy
            + neighbor_cov[..., 2] * dy * dy
        )
        support = 3.0 * (
            torch.sqrt(torch.clamp_min(center_var, 0.0))
            + torch.sqrt(torch.clamp_min(neighbor_var, 0.0))
        )
        ratios = center_distance / torch.clamp_min(support, eps)
        pair_valid = in_view[start:end, None] & positive_depth[neighbors]
        pair_valid &= torch.isfinite(ratios)

        batch_values, batch_valid = reduce_pair_ratios(ratios, pair_valid, reduction)
        values[start:end] = batch_values
        metric_valid[start:end] = batch_valid

    return values, metric_valid


def projected_axis_values(
    gaussians, view, neighbor_indices, metric, covariance=None, batch_size=100_000
):
    centers_2d, covariance_2d, positive_depth, in_view = projected_gaussian_geometry(
        gaussians, view, covariance=covariance
    )
    reduction = "mean" if metric.startswith("mean_") else "max"
    major_axis = "major_axis" in metric

    xx, xy, yy = covariance_2d.unbind(dim=1)
    midpoint = 0.5 * (xx + yy)
    half_difference = 0.5 * (xx - yy)
    eigen_offset = torch.sqrt(
        torch.clamp_min(half_difference * half_difference + xy * xy, 0.0)
    )
    eigenvalue = midpoint + eigen_offset if major_axis else midpoint - eigen_offset

    eps = torch.finfo(centers_2d.dtype).eps
    axis_scale = torch.sqrt(torch.clamp_min(eigenvalue, eps))
    point_count = centers_2d.shape[0]
    values = torch.zeros(point_count, dtype=centers_2d.dtype, device=centers_2d.device)
    metric_valid = torch.zeros(point_count, dtype=torch.bool, device=centers_2d.device)

    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        neighbors = neighbor_indices[start:end].to(device=centers_2d.device, dtype=torch.long)
        delta = centers_2d[neighbors] - centers_2d[start:end, None, :]
        projected_distances = torch.linalg.vector_norm(delta, dim=2)
        ratios = projected_distances / axis_scale[start:end, None]
        pair_valid = in_view[start:end, None] & positive_depth[neighbors]
        pair_valid &= torch.isfinite(ratios)

        batch_values, batch_valid = reduce_pair_ratios(ratios, pair_valid, reduction)
        values[start:end] = batch_values
        metric_valid[start:end] = batch_valid

    return values, metric_valid


def view_perp_support_values(
    gaussians, view, neighbor_indices, metric, covariance=None, batch_size=100_000
):
    xyz = gaussians.get_xyz
    if covariance is None:
        covariance = gaussians.get_covariance()

    reduction = "mean" if metric.startswith("mean_") else "max"
    device, dtype = xyz.device, xyz.dtype
    eps = torch.finfo(dtype).eps

    camera_center = view.camera_center.to(device=device, dtype=dtype)
    camera_rays = xyz - camera_center.unsqueeze(0)
    ray_norms = torch.linalg.vector_norm(camera_rays, dim=1)
    view_directions = camera_rays / torch.clamp_min(ray_norms[:, None], eps)
    center_valid = center_gaussian_validity(
        gaussians, view, covariance=covariance, batch_size=batch_size
    ) & (ray_norms > eps)

    point_count = xyz.shape[0]
    values = torch.zeros(point_count, dtype=dtype, device=device)
    metric_valid = torch.zeros(point_count, dtype=torch.bool, device=device)

    for start in range(0, point_count, batch_size):
        end = min(start + batch_size, point_count)
        neighbors = neighbor_indices[start:end].to(device=device, dtype=torch.long)

        displacement = xyz[neighbors] - xyz[start:end, None, :]
        center_view = view_directions[start:end, None, :]
        parallel = (displacement * center_view).sum(dim=2, keepdim=True) * center_view
        perpendicular = displacement - parallel
        perpendicular_distance = torch.linalg.vector_norm(perpendicular, dim=2)
        direction = perpendicular / torch.clamp_min(perpendicular_distance[..., None], eps)
        ux, uy, uz = direction.unbind(dim=2)

        center_cov = covariance[start:end, None, :]
        neighbor_cov = covariance[neighbors]
        center_var = (
            center_cov[..., 0] * ux * ux
            + 2.0 * center_cov[..., 1] * ux * uy
            + 2.0 * center_cov[..., 2] * ux * uz
            + center_cov[..., 3] * uy * uy
            + 2.0 * center_cov[..., 4] * uy * uz
            + center_cov[..., 5] * uz * uz
        )
        neighbor_var = (
            neighbor_cov[..., 0] * ux * ux
            + 2.0 * neighbor_cov[..., 1] * ux * uy
            + 2.0 * neighbor_cov[..., 2] * ux * uz
            + neighbor_cov[..., 3] * uy * uy
            + 2.0 * neighbor_cov[..., 4] * uy * uz
            + neighbor_cov[..., 5] * uz * uz
        )
        support = 3.0 * (
            torch.sqrt(torch.clamp_min(center_var, 0.0))
            + torch.sqrt(torch.clamp_min(neighbor_var, 0.0))
        )
        ratios = perpendicular_distance / torch.clamp_min(support, eps)
        # The definition is over the original K 3D neighbors. Neighbor visibility
        # must not change either the mean denominator or the maximum candidate set.
        pair_valid = center_valid[start:end, None].expand_as(ratios) & torch.isfinite(
            ratios
        )

        batch_values, batch_valid = reduce_pair_ratios(ratios, pair_valid, reduction)
        values[start:end] = batch_values
        metric_valid[start:end] = batch_valid

    return values, metric_valid


# -----------------------------------------------------------------------------
# Unified metric entry point
# -----------------------------------------------------------------------------


def compute_metric_values(gaussians, view, metric, k, context=None):
    """Return ``(values, valid_mask)`` for one metric."""
    if metric not in METRIC_DESCRIPTIONS:
        raise KeyError(f"Unknown metric: {metric}")

    if metric in STATIC_METRICS:
        values = stored_metric_values(gaussians, metric, k)
        return values, torch.isfinite(values)

    if metric in COVARIANCE_METRICS:
        values = covariance_metric_values(gaussians, metric, k)
        return values, torch.isfinite(values)

    if metric in CAMERA_DEPTH_METRICS:
        if view is None:
            raise ValueError(f"{metric} requires a camera view")
        return camera_depth_values(
            gaussians,
            view,
            camera_depth_base_values(gaussians, metric, k),
            covariance=None if context is None else context.covariance,
        )

    if metric in VIEW_DEPENDENT_PAIR_METRICS:
        if view is None or context is None or context.neighbor_indices is None:
            raise ValueError(f"{metric} requires a camera view and pair context")

        if metric in PROJECTED_FOOTPRINT_METRICS:
            function = projected_footprint_values
        elif metric in PROJECTED_AXIS_METRICS:
            function = projected_axis_values
        else:
            function = view_perp_support_values

        return function(
            gaussians,
            view,
            context.neighbor_indices,
            metric,
            covariance=context.covariance,
        )

    raise KeyError(metric)
