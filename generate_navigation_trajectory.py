"""Generate a collision-free A->B trajectory with a deliberately poor initial view.

The generator never uses an object or scene centroid. Positions are sampled in
free space around a standalone reference camera. A sparse-view proxy then finds
endpoint headings that look acceptable at A and B while their interpolation
exposes relatively under-reconstructed geometry at interior waypoints.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from trajectory.reference_camera import ReferenceCamera, load_reference_cameras


@dataclass
class GaussianSample:
    centers: np.ndarray
    radii: np.ndarray
    sparsity: np.ndarray
    tree: cKDTree


@dataclass
class SafeSegment:
    positions: np.ndarray
    sampled_clearance: float
    maximum_ground_distance: float


@dataclass
class ViewScenario:
    segment: SafeSegment
    directions: np.ndarray
    score: float
    initial_scores: np.ndarray
    alternative_scores: np.ndarray
    alternative_directions: np.ndarray
    coverages: np.ndarray
    start_heading_degrees: float
    end_heading_degrees: float
    minimum_forward_alignment: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return vector / norm


def _camera_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(np.asarray(forward, dtype=np.float64))
    up = _normalize(np.asarray(up, dtype=np.float64))
    forward = forward - np.dot(forward, up) * up
    forward = _normalize(forward)
    right = _normalize(np.cross(forward, up))
    down = _normalize(np.cross(forward, right))
    return np.stack((right, down, forward), axis=1).astype(np.float32)


def _yaw_directions(up: np.ndarray, reference_forward: np.ndarray, count: int):
    if count < 4:
        raise ValueError("view.heading_count must be at least 4")
    up = _normalize(np.asarray(up, dtype=np.float64))
    base = np.asarray(reference_forward, dtype=np.float64)
    base = _normalize(base - np.dot(base, up) * up)
    side = _normalize(np.cross(up, base))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return angles, _directions_for_angles(angles, base, side)


def _directions_for_angles(
    angles: np.ndarray, base: np.ndarray, side: np.ndarray
) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float64)
    directions = (
        np.cos(angles[:, None]) * base[None]
        + np.sin(angles[:, None]) * side[None]
    )
    return directions.astype(np.float32)


def _interpolate_angle(start: float, end: float, alphas: np.ndarray) -> np.ndarray:
    delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
    return start + alphas * delta


def _validate_config(config) -> None:
    if config.sample.count < 1000:
        raise ValueError("sample.count must be at least 1000")
    if config.sample.knn_k < 1 or config.sample.knn_k >= config.sample.count:
        raise ValueError("sample.knn_k must be positive and smaller than sample.count")
    if not 0.0 <= config.sample.opacity_min <= 1.0:
        raise ValueError("sample.opacity_min must be in [0, 1]")
    if not 0.0 < config.sample.max_scale_percentile <= 100.0:
        raise ValueError("sample.max_scale_percentile must be in (0, 100]")
    if config.path.num_waypoints < 3:
        raise ValueError("path.num_waypoints must be at least 3")
    lower = np.asarray(config.path.xy_min, dtype=np.float64)
    upper = np.asarray(config.path.xy_max, dtype=np.float64)
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("path.xy_min/xy_max must define a valid relative XY box")
    if not 0 < config.path.minimum_endpoint_distance < config.path.maximum_endpoint_distance:
        raise ValueError("Endpoint distances must satisfy 0 < minimum < maximum")
    if config.path.segment_candidates < 1 or config.path.max_attempts < 1:
        raise ValueError("Path candidate counts must be positive")
    if config.collision.segment_samples < 3:
        raise ValueError("collision.segment_samples must be at least 3")
    if config.collision.clearance <= 0 or config.collision.ground_support_radius <= 0:
        raise ValueError("Collision clearance and support radius must be positive")
    if not 0 < config.view.minimum_coverage <= 1:
        raise ValueError("view.minimum_coverage must be in (0, 1]")
    if not 0 <= config.view.maximum_heading_from_travel_degrees < 90:
        raise ValueError(
            "view.maximum_heading_from_travel_degrees must be in [0, 90)"
        )
    grid = tuple(config.view.grid)
    if len(grid) != 2 or min(grid) < 2:
        raise ValueError("view.grid must contain width and height >= 2")
    selection = str(getattr(config.view, "selection", "best"))
    if selection not in {"best", "random"}:
        raise ValueError("view.selection must be 'best' or 'random'")


def load_gaussian_sample(config) -> GaussianSample:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to sample the Gaussian PLY") from exc

    ply_path = Path(config.input.ply_path).expanduser().resolve()
    vertex = PlyData.read(str(ply_path), mmap="r")["vertex"]
    point_count = len(vertex.data)
    if point_count < 2:
        raise ValueError(f"Gaussian PLY is empty: {ply_path}")
    sample_count = min(int(config.sample.count), point_count)
    rng = np.random.default_rng(int(config.seed))
    indices = np.sort(rng.choice(point_count, size=sample_count, replace=False))

    centers = np.column_stack(
        (vertex["x"][indices], vertex["y"][indices], vertex["z"][indices])
    ).astype(np.float64)
    opacity = _sigmoid(np.asarray(vertex["opacity"][indices], dtype=np.float64))
    scales = np.exp(
        np.column_stack(
            (
                vertex["scale_0"][indices],
                vertex["scale_1"][indices],
                vertex["scale_2"][indices],
            )
        ).astype(np.float64)
    )
    radii = np.max(scales, axis=1)
    sparsity_property = getattr(config.sample, "sparsity_property", None)
    if sparsity_property is not None:
        available = {prop.name for prop in vertex.properties}
        if sparsity_property not in available:
            raise ValueError(
                f"Gaussian PLY has no {sparsity_property!r}; available KNN "
                f"properties: {sorted(name for name in available if name.startswith('knn_'))}"
            )
        raw_sparsity = np.asarray(
            vertex[sparsity_property][indices], dtype=np.float64
        )
    else:
        raw_sparsity = None
    finite = (
        np.isfinite(centers).all(axis=1)
        & np.isfinite(opacity)
        & np.isfinite(radii)
        & (opacity >= float(config.sample.opacity_min))
        & (radii > 0)
    )
    if raw_sparsity is not None:
        finite &= np.isfinite(raw_sparsity) & (raw_sparsity >= 0)
    centers, radii = centers[finite], radii[finite]
    if raw_sparsity is not None:
        raw_sparsity = raw_sparsity[finite]
    if len(centers) <= config.sample.knn_k:
        raise ValueError("Too few valid Gaussian samples after opacity filtering")
    scale_limit = np.percentile(radii, float(config.sample.max_scale_percentile))
    keep = radii <= scale_limit
    centers = np.ascontiguousarray(centers[keep])
    radii = np.ascontiguousarray(radii[keep])
    if raw_sparsity is not None:
        raw_sparsity = np.ascontiguousarray(raw_sparsity[keep])

    tree = cKDTree(centers)
    if raw_sparsity is None:
        distances, _ = tree.query(
            centers,
            k=int(config.sample.knn_k) + 1,
            workers=int(config.sample.workers),
        )
        sparsity = distances[:, 1:].mean(axis=1)
        source = "sampled KNN"
    else:
        sparsity = raw_sparsity
        source = str(sparsity_property)
    configured_tail = getattr(config.sample, "sparsity_tail_threshold", None)
    tail = (
        float(configured_tail)
        if configured_tail is not None
        else np.percentile(sparsity, float(config.sample.sparsity_tail_percentile))
    )
    if not np.isfinite(tail) or tail <= 0:
        raise ValueError("Could not derive a positive sparsity normalization")
    sparsity = np.clip(sparsity / tail, 0.0, 1.0).astype(np.float32)
    print(
        f"Gaussian proxy: {len(centers):,}/{point_count:,} samples, "
        f"source={source}, tail={tail:.6g}, scale cap={scale_limit:.6g}"
    )
    return GaussianSample(centers, radii.astype(np.float32), sparsity, tree)


def _segment_positions(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return (1.0 - alphas[:, None]) * start + alphas[:, None] * end


def _segment_is_safe(
    positions: np.ndarray,
    sample: GaussianSample,
    reference: ReferenceCamera,
    config,
) -> Tuple[bool, float, float]:
    nearest_distances, nearest_indices = sample.tree.query(
        positions,
        k=int(config.collision.neighbor_count),
        workers=int(config.sample.workers),
    )
    nearest_distances = np.atleast_2d(nearest_distances)
    nearest_indices = np.atleast_2d(nearest_indices)
    if nearest_distances.shape[0] != len(positions):
        nearest_distances = nearest_distances.T
        nearest_indices = nearest_indices.T
    radii = np.minimum(
        sample.radii[nearest_indices] * float(config.collision.scale_multiplier),
        float(config.collision.maximum_gaussian_radius),
    )
    effective_clearance = nearest_distances - radii
    minimum_clearance = float(effective_clearance.min())
    if minimum_clearance < float(config.collision.clearance):
        return False, minimum_clearance, math.inf

    camera_height = float(reference.position[2])
    min_ground_z = camera_height - float(config.collision.maximum_ground_drop)
    max_ground_z = camera_height - float(config.collision.minimum_ground_drop)
    ground_mask = (
        (sample.centers[:, 2] >= min_ground_z)
        & (sample.centers[:, 2] <= max_ground_z)
    )
    ground_xy = sample.centers[ground_mask, :2]
    if len(ground_xy) == 0:
        return False, minimum_clearance, math.inf
    ground_tree = cKDTree(ground_xy)
    ground_distance, _ = ground_tree.query(
        positions[:, :2], k=1, workers=int(config.sample.workers)
    )
    maximum_ground_distance = float(np.max(ground_distance))
    safe = maximum_ground_distance <= float(config.collision.ground_support_radius)
    return safe, minimum_clearance, maximum_ground_distance


def sample_safe_segments(
    sample: GaussianSample,
    reference: ReferenceCamera,
    config,
) -> list:
    rng = np.random.default_rng(int(config.seed) + 1)
    lower = reference.position[:2] + np.asarray(config.path.xy_min, dtype=np.float64)
    upper = reference.position[:2] + np.asarray(config.path.xy_max, dtype=np.float64)
    segments = []
    for _ in range(int(config.path.max_attempts)):
        start_xy = rng.uniform(lower, upper)
        end_xy = rng.uniform(lower, upper)
        distance = np.linalg.norm(end_xy - start_xy)
        if not (
            float(config.path.minimum_endpoint_distance)
            <= distance
            <= float(config.path.maximum_endpoint_distance)
        ):
            continue
        start = np.array([start_xy[0], start_xy[1], reference.position[2]])
        end = np.array([end_xy[0], end_xy[1], reference.position[2]])
        collision_positions = _segment_positions(
            start, end, int(config.collision.segment_samples)
        )
        safe, clearance, ground_distance = _segment_is_safe(
            collision_positions, sample, reference, config
        )
        if not safe:
            continue
        waypoint_positions = _segment_positions(
            start, end, int(config.path.num_waypoints)
        ).astype(np.float32)
        segments.append(SafeSegment(waypoint_positions, clearance, ground_distance))
        if len(segments) >= int(config.path.segment_candidates):
            break
    if not segments:
        raise RuntimeError(
            "Could not find a supported collision-free line segment; widen the "
            "path bounds or relax the collision settings"
        )
    print(f"Safe straight-line candidates: {len(segments)}")
    return segments


def verify_segments_against_full_ply(segments: list, config) -> list:
    """Reject lines that approach any meaningful Gaussian in the full PLY.

    Candidate discovery uses a small proxy sample. This final streaming pass is
    deliberately stricter: it reads every Gaussian once and checks all line
    segments, so a thin obstacle omitted by the proxy cannot be accepted.
    """
    if not bool(config.collision.exact_verification):
        return segments
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required for exact collision verification") from exc

    ply_path = Path(config.input.ply_path).expanduser().resolve()
    vertex = PlyData.read(str(ply_path), mmap="r")["vertex"]
    minimum_clearances = np.full(len(segments), np.inf, dtype=np.float64)
    starts = np.stack([segment.positions[0] for segment in segments]).astype(np.float64)
    ends = np.stack([segment.positions[-1] for segment in segments]).astype(np.float64)
    vectors = ends - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    chunk_size = int(config.collision.exact_chunk_size)
    if chunk_size < 1:
        raise ValueError("collision.exact_chunk_size must be positive")

    for chunk_start in range(0, len(vertex.data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(vertex.data))
        centers = np.column_stack(
            (
                vertex["x"][chunk_start:chunk_end],
                vertex["y"][chunk_start:chunk_end],
                vertex["z"][chunk_start:chunk_end],
            )
        ).astype(np.float64)
        opacity = _sigmoid(
            np.asarray(vertex["opacity"][chunk_start:chunk_end], dtype=np.float64)
        )
        valid = np.isfinite(centers).all(axis=1) & (
            opacity >= float(config.sample.opacity_min)
        )
        if not valid.any():
            continue
        centers = centers[valid]
        radii = np.exp(
            np.column_stack(
                (
                    vertex["scale_0"][chunk_start:chunk_end][valid],
                    vertex["scale_1"][chunk_start:chunk_end][valid],
                    vertex["scale_2"][chunk_start:chunk_end][valid],
                )
            ).astype(np.float64)
        ).max(axis=1)
        radii = np.minimum(
            radii * float(config.collision.scale_multiplier),
            float(config.collision.maximum_gaussian_radius),
        )
        finite = np.isfinite(radii)
        centers, radii = centers[finite], radii[finite]

        for index, (start, vector, squared_length) in enumerate(
            zip(starts, vectors, squared_lengths)
        ):
            relative = centers - start
            alpha = np.clip((relative @ vector) / squared_length, 0.0, 1.0)
            closest = start + alpha[:, None] * vector
            center_distance = np.linalg.norm(centers - closest, axis=1)
            effective_clearance = float(np.min(center_distance - radii))
            minimum_clearances[index] = min(
                minimum_clearances[index], effective_clearance
            )
        report_stride = max(chunk_size, 5_000_000)
        crossed_report_boundary = (
            chunk_start // report_stride != chunk_end // report_stride
        )
        if crossed_report_boundary or chunk_end == len(vertex.data):
            print(
                f"Exact collision scan: {chunk_end:,}/{len(vertex.data):,}",
                flush=True,
            )
    print()

    verified = []
    for segment, clearance in zip(segments, minimum_clearances):
        if clearance < float(config.collision.clearance):
            continue
        segment.sampled_clearance = float(clearance)
        verified.append(segment)
    print(
        f"Exact collision-free candidates: {len(verified)}/{len(segments)} "
        f"(required clearance={float(config.collision.clearance):g})"
    )
    return verified


def _view_score(
    position: np.ndarray,
    direction: np.ndarray,
    up: np.ndarray,
    reference: ReferenceCamera,
    sample: GaussianSample,
    config,
) -> Tuple[float, float]:
    rotation = _camera_rotation(direction, up).astype(np.float64)
    relative = sample.centers - position[None]
    camera_points = relative @ rotation
    depth = camera_points[:, 2]
    near = float(config.view.near)
    far = float(config.view.far)
    mask = (depth > near) & (depth < far)
    if not mask.any():
        return float(config.view.empty_cost), 0.0

    camera_points = camera_points[mask]
    depth = depth[mask]
    point_sparsity = sample.sparsity[mask]
    normalized_x = camera_points[:, 0] / depth
    normalized_y = camera_points[:, 1] / depth
    tan_x = math.tan(reference.FoVx / 2.0)
    tan_y = math.tan(reference.FoVy / 2.0)
    visible = (np.abs(normalized_x) <= tan_x) & (np.abs(normalized_y) <= tan_y)
    if not visible.any():
        return float(config.view.empty_cost), 0.0

    normalized_x = normalized_x[visible]
    normalized_y = normalized_y[visible]
    depth = depth[visible]
    point_sparsity = point_sparsity[visible]
    grid_width, grid_height = map(int, config.view.grid)
    pixel_x = np.minimum(
        ((normalized_x / tan_x + 1.0) * 0.5 * grid_width).astype(np.int64),
        grid_width - 1,
    )
    pixel_y = np.minimum(
        ((normalized_y / tan_y + 1.0) * 0.5 * grid_height).astype(np.int64),
        grid_height - 1,
    )
    bins = pixel_y * grid_width + pixel_x
    order = np.lexsort((depth, bins))
    sorted_bins = bins[order]
    first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    visible_values = point_sparsity[order[first]]
    bin_count = grid_width * grid_height
    coverage = len(visible_values) / bin_count
    score = (
        visible_values.sum()
        + (bin_count - len(visible_values)) * float(config.view.empty_cost)
    ) / bin_count
    return float(score), float(coverage)


def _score_segment_headings(
    segment: SafeSegment,
    heading_angles: np.ndarray,
    heading_directions: np.ndarray,
    up: np.ndarray,
    reference: ReferenceCamera,
    sample: GaussianSample,
    config,
) -> Tuple[np.ndarray, np.ndarray]:
    waypoint_count = len(segment.positions)
    heading_count = len(heading_angles)
    scores = np.empty((waypoint_count, heading_count), dtype=np.float64)
    coverages = np.empty_like(scores)
    for waypoint_index, position in enumerate(segment.positions):
        for heading_index, direction in enumerate(heading_directions):
            scores[waypoint_index, heading_index], coverages[
                waypoint_index, heading_index
            ] = _view_score(
                position, direction, up, reference, sample, config
            )
    return scores, coverages


def select_view_scenario(
    segments: list,
    sample: GaussianSample,
    reference: ReferenceCamera,
    config,
) -> ViewScenario:
    up = _normalize(reference.up)
    base = np.asarray(reference.forward, dtype=np.float64)
    base = _normalize(base - np.dot(base, up) * up)
    side = _normalize(np.cross(up, base))
    heading_angles, heading_directions = _yaw_directions(
        up, reference.forward, int(config.view.heading_count)
    )
    alphas = np.linspace(0.0, 1.0, int(config.path.num_waypoints))
    selection = str(getattr(config.view, "selection", "best"))
    selection_rng = np.random.default_rng(int(config.seed) + 2)
    selected = None
    qualified_count = 0
    for segment_index, segment in enumerate(segments):
        travel_direction = segment.positions[-1] - segment.positions[0]
        travel_direction = _normalize(
            travel_direction - np.dot(travel_direction, up) * up
        )
        minimum_forward_alignment = math.cos(
            math.radians(float(config.view.maximum_heading_from_travel_degrees))
        )
        heading_alignment = heading_directions @ travel_direction
        navigation_headings = heading_alignment >= minimum_forward_alignment
        scores, coverages = _score_segment_headings(
            segment,
            heading_angles,
            heading_directions,
            up,
            reference,
            sample,
            config,
        )
        valid = coverages >= float(config.view.minimum_coverage)
        valid_scores = np.where(valid, scores, np.inf)
        alternative_scores = valid_scores.min(axis=1)
        alternative_indices = valid_scores.argmin(axis=1)
        if not np.isfinite(alternative_scores).all():
            continue

        for start_index, start_angle in enumerate(heading_angles):
            if not navigation_headings[start_index]:
                continue
            if not valid[0, start_index]:
                continue
            if scores[0, start_index] > alternative_scores[0] + float(
                config.view.endpoint_score_margin
            ):
                continue
            for end_index, end_angle in enumerate(heading_angles):
                if not navigation_headings[end_index]:
                    continue
                if not valid[-1, end_index]:
                    continue
                if scores[-1, end_index] > alternative_scores[-1] + float(
                    config.view.endpoint_score_margin
                ):
                    continue
                angle_delta = abs((end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi)
                if math.degrees(angle_delta) > float(config.view.maximum_heading_change_degrees):
                    continue
                interpolated_angles = _interpolate_angle(start_angle, end_angle, alphas)
                step = 2.0 * np.pi / len(heading_angles)
                interpolated_indices = np.mod(
                    np.rint(interpolated_angles / step).astype(np.int64),
                    len(heading_angles),
                )
                waypoint_indices = np.arange(len(alphas))
                initial_scores = scores[waypoint_indices, interpolated_indices]
                initial_coverages = coverages[waypoint_indices, interpolated_indices]
                if np.any(initial_coverages < float(config.view.minimum_coverage)):
                    continue
                coarse_improvement = (
                    initial_scores[1:-1] - alternative_scores[1:-1]
                )
                if coarse_improvement.mean() < 0.75 * float(
                    config.view.minimum_mean_improvement
                ):
                    continue
                if coarse_improvement.max() < 0.75 * float(
                    config.view.minimum_peak_improvement
                ):
                    continue
                directions = _directions_for_angles(
                    interpolated_angles, base, side
                )
                continuous_alignment = directions @ travel_direction
                if np.any(continuous_alignment < minimum_forward_alignment):
                    continue
                continuous = [
                    _view_score(
                        position,
                        direction,
                        up,
                        reference,
                        sample,
                        config,
                    )
                    for position, direction in zip(segment.positions, directions)
                ]
                initial_scores = np.asarray(
                    [value[0] for value in continuous], dtype=np.float64
                )
                initial_coverages = np.asarray(
                    [value[1] for value in continuous], dtype=np.float64
                )
                if np.any(initial_coverages < float(config.view.minimum_coverage)):
                    continue
                improvement = initial_scores[1:-1] - alternative_scores[1:-1]
                mean_improvement = float(improvement.mean())
                peak_improvement = float(improvement.max())
                if mean_improvement < float(config.view.minimum_mean_improvement):
                    continue
                if peak_improvement < float(config.view.minimum_peak_improvement):
                    continue
                scenario_score = mean_improvement + 0.5 * peak_improvement
                candidate = ViewScenario(
                    segment=segment,
                    directions=directions,
                    score=scenario_score,
                    initial_scores=initial_scores.astype(np.float32),
                    alternative_scores=alternative_scores.astype(np.float32),
                    alternative_directions=heading_directions[
                        alternative_indices
                    ].copy(),
                    coverages=initial_coverages.astype(np.float32),
                    start_heading_degrees=math.degrees(start_angle),
                    end_heading_degrees=math.degrees(end_angle),
                    minimum_forward_alignment=float(continuous_alignment.min()),
                )
                qualified_count += 1
                if selection == "best":
                    if selected is None or candidate.score > selected.score:
                        selected = candidate
                elif selection_rng.integers(qualified_count) == 0:
                    # Reservoir sampling makes every qualified scenario equally
                    # likely without retaining all scenarios in memory.
                    selected = candidate
        print(
            f"View search: {segment_index + 1}/{len(segments)} "
            f"qualified={qualified_count} "
            f"selected={None if selected is None else round(selected.score, 4)}"
        )
    if selected is None:
        raise RuntimeError(
            "No safe segment produced the requested interior view-cost gap; "
            "relax the view improvement thresholds or increase path candidates"
        )
    print(
        f"View selection: mode={selection}, qualified={qualified_count}, "
        f"score={selected.score:.4f}"
    )
    return selected


def _poses_from_positions_directions(
    positions: np.ndarray, directions: np.ndarray, up: np.ndarray
) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    poses[:, :3, 3] = positions
    for index, direction in enumerate(directions):
        poses[index, :3, :3] = _camera_rotation(direction, up)
    return poses


def save_scenario(scenario: ViewScenario, reference: ReferenceCamera, config) -> Path:
    output_path = Path(config.output.path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    up = _normalize(reference.up).astype(np.float32)
    poses = _poses_from_positions_directions(
        scenario.segment.positions, scenario.directions, up
    )
    np.savez_compressed(
        output_path,
        cubic_trajectory=scenario.segment.positions,
        cubic_trajectory_directions=scenario.directions,
        poses_c2w=poses,
        start_position=scenario.segment.positions[0],
        goal_position=scenario.segment.positions[-1],
        start_direction=scenario.directions[0],
        goal_direction=scenario.directions[-1],
        up=up,
        waypoint_up=np.broadcast_to(up, scenario.directions.shape).copy(),
        proxy_initial_scores=scenario.initial_scores,
        proxy_alternative_scores=scenario.alternative_scores,
        proxy_alternative_directions=scenario.alternative_directions,
        proxy_coverages=scenario.coverages,
    )
    diagnostic = {
        "output": str(output_path),
        "start_position": scenario.segment.positions[0].tolist(),
        "goal_position": scenario.segment.positions[-1].tolist(),
        "start_heading_degrees": scenario.start_heading_degrees,
        "end_heading_degrees": scenario.end_heading_degrees,
        "minimum_forward_alignment": scenario.minimum_forward_alignment,
        "minimum_clearance": scenario.segment.sampled_clearance,
        "maximum_ground_support_distance": scenario.segment.maximum_ground_distance,
        "scenario_score": scenario.score,
        "proxy_initial_scores": scenario.initial_scores.tolist(),
        "proxy_alternative_scores": scenario.alternative_scores.tolist(),
        "proxy_alternative_directions": scenario.alternative_directions.tolist(),
        "proxy_improvements": (
            scenario.initial_scores - scenario.alternative_scores
        ).tolist(),
        "proxy_coverages": scenario.coverages.tolist(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    OmegaConf.save(config, output_path.with_suffix(".yaml"))
    print(f"Saved navigation trajectory: {output_path}")
    print(
        f"Proxy interior improvement: mean="
        f"{np.mean(scenario.initial_scores[1:-1] - scenario.alternative_scores[1:-1]):.4f}, "
        f"peak={np.max(scenario.initial_scores[1:-1] - scenario.alternative_scores[1:-1]):.4f}"
    )
    return output_path


def generate(config) -> Path:
    _validate_config(config)
    cameras = load_reference_cameras(
        config.input.camera_json,
        znear=float(config.view.near),
        zfar=float(config.view.far),
    )
    camera_index = int(config.input.camera_index)
    if not -len(cameras) <= camera_index < len(cameras):
        raise IndexError("input.camera_index is out of range")
    reference = cameras[camera_index]
    sample = load_gaussian_sample(config)
    segments = sample_safe_segments(sample, reference, config)
    if bool(config.collision.exact_verification):
        remaining_segments = list(segments)
        while remaining_segments:
            scenario = select_view_scenario(
                remaining_segments, sample, reference, config
            )
            verified = verify_segments_against_full_ply(
                [scenario.segment], config
            )
            if verified:
                scenario.segment = verified[0]
                break
            remaining_segments = [
                segment
                for segment in remaining_segments
                if segment is not scenario.segment
            ]
            print("Rejected top view scenario after exact collision check")
        else:
            raise RuntimeError(
                "Every forward-facing view scenario approached an obstacle in "
                "the full PLY"
            )
    else:
        scenario = select_view_scenario(segments, sample, reference, config)
    return save_scenario(scenario, reference, config)


def load_config():
    parser = argparse.ArgumentParser(
        description="Generate a safe navigation path with a poor initial view"
    )
    parser.add_argument(
        "--config", default="configs/church_trajectory_generate.yaml"
    )
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    generate(load_config())
