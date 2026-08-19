import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np


def load_kaolin_camera_trajectory(
    path: Union[str, Path],
    key: str,
    up: Sequence[float],
    direction_key: str = None,
) -> np.ndarray:
    """Load Kaolin camera centers from NPZ and return canonical 3DGS C2W poses.

    When a direction array is present it is used directly. Otherwise camera
    forward directions are reconstructed from the path tangent, preserving
    compatibility with position-only Kaolin exports.
    """
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"{path} has no {key!r}; available keys: {archive.files}")
        positions = np.asarray(archive[key], dtype=np.float32)
        resolved_direction_key = direction_key
        automatic_direction_key = f"{key}_directions"
        if resolved_direction_key is None and automatic_direction_key in archive:
            resolved_direction_key = automatic_direction_key
        if resolved_direction_key is not None:
            if resolved_direction_key not in archive:
                raise KeyError(
                    f"{path} has no {resolved_direction_key!r}; "
                    f"available keys: {archive.files}"
                )
            forwards = np.asarray(
                archive[resolved_direction_key],
                dtype=np.float32,
            )
        else:
            forwards = None

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{key!r} must have shape [N, 3], got {positions.shape}")
    if len(positions) < 2:
        raise ValueError("camera trajectory must contain at least two positions")
    if not np.isfinite(positions).all():
        raise ValueError("camera trajectory contains non-finite positions")

    up = np.asarray(up, dtype=np.float32)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("trajectory.up must be a finite non-zero 3D vector")

    if forwards is None:
        forwards = _path_tangents(positions)
    else:
        if forwards.shape != positions.shape:
            raise ValueError(
                f"{resolved_direction_key!r} must have shape {positions.shape}, "
                f"got {forwards.shape}"
            )
        if not np.isfinite(forwards).all():
            raise ValueError("camera directions contain non-finite values")
        forwards = _normalize(forwards, "camera directions must be non-zero")
    forwards = _directions_with_fixed_up(forwards, up)
    kaolin_c2w = _kaolin_look_at_c2w(positions, forwards, up)

    # Kaolin uses OpenGL-style -Z forward and +Y up. 3DGS/COLMAP uses +Z
    # forward and +Y down, so flip the camera Y and Z axes.
    kaolin_to_3dgs = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return kaolin_c2w @ kaolin_to_3dgs


def save_trajectory(
    path: Union[str, Path],
    poses_c2w: np.ndarray,
    source_path: Union[str, Path],
    source_key: str,
) -> None:
    """Save canonical 3DGS C2W poses with explicit source metadata."""
    poses_c2w = np.asarray(poses_c2w, dtype=np.float32)
    _validate_c2w(poses_c2w)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cubic_trajectory=poses_c2w[:, :3, 3],
        poses_c2w=poses_c2w,
    )
    metadata = {
        "pose_convention": "3dgs_c2w",
        "source_format": "kaolin_camera_positions_npz",
        "source_path": str(source_path),
        "source_key": source_key,
        "keys": {
            "cubic_trajectory": list(poses_c2w[:, :3, 3].shape),
            "poses_c2w": list(poses_c2w.shape),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def _path_tangents(positions: np.ndarray) -> np.ndarray:
    tangents = np.empty_like(positions)
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]
    tangents[1:-1] = positions[2:] - positions[:-2]
    return _normalize(tangents, "Consecutive camera positions must not coincide")


def _directions_with_fixed_up(directions: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Project directions to the plane normal to up, fixing camera up exactly."""
    up = up / np.linalg.norm(up)
    directions = directions - (directions @ up)[:, None] * up[None]
    parallel = np.linalg.norm(directions, axis=1) < 1e-6
    if parallel.any():
        fallback_axis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(up))]
        fallback = fallback_axis - np.dot(fallback_axis, up) * up
        directions[parallel] = fallback
    return _normalize(directions, "Could not construct fixed-up directions")


def _kaolin_look_at_c2w(positions, forwards, up):
    up = up / np.linalg.norm(up)
    right = np.cross(forwards, np.broadcast_to(up, forwards.shape))

    parallel = np.linalg.norm(right, axis=1) < 1e-6
    if parallel.any():
        fallback = np.eye(3, dtype=np.float32)[
            np.argmin(np.abs(forwards[parallel]), axis=1)
        ]
        right[parallel] = np.cross(forwards[parallel], fallback)

    right = _normalize(right, "Could not construct camera right vectors")
    corrected_up = _normalize(
        np.cross(-forwards, right),
        "Could not construct camera up vectors",
    )

    poses = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    poses[:, :3, 0] = right
    poses[:, :3, 1] = corrected_up
    poses[:, :3, 2] = -forwards
    poses[:, :3, 3] = positions
    return poses


def _normalize(vectors: np.ndarray, error: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError(error)
    return vectors / norms


def _validate_c2w(poses: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"trajectory must have shape [N, 4, 4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("trajectory contains non-finite values")
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype)
    if not np.allclose(poses[:, 3, :], expected, atol=1e-5):
        raise ValueError("trajectory contains invalid homogeneous transforms")
