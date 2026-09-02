import json
import re
from pathlib import Path
from typing import Sequence, Union

import numpy as np


_SIBR_PARAMETER = re.compile(r"(?:^|\s)-D\s+([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def resolve_camera_trajectory_path(path: Union[str, Path]) -> Path:
    """Resolve an NPZ trajectory or SIBR key-camera .lookat export."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    raise FileNotFoundError(f"Camera trajectory not found: {path}")


def load_camera_trajectory(
    path: Union[str, Path],
    key: str,
    up: Sequence[float],
    direction_key: str = None,
) -> np.ndarray:
    """Load NPZ or SIBR .lookat cameras as canonical 3DGS C2W poses."""
    resolved_path = resolve_camera_trajectory_path(path)
    suffix = resolved_path.suffix.lower()
    if suffix == ".lookat":
        return load_sibr_lookat_trajectory(resolved_path)
    if suffix == ".npz":
        return load_kaolin_camera_trajectory(
            resolved_path,
            key,
            up,
            direction_key=direction_key,
        )
    raise ValueError(
        f"Unsupported camera trajectory format {suffix!r}: {resolved_path}; "
        "expected a SIBR key-camera .lookat file or a legacy .npz file"
    )


def load_sibr_lookat_trajectory(path: Union[str, Path]) -> np.ndarray:
    """Load SIBR origin/target/up records as canonical 3DGS C2W poses.

    SIBR cameras use OpenGL local axes (+X right, +Y up, -Z viewing
    direction). Canonical 3DGS camera axes are +X right, +Y down, +Z forward.
    """
    path = Path(path)
    poses = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parameters = dict(_SIBR_PARAMETER.findall(line))
        missing = {"origin", "target", "up"} - parameters.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number} is missing SIBR parameters: {sorted(missing)}"
            )
        origin = _parse_sibr_vector(parameters["origin"], path, line_number, "origin")
        target = _parse_sibr_vector(parameters["target"], path, line_number, "target")
        camera_up = _parse_sibr_vector(parameters["up"], path, line_number, "up")
        poses.append(_sibr_lookat_c2w(origin, target, camera_up, path, line_number))

    if len(poses) < 2:
        raise ValueError(f"SIBR trajectory must contain at least two cameras: {path}")
    poses = np.stack(poses).astype(np.float32, copy=False)
    _validate_c2w(poses)
    return poses


def save_sibr_lookat_trajectory(
    path: Union[str, Path],
    poses_c2w: np.ndarray,
    up,
    fovy_degrees: float,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> None:
    """Save canonical 3DGS C2W poses as a SIBR key-camera ``.lookat`` file."""
    poses_c2w = np.asarray(poses_c2w, dtype=np.float32)
    _validate_c2w(poses_c2w)
    if up is None:
        up_vectors = -poses_c2w[:, :3, 1]
    else:
        up_vectors = np.asarray(up, dtype=np.float32)
        if up_vectors.shape == (3,):
            up_vectors = np.broadcast_to(up_vectors, (len(poses_c2w), 3)).copy()
        if up_vectors.shape != (len(poses_c2w), 3):
            raise ValueError("up must have shape [3] or [N, 3]")
    if not np.isfinite(up_vectors).all():
        raise ValueError("up vectors must be finite")
    up_norms = np.linalg.norm(up_vectors, axis=1, keepdims=True)
    if np.any(up_norms < 1e-8):
        raise ValueError("up vectors must be non-zero")
    up_vectors = up_vectors / up_norms
    if not np.isfinite(fovy_degrees) or fovy_degrees <= 0:
        raise ValueError("fovy_degrees must be finite and positive")
    if not np.isfinite([znear, zfar]).all() or znear <= 0 or zfar <= znear:
        raise ValueError("clip planes must satisfy 0 < znear < zfar")

    lines = []
    for pose, camera_up in zip(poses_c2w, up_vectors):
        origin = pose[:3, 3]
        target = origin + pose[:3, 2]
        lines.append(
            " -D origin={} -D target={} -D up={} -D fovy={:.6f} "
            "-D clip={:.6f},{:.6f}".format(
                _format_sibr_vector(origin),
                _format_sibr_vector(target),
                _format_sibr_vector(camera_up),
                fovy_degrees,
                znear,
                zfar,
            )
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _format_sibr_vector(vector: np.ndarray) -> str:
    return ",".join(f"{float(value):.9f}" for value in vector)


def _parse_sibr_vector(value, path: Path, line_number: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray([float(part) for part in value.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number} has an invalid {name} vector: {value!r}"
        ) from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(
            f"{path}:{line_number} has an invalid {name} vector: {value!r}"
        )
    return vector


def _sibr_lookat_c2w(origin, target, camera_up, path: Path, line_number: int):
    forward = target - origin
    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(camera_up)
    if forward_norm < 1e-8:
        raise ValueError(f"{path}:{line_number} has coincident origin and target")
    if up_norm < 1e-8:
        raise ValueError(f"{path}:{line_number} has a zero up vector")
    forward = forward / forward_norm
    camera_up = camera_up / up_norm
    right = np.cross(forward, camera_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError(
            f"{path}:{line_number} has parallel viewing and up directions"
        )
    right = right / right_norm
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = origin
    return pose


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
        if positions.ndim == 3 and positions.shape[1:] == (4, 4):
            poses_c2w = positions.copy()
            _validate_c2w(poses_c2w)
            return poses_c2w
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
    try:
        resolved_source = resolve_camera_trajectory_path(source_path)
    except FileNotFoundError:
        resolved_source = Path(source_path)
    source_format = (
        "sibr_lookat"
        if resolved_source.suffix.lower() == ".lookat"
        else "kaolin_camera_positions_npz"
    )
    metadata = {
        "pose_convention": "3dgs_c2w",
        "source_format": source_format,
        "source_path": str(resolved_source),
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
