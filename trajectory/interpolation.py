import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_pair_linear(
    start_c2w: np.ndarray,
    end_c2w: np.ndarray,
    intermediate_cameras: int,
    up=None,
) -> np.ndarray:
    """Place cameras uniformly between two C2W poses.

    Camera centers are linearly interpolated. Rotations use spherical linear
    interpolation so every generated camera keeps a valid rotation matrix.
    The returned array includes both the start and end cameras and therefore
    has ``intermediate_cameras + 2`` poses.
    """
    if intermediate_cameras < 0:
        raise ValueError("intermediate_cameras must be non-negative")

    start_c2w = np.asarray(start_c2w, dtype=np.float64)
    end_c2w = np.asarray(end_c2w, dtype=np.float64)
    for name, pose in (("start_c2w", start_c2w), ("end_c2w", end_c2w)):
        if pose.shape != (4, 4):
            raise ValueError(f"{name} must have shape [4, 4], got {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError(f"{name} contains non-finite values")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"{name} must be a homogeneous C2W transform")

    alphas = np.linspace(
        0.0,
        1.0,
        intermediate_cameras + 2,
        dtype=np.float64,
    )
    rotation_slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack((start_c2w[:3, :3], end_c2w[:3, :3]))
        ),
    )
    rotations = rotation_slerp(alphas).as_matrix()
    if up is not None:
        rotations = _rotations_with_fixed_up(rotations, up)

    translations = (
        (1.0 - alphas[:, None]) * start_c2w[:3, 3]
        + alphas[:, None] * end_c2w[:3, 3]
    )
    poses = np.tile(
        np.eye(4, dtype=np.float32),
        (intermediate_cameras + 2, 1, 1),
    )
    poses[:, :3, :3] = rotations.astype(np.float32)
    poses[:, :3, 3] = translations.astype(np.float32)

    # Avoid tiny endpoint drift introduced by the rotation conversion.
    poses[0] = start_c2w.astype(np.float32)
    poses[-1] = end_c2w.astype(np.float32)
    return poses


def interpolate_trajectory(
    poses_c2w: np.ndarray,
    intermediate_frames: int,
    up=None,
) -> np.ndarray:
    """Insert smooth C2W poses between each pair of key cameras."""
    if intermediate_frames < 0:
        raise ValueError("intermediate_frames must be non-negative")

    output = []
    for start, end in zip(poses_c2w[:-1], poses_c2w[1:]):
        segment = interpolate_camera_pair_linear(
            start,
            end,
            intermediate_frames,
            up=up,
        )
        output.append(segment[:-1])

    output.append(poses_c2w[-1:].astype(np.float32))
    return np.concatenate(output, axis=0)


def _rotations_with_fixed_up(rotations: np.ndarray, up) -> np.ndarray:
    """Remove roll from interpolated 3DGS C2W rotations."""
    up = np.asarray(up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("up must be a finite 3D vector")
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        raise ValueError("up must be non-zero")
    up = up / up_norm

    forward = rotations[:, :, 2]
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)
    forward = forward - (forward @ up)[:, None] * up[None]
    parallel_forward = np.linalg.norm(forward, axis=1) < 1e-6
    if parallel_forward.any():
        fallback_axis = np.eye(3)[np.argmin(np.abs(up))]
        fallback_forward = fallback_axis - np.dot(fallback_axis, up) * up
        forward[parallel_forward] = fallback_forward
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(forward, np.broadcast_to(up, forward.shape))
    parallel = np.linalg.norm(right, axis=1) < 1e-6
    if parallel.any():
        fallback = np.eye(3)[np.argmin(np.abs(forward[parallel]), axis=1)]
        right[parallel] = np.cross(forward[parallel], fallback)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(forward, right)
    return np.stack((right, down, forward), axis=-1)
