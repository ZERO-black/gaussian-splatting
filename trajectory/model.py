from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn

from .interpolation import interpolate_camera_pair_linear
from .io import load_camera_trajectory, resolve_camera_trajectory_path


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3] axis-angle vectors to [..., 3, 3] rotation matrices."""
    x, y, z = axis_angle.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        dim=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))

    angle = torch.linalg.vector_norm(axis_angle, dim=-1)
    # torch.sinc(x) = sin(pi*x)/(pi*x), including the stable value at zero.
    sin_over_angle = torch.sinc(angle / torch.pi)
    one_minus_cos_over_angle_sq = 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()

    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(axis_angle.shape[:-1] + (3, 3))
    return (
        identity
        + sin_over_angle[..., None, None] * skew
        + one_minus_cos_over_angle_sq[..., None, None] * (skew @ skew)
    )


def _spline_delta_basis(
    num_waypoints: int,
    num_control_points: int,
) -> np.ndarray:
    """Map smooth interior B-spline controls to interior waypoint deltas."""
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be positive")
    if num_control_points < 1:
        raise ValueError("num_control_points must be positive")

    # Two additional, fixed zero controls anchor both endpoint deltas. The
    # trainable controls therefore describe broad changes to the whole path
    # instead of allowing every waypoint to move independently.
    total_controls = num_control_points + 2
    degree = min(3, total_controls - 1)
    internal_knots = total_controls - degree - 1
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, internal_knots + 2)[1:-1],
            np.ones(degree + 1),
        )
    )
    control_basis = np.eye(total_controls, dtype=np.float64)
    sample_times = np.linspace(0.0, 1.0, num_waypoints + 2)[1:-1]
    basis = BSpline(knots, control_basis, degree)(sample_times)
    return basis[:, 1:-1].astype(np.float32)


class TrainableTrajectory(nn.Module):
    def __init__(
        self,
        initial_c2w: Union[np.ndarray, torch.Tensor],
        spline_control_points: Optional[int] = None,
    ):
        super().__init__()
        initial_c2w = torch.as_tensor(initial_c2w, dtype=torch.float32)
        self._validate(initial_c2w)

        self.register_buffer("initial_c2w", initial_c2w.clone())
        num_interior = initial_c2w.shape[0] - 2
        if spline_control_points is not None and (
            isinstance(spline_control_points, bool)
            or not isinstance(spline_control_points, int)
            or spline_control_points < 1
        ):
            raise ValueError(
                "spline_control_points must be a positive integer when provided"
            )

        num_controls = (
            min(spline_control_points, num_interior)
            if spline_control_points is not None and num_interior > 0
            else num_interior
        )
        if spline_control_points is None or num_interior == 0:
            delta_basis = torch.eye(
                num_interior,
                dtype=initial_c2w.dtype,
                device=initial_c2w.device,
            )
        else:
            delta_basis = initial_c2w.new_tensor(
                _spline_delta_basis(num_interior, num_controls)
            )
        self.register_buffer("delta_basis", delta_basis)
        self.translation_delta = nn.Parameter(initial_c2w.new_zeros(num_controls, 3))
        self.rotation_delta = nn.Parameter(initial_c2w.new_zeros(num_controls, 3))

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        key: str,
        up,
        device: Optional[Union[str, torch.device]] = None,
        direction_key: Optional[str] = None,
        intermediate_waypoints: int = 0,
        spline_control_points: Optional[int] = None,
    ) -> "TrainableTrajectory":
        resolved_path = resolve_camera_trajectory_path(path)
        initial_c2w = load_camera_trajectory(
            resolved_path,
            key,
            up,
            direction_key=direction_key,
        )
        if intermediate_waypoints < 0:
            raise ValueError("intermediate_waypoints must be non-negative")
        if initial_c2w.shape[0] == 2 and intermediate_waypoints > 0:
            initial_c2w = interpolate_camera_pair_linear(
                initial_c2w[0],
                initial_c2w[1],
                intermediate_waypoints,
                up=None,
            )
        trajectory = cls(initial_c2w, spline_control_points=spline_control_points)
        return trajectory.to(device) if device is not None else trajectory

    @classmethod
    def from_npz(
        cls,
        path: Union[str, Path],
        key: str,
        up,
        device: Optional[Union[str, torch.device]] = None,
        direction_key: Optional[str] = None,
        intermediate_waypoints: int = 0,
        spline_control_points: Optional[int] = None,
    ) -> "TrainableTrajectory":
        """Backward-compatible alias for NPZ and SIBR trajectory inputs."""
        return cls.from_file(
            path,
            key,
            up,
            device,
            direction_key,
            intermediate_waypoints,
            spline_control_points,
        )

    @staticmethod
    def _validate(initial_c2w: torch.Tensor) -> None:
        if initial_c2w.ndim != 3 or initial_c2w.shape[-2:] != (4, 4):
            raise ValueError("initial_c2w must have shape [N, 4, 4]")
        if initial_c2w.shape[0] < 2:
            raise ValueError("A trajectory must contain at least two cameras")
        if not torch.isfinite(initial_c2w).all():
            raise ValueError("initial_c2w contains non-finite values")

        expected_last_row = initial_c2w.new_tensor([0.0, 0.0, 0.0, 1.0])
        if not torch.allclose(initial_c2w[:, 3], expected_last_row.expand_as(initial_c2w[:, 3])):
            raise ValueError("initial_c2w must contain homogeneous 4x4 transforms")

    def all_poses(self) -> torch.Tensor:
        initial_interior = self.initial_c2w[1:-1]
        translation_delta = self.delta_basis @ self.translation_delta
        rotation_delta = self.delta_basis @ self.rotation_delta
        proposed_rotation = (
            initial_interior[:, :3, :3]
            @ _axis_angle_to_matrix(rotation_delta)
        )
        corrected_center = initial_interior[:, :3, 3] + translation_delta

        bottom_row = initial_interior[:, 3:4, :]
        interior_c2w = torch.cat(
            (
                torch.cat((proposed_rotation, corrected_center[..., None]), dim=-1),
                bottom_row,
            ),
            dim=-2,
        )

        return torch.cat(
            (self.initial_c2w[:1], interior_c2w, self.initial_c2w[-1:]),
            dim=0,
        )

    def forward(self, index=None) -> torch.Tensor:
        if index is None:
            return self.all_poses()

        num_poses = self.initial_c2w.shape[0]
        index = int(index)
        if index < 0:
            index += num_poses
        if index < 0 or index >= num_poses:
            raise IndexError(f"trajectory index {index} is out of range")
        if index == 0 or index == num_poses - 1:
            return self.initial_c2w[index]

        delta_index = index - 1
        initial_pose = self.initial_c2w[index]
        translation_delta = self.delta_basis[delta_index] @ self.translation_delta
        rotation_delta = self.delta_basis[delta_index] @ self.rotation_delta
        proposed_rotation = (
            initial_pose[:3, :3]
            @ _axis_angle_to_matrix(rotation_delta)
        )
        corrected_center = (
            initial_pose[:3, 3] + translation_delta
        )
        return torch.cat(
            (
                torch.cat(
                    (proposed_rotation, corrected_center[:, None]), dim=-1
                ),
                initial_pose[3:4],
            ),
            dim=0,
        )
