from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .io import load_kaolin_camera_trajectory


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


def _rotation_with_fixed_up(
    rotation: torch.Tensor,
    world_up: torch.Tensor,
) -> torch.Tensor:
    """Construct a rotation whose camera-up is exactly ``world_up``."""
    proposed_forward = torch.nn.functional.normalize(
        rotation[..., :, 2],
        dim=-1,
    )
    up = world_up.expand_as(proposed_forward)
    forward = proposed_forward - (
        proposed_forward * up
    ).sum(dim=-1, keepdim=True) * up

    # A proposed direction parallel to up has no valid fixed-up camera pose.
    # Choose a deterministic direction in the plane as a rare-case fallback.
    parallel = torch.linalg.vector_norm(forward, dim=-1, keepdim=True) < 1e-6
    axes = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    fallback_axis = axes[world_up.abs().argmin()].expand_as(forward)
    fallback_forward = fallback_axis - (
        fallback_axis * up
    ).sum(dim=-1, keepdim=True) * up
    forward = torch.where(parallel, fallback_forward, forward)
    forward = torch.nn.functional.normalize(forward, dim=-1)

    right = torch.linalg.cross(forward, up, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)

    # 3DGS camera coordinates use +Y down and +Z forward.
    down = torch.linalg.cross(forward, right, dim=-1)
    return torch.stack((right, down, forward), dim=-1)


class TrainableTrajectory(nn.Module):
    def __init__(
        self,
        initial_c2w: Union[np.ndarray, torch.Tensor],
        world_up: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        initial_c2w = torch.as_tensor(initial_c2w, dtype=torch.float32)
        self._validate(initial_c2w)

        if world_up is None:
            world_up = -initial_c2w[0, :3, 1]
        world_up = torch.as_tensor(
            world_up,
            dtype=initial_c2w.dtype,
            device=initial_c2w.device,
        )
        if world_up.shape != (3,) or not torch.isfinite(world_up).all():
            raise ValueError("world_up must be a finite 3D vector")
        if torch.linalg.vector_norm(world_up) < 1e-8:
            raise ValueError("world_up must be non-zero")

        self.register_buffer("initial_c2w", initial_c2w.clone())
        self.register_buffer(
            "world_up",
            world_up / torch.linalg.vector_norm(world_up),
            persistent=False,
        )
        num_interior = initial_c2w.shape[0] - 2
        self.translation_delta = nn.Parameter(initial_c2w.new_zeros(num_interior, 3))
        self.rotation_delta = nn.Parameter(initial_c2w.new_zeros(num_interior, 3))

    @classmethod
    def from_npz(
        cls,
        path: Union[str, Path],
        key: str,
        up,
        device: Optional[Union[str, torch.device]] = None,
        direction_key: Optional[str] = None,
    ) -> "TrainableTrajectory":
        trajectory = cls(
            load_kaolin_camera_trajectory(
                path,
                key,
                up,
                direction_key=direction_key,
            ),
            world_up=up,
        )
        return trajectory.to(device) if device is not None else trajectory

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
        proposed_rotation = (
            initial_interior[:, :3, :3]
            @ _axis_angle_to_matrix(self.rotation_delta)
        )
        corrected_rotation = _rotation_with_fixed_up(
            proposed_rotation,
            self.world_up,
        )
        corrected_center = initial_interior[:, :3, 3] + self.translation_delta

        bottom_row = initial_interior[:, 3:4, :]
        interior_c2w = torch.cat(
            (
                torch.cat((corrected_rotation, corrected_center[..., None]), dim=-1),
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
        proposed_rotation = (
            initial_pose[:3, :3]
            @ _axis_angle_to_matrix(self.rotation_delta[delta_index])
        )
        corrected_rotation = _rotation_with_fixed_up(
            proposed_rotation,
            self.world_up,
        )
        corrected_center = (
            initial_pose[:3, 3] + self.translation_delta[delta_index]
        )
        return torch.cat(
            (
                torch.cat(
                    (corrected_rotation, corrected_center[:, None]), dim=-1
                ),
                initial_pose[3:4],
            ),
            dim=0,
        )
