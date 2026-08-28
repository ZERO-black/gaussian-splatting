import torch


def uncertainty_loss(uncertainty_map: torch.Tensor) -> torch.Tensor:
    """Minimize the rendered predictive photometric uncertainty.

    The uncertainty checkpoint was fitted against a per-pixel RGB/SSIM error
    map. Its rendered scalar channel therefore already is the visual cost; the
    trajectory objective is its spatial mean.
    """
    if uncertainty_map.ndim != 3 or uncertainty_map.shape[0] != 1:
        raise ValueError(
            "uncertainty_map must have shape [1, height, width], got "
            f"{tuple(uncertainty_map.shape)}"
        )
    if uncertainty_map.numel() == 0:
        raise ValueError("uncertainty_map must not be empty")
    return uncertainty_map.mean()


def apply_knn_threshold(
    knn_map: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Flatten the rendered KNN objective below an absolute pixel threshold."""
    if knn_map.ndim != 3 or knn_map.shape[0] != 1:
        raise ValueError(
            "knn_map must have shape [1, height, width], got "
            f"{tuple(knn_map.shape)}"
        )
    if knn_map.numel() == 0:
        raise ValueError("knn_map must not be empty")
    if not torch.isfinite(knn_map).all():
        raise ValueError("knn_map contains non-finite values")
    threshold = torch.as_tensor(
        threshold,
        dtype=knn_map.dtype,
        device=knn_map.device,
    )
    if threshold.numel() != 1 or not torch.isfinite(threshold):
        raise ValueError("KNN threshold must be a finite scalar")
    if threshold < 0:
        raise ValueError("KNN threshold must be non-negative")
    return torch.clamp_min(knn_map, threshold)


def knn_loss(knn_map: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Minimize thresholded rendered KNN/camera-distance values."""
    return apply_knn_threshold(knn_map, threshold).mean()


def roll_alignment_loss(
    rotations: torch.Tensor,
    reference_up: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize roll while leaving the camera forward direction unconstrained."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [N, 3, 3]")
    if reference_up.shape != rotations.shape[:-2] + (3,):
        raise ValueError("reference_up must have shape [N, 3]")
    if not torch.isfinite(rotations).all() or not torch.isfinite(reference_up).all():
        raise ValueError("roll alignment inputs must be finite")
    if rotations.shape[0] == 0:
        return rotations.new_zeros(())

    forward = torch.nn.functional.normalize(rotations[..., :, 2], dim=-1)
    current_up = torch.nn.functional.normalize(-rotations[..., :, 1], dim=-1)
    reference_up = torch.nn.functional.normalize(reference_up, dim=-1)
    projected_up = reference_up - (
        reference_up * forward
    ).sum(dim=-1, keepdim=True) * forward
    projected_norm = torch.linalg.vector_norm(projected_up, dim=-1, keepdim=True)
    target_up = projected_up / projected_norm.clamp_min(eps)
    cosine = (current_up * target_up).sum(dim=-1).clamp(-1.0, 1.0)

    # Roll is undefined when the view direction is parallel to the reference up.
    valid = projected_norm.squeeze(-1) >= eps
    penalties = torch.where(valid, 1.0 - cosine, torch.zeros_like(cosine))
    return penalties.mean()


def rotation_acceleration_loss(rotations: torch.Tensor) -> torch.Tensor:
    """Penalize changes in relative rotation between adjacent path segments."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [N, 3, 3]")
    if not torch.isfinite(rotations).all():
        raise ValueError("rotations must be finite")
    if rotations.shape[0] < 3:
        return rotations.new_zeros(())

    previous_step = rotations[:-2].transpose(-1, -2) @ rotations[1:-1]
    next_step = rotations[1:-1].transpose(-1, -2) @ rotations[2:]
    return (next_step - previous_step).square().mean()


def tangent_alignment_loss(
    positions: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize angular changes between normalized adjacent path segments."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if positions.shape[0] < 3:
        return positions.new_zeros(())

    segments = positions[1:] - positions[:-1]
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    directions = segments / lengths.clamp_min(eps).unsqueeze(-1)
    cosine = (directions[:-1] * directions[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    valid = (lengths[:-1] >= eps) & (lengths[1:] >= eps)
    penalties = torch.where(valid, 1.0 - cosine, torch.zeros_like(cosine))
    return penalties.sum() / valid.sum().clamp_min(1)
