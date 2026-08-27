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
