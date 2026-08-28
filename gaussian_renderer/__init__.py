#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import (
    CameraGaussianRasterizer,
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    use_separate_sh = separate_sh and override_color is None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if use_separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if use_separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def render_uncertainty(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    uncertainty_sh=None,
):
    """Render the frozen scalar uncertainty SH channel as a one-channel map."""
    if pc.get_change_feature.numel() == 0 or pc.uncertainty_sh_degree is None:
        raise RuntimeError(
            "No uncertainty feature is loaded; use load_ply_uncertainty() first"
        )

    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        device="cuda",
    )

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.uncertainty_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    rasterizer = CameraGaussianRasterizer(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if uncertainty_sh is None:
        uncertainty_sh = pc.get_change_feature.repeat(1, 1, 3)
    rendered_uncertainty, radii, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=uncertainty_sh,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "uncertainty": rendered_uncertainty[:1],
        "uncertainty_viewspace_points": screenspace_points,
        "uncertainty_visibility_filter": radii > 0,
    }


def render_knn(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    knn_values: torch.Tensor,
    scaling_modifier=1.0,
    normalize_by_camera_distance=False,
):
    """Render a frozen scalar KNN cost while retaining camera-pose gradients."""
    expected_count = pc.get_xyz.shape[0]
    if (
        knn_values.ndim != 2
        or knn_values.shape[0] != expected_count
        or knn_values.shape[1] not in (1, 3)
    ):
        raise ValueError(
            "knn_values must have shape [num_gaussians, 1 or 3], got "
            f"{tuple(knn_values.shape)}"
        )
    if not torch.isfinite(knn_values).all():
        raise ValueError("knn_values contains non-finite values")

    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        device="cuda",
    )
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    rasterizer_class = (
        GaussianRasterizer
        if normalize_by_camera_distance
        else CameraGaussianRasterizer
    )
    rasterizer = rasterizer_class(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    knn_colors = (
        knn_values.repeat(1, 3) if knn_values.shape[1] == 1 else knn_values
    )
    if normalize_by_camera_distance:
        knn_colors = normalize_knn_by_camera_distance(
            knn_colors,
            pc.get_xyz,
            viewpoint_camera.camera_center,
        )
    rendered_knn, radii, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=knn_colors,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "knn": rendered_knn[:1],
        "knn_radii": radii,
        "knn_viewspace_points": screenspace_points,
        "knn_visibility_filter": radii > 0,
    }


def multiply_knn_by_splat_radius(
    knn_values: torch.Tensor,
    splat_radii: torch.Tensor,
) -> torch.Tensor:
    """Multiply each Gaussian's KNN value by its CUDA-computed pixel radius."""
    if knn_values.ndim != 2:
        raise ValueError("knn_values must have shape [N, C]")
    if splat_radii.ndim != 1 or splat_radii.shape[0] != knn_values.shape[0]:
        raise ValueError("splat_radii must have shape [N]")
    if not torch.isfinite(knn_values).all():
        raise ValueError("knn_values contains non-finite values")
    if (splat_radii < 0).any():
        raise ValueError("splat_radii must be non-negative")
    return knn_values * splat_radii.to(
        dtype=knn_values.dtype,
        device=knn_values.device,
    ).unsqueeze(-1)


def render_knn_times_splat_radius(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    knn_values: torch.Tensor,
    scaling_modifier=1.0,
):
    """Render KNN multiplied by the projected radius from CUDA preprocessing."""
    # forward.cu writes the projected integer pixel radius to radii[idx]. The
    # radius is returned by the rasterizer but is not available until that pass
    # finishes, so obtain it once and render the radius-weighted values again.
    # Rendering ones over black also gives an alpha/coverage map at no extra pass.
    with torch.no_grad():
        radius_probe = render_knn(
            viewpoint_camera,
            pc,
            pipe,
            torch.zeros_like(bg_color),
            torch.ones_like(knn_values),
            scaling_modifier=scaling_modifier,
            normalize_by_camera_distance=False,
        )
        splat_radii = radius_probe["knn_radii"]
        coverage = radius_probe["knn"]
        radius_weighted_knn = multiply_knn_by_splat_radius(
            knn_values,
            splat_radii,
        )
        del radius_probe

    output = render_knn(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        radius_weighted_knn,
        scaling_modifier=scaling_modifier,
        normalize_by_camera_distance=False,
    )
    output["knn_splat_radii"] = splat_radii
    output["knn_coverage"] = coverage
    return output


def normalize_knn_by_camera_distance(
    knn_values: torch.Tensor,
    gaussian_xyz: torch.Tensor,
    camera_center: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Divide Gaussian KNN values by differentiable camera-center distance."""
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must have shape [N, 3]")
    if knn_values.ndim != 2 or knn_values.shape[0] != gaussian_xyz.shape[0]:
        raise ValueError("knn_values must have shape [N, C]")
    if camera_center.shape != (3,):
        raise ValueError("camera_center must have shape [3]")
    if eps <= 0:
        raise ValueError("eps must be positive")

    camera_distance = torch.linalg.vector_norm(
        gaussian_xyz - camera_center.unsqueeze(0),
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    return knn_values / camera_distance
