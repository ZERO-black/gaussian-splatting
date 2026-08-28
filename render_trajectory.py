import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gaussian_renderer import GaussianModel
from scene import Scene
from trajectory.interpolation import interpolate_trajectory
from trajectory.io import (
    load_camera_trajectory,
    resolve_camera_trajectory_path,
    save_trajectory,
)
from trajectory.renderer import TrajectoryRenderer
from utils.general_utils import safe_state


def render_trajectory(config) -> None:
    if config.uncertainty.enabled:
        if config.uncertainty.iteration == 0 or config.uncertainty.iteration < -1:
            raise ValueError(
                "uncertainty.iteration must be positive or -1 for the latest checkpoint"
            )
        if len(config.uncertainty.background) != 3:
            raise ValueError("uncertainty.background must contain three values")

    safe_state(True)
    gaussians = GaussianModel(config.model.sh_degree)
    scene = Scene(
        config.model,
        gaussians,
        load_iteration=config.model.iteration,
        uncertainty_iteration=(
            config.uncertainty.iteration
            if config.uncertainty.enabled
            else None
        ),
        shuffle=False,
        load_camera_images=False,
    )

    if config.trajectory.camera_split == "train":
        reference_cameras = scene.getTrainCameras()
    elif config.trajectory.camera_split == "test":
        reference_cameras = scene.getTestCameras()
    else:
        raise ValueError("trajectory.camera_split must be 'train' or 'test'")

    trajectory_path = resolve_camera_trajectory_path(config.trajectory.path)
    poses_c2w = load_camera_trajectory(
        trajectory_path,
        config.trajectory.key,
        config.trajectory.up,
        direction_key=config.trajectory.direction_key,
    )
    reference_camera = reference_cameras[config.trajectory.reference_camera_index]

    interpolated_c2w = interpolate_trajectory(
        poses_c2w,
        config.trajectory.intermediate_frames,
        up=(
            None
            if trajectory_path.suffix.lower() == ".lookat"
            else config.trajectory.up
        ),
    )
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    save_trajectory(
        output_dir / "keyframes.npz",
        poses_c2w,
        trajectory_path,
        config.trajectory.key,
    )
    save_trajectory(
        output_dir / "interpolated.npz",
        interpolated_c2w,
        trajectory_path,
        config.trajectory.key,
    )

    background_value = 1.0 if config.model.white_background else 0.0
    background = torch.full((3,), background_value, dtype=torch.float32, device="cuda")
    uncertainty_background = None
    if config.uncertainty.enabled:
        uncertainty_background = torch.tensor(
            config.uncertainty.background,
            dtype=torch.float32,
            device="cuda",
        )
    renderer = TrajectoryRenderer(
        gaussians,
        config.pipeline,
        background,
        uncertainty_background=uncertainty_background,
    )
    renderer.render_keyframes(
        poses_c2w,
        reference_camera,
        output_dir / "keyframes",
    )
    renderer.render_video(
        interpolated_c2w,
        reference_camera,
        output_dir / config.output.video_name,
        config.output.fps,
        config.output.codec,
        config.output.save_video_frames,
    )


def load_config():
    parser = argparse.ArgumentParser(description="Render a camera trajectory")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    default_path = Path(__file__).parent / "configs" / "trajectory_render.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(default_path),
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    render_trajectory(load_config())
