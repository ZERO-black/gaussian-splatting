import argparse
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gaussian_renderer import GaussianModel
from scene import Scene
from trajectory.interpolation import (
    interpolate_camera_pair_linear,
    interpolate_trajectory,
)
from trajectory.io import save_sibr_lookat_trajectory
from trajectory.renderer import TrajectoryRenderer
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2


def _camera_c2w(camera) -> np.ndarray:
    """Recover canonical C2W while preserving the source camera orientation."""
    return np.linalg.inv(
        getWorld2View2(camera.R, camera.T)
    ).astype(
        np.float32,
        copy=False,
    )


def _next_case_index(output_dir: Path) -> int:
    indices = [
        int(path.stem)
        for path in output_dir.iterdir()
        if path.is_file() and path.stem.isdigit()
    ]
    return max(indices, default=0) + 1


def generate_testcases(config, count: int, seed: int, output_dir: Path) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    if config.trajectory.intermediate_waypoints < 1:
        raise ValueError("trajectory.intermediate_waypoints must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_state(True)

    gaussians = GaussianModel(config.model.sh_degree)
    scene = Scene(
        config.model,
        gaussians,
        load_iteration=config.model.iteration,
        shuffle=False,
        load_camera_images=False,
    )
    if config.trajectory.camera_split == "train":
        cameras = scene.getTrainCameras()
    elif config.trajectory.camera_split == "test":
        cameras = scene.getTestCameras()
    else:
        raise ValueError("trajectory.camera_split must be 'train' or 'test'")
    if len(cameras) < 2:
        raise ValueError("The selected camera split must contain at least two cameras")

    background_value = 1.0 if config.model.white_background else 0.0
    background = torch.full(
        (3,),
        background_value,
        dtype=torch.float32,
        device="cuda",
    )
    renderer = TrajectoryRenderer(gaussians, config.pipeline, background)
    rng = np.random.default_rng(seed)
    case_index = _next_case_index(output_dir)

    for offset in range(count):
        start_index, end_index = rng.choice(len(cameras), size=2, replace=False)
        start_camera = cameras[int(start_index)]
        end_camera = cameras[int(end_index)]
        start_pose = _camera_c2w(start_camera)
        end_pose = _camera_c2w(end_camera)
        poses_c2w = interpolate_camera_pair_linear(
            start_pose,
            end_pose,
            config.trajectory.intermediate_waypoints,
            up=None,
        )
        video_poses_c2w = interpolate_trajectory(
            poses_c2w,
            config.checkpoint.intermediate_frames,
            up=None,
        )

        name = f"{case_index + offset:04d}"
        camera_path = output_dir / f"{name}.lookat"
        video_path = output_dir / f"{name}.mp4"
        save_sibr_lookat_trajectory(
            camera_path,
            poses_c2w,
            None,
            fovy_degrees=math.degrees(start_camera.FoVy),
            znear=start_camera.znear,
            zfar=start_camera.zfar,
        )
        renderer.render_video(
            video_poses_c2w,
            start_camera,
            video_path,
            config.checkpoint.fps,
            config.checkpoint.codec,
            output_type="rgb",
        )
        print(
            f"Saved testcase {name}: cameras {int(start_index)} -> "
            f"{int(end_index)}, {len(poses_c2w)} optimizer waypoints, "
            f"{len(video_poses_c2w)} video frames "
            f"({camera_path}, {video_path})"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate trajectory testcases from random source cameras"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "trajectory_train.yaml",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/trajectory_testcases"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_testcases(
        OmegaConf.load(args.config),
        args.count,
        args.seed,
        args.output.expanduser().resolve(),
    )
