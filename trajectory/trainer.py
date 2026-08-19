import time
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from gaussian_renderer import GaussianModel
from scene import Scene
from trajectory.interpolation import interpolate_trajectory
from trajectory.io import save_trajectory
from trajectory.losses import uncertainty_loss
from trajectory.model import TrainableTrajectory
from trajectory.renderer import TrajectoryRenderer
from utils.general_utils import safe_state

try:
    import wandb
except ImportError:
    wandb = None


class TrajectoryTrainer:
    """Optimize a camera trajectory while keeping all Gaussians frozen."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.output_dir = (
            Path(config.output.directory).expanduser()
            / time.strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "previews"
        self.iteration = 0
        self.ema_loss = None

        self._validate_config()
        self._prepare_output()
        self._setup_scene()
        self._setup_trajectory()
        self._setup_optimizer()
        self._setup_logger()

        if config.checkpoint.resume:
            self.load_checkpoint(config.checkpoint.resume)

    def _validate_config(self) -> None:
        if self.config.optimization.iterations < 1:
            raise ValueError("optimization.iterations must be positive")
        if self.config.checkpoint.interval < 1:
            raise ValueError("checkpoint.interval must be positive")
        if self.config.checkpoint.intermediate_frames < 0:
            raise ValueError("checkpoint.intermediate_frames must be non-negative")
        if self.config.checkpoint.fps < 1:
            raise ValueError("checkpoint.fps must be positive")
        if len(self.config.checkpoint.codec) != 4:
            raise ValueError("checkpoint.codec must contain exactly four characters")
        if self.config.logging.preview_interval < 1:
            raise ValueError("logging.preview_interval must be positive")
        if self.config.uncertainty.enabled:
            if self.config.uncertainty.iteration == 0 or self.config.uncertainty.iteration < -1:
                raise ValueError(
                    "uncertainty.iteration must be positive or -1 for the latest checkpoint"
                )
            if len(self.config.uncertainty.background) != 3:
                raise ValueError("uncertainty.background must contain three values")

    def _prepare_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, self.output_dir / "config.yaml")
        print(f"Trajectory outputs: {self.output_dir}")

    def _setup_scene(self) -> None:
        safe_state(self.config.runtime.quiet)
        self.gaussians = GaussianModel(self.config.model.sh_degree)
        self.scene = Scene(
            self.config.model,
            self.gaussians,
            load_iteration=self.config.model.iteration,
            uncertainty_iteration=(
                self.config.uncertainty.iteration
                if self.config.uncertainty.enabled
                else None
            ),
            shuffle=False,
        )
        self._freeze_gaussians()

        if self.config.trajectory.camera_split == "train":
            cameras = self.scene.getTrainCameras()
        elif self.config.trajectory.camera_split == "test":
            cameras = self.scene.getTestCameras()
        else:
            raise ValueError("trajectory.camera_split must be 'train' or 'test'")
        if not cameras:
            raise ValueError("The selected reference camera split is empty")

        reference_index = self.config.trajectory.reference_camera_index
        if not -len(cameras) <= reference_index < len(cameras):
            raise IndexError(
                f"reference_camera_index {reference_index} is out of range "
                f"for {len(cameras)} cameras"
            )
        self.reference_camera = cameras[reference_index]

        background_value = 1.0 if self.config.model.white_background else 0.0
        self.background = torch.full(
            (3,), background_value, dtype=torch.float32, device=self.device
        )
        self.renderer = TrajectoryRenderer(
            self.gaussians,
            self.config.pipeline,
            self.background,
            uncertainty_background=(
                torch.tensor(
                    self.config.uncertainty.background,
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.config.uncertainty.enabled
                else None
            ),
        )

    def _freeze_gaussians(self) -> None:
        parameter_names = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "_change_feature",
            "_exposure",
        )
        for name in parameter_names:
            value = getattr(self.gaussians, name, None)
            if isinstance(value, torch.Tensor):
                value.requires_grad_(False)

    def _setup_trajectory(self) -> None:
        self.trajectory_model = TrainableTrajectory.from_npz(
            self.config.trajectory.path,
            self.config.trajectory.key,
            self.config.trajectory.up,
            device=self.device,
            direction_key=self.config.trajectory.direction_key,
        )

    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [self.trajectory_model.translation_delta],
                    "lr": self.config.optimization.translation_lr,
                    "name": "translation",
                },
                {
                    "params": [self.trajectory_model.rotation_delta],
                    "lr": self.config.optimization.rotation_lr,
                    "name": "rotation",
                },
            ]
        )

    def _setup_logger(self) -> None:
        self.wandb_run = None
        if not self.config.logging.wandb.enabled:
            return
        if wandb is None:
            raise ImportError(
                "W&B logging is enabled but wandb is not installed. "
                "Install it with `pip install wandb`."
            )

        self.wandb_run = wandb.init(
            project=self.config.logging.wandb.project,
            entity=self.config.logging.wandb.entity,
            name=self.config.logging.wandb.name,
            mode=self.config.logging.wandb.mode,
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _get_path_losses(self):
        poses = self.trajectory_model()
        positions = poses[:, :3, 3]

        if self.trajectory_model.translation_delta.numel() > 0:
            translation_offset = self.trajectory_model.translation_delta.square().mean()
            rotation_offset = self.trajectory_model.rotation_delta.square().mean()
        else:
            translation_offset = positions.new_zeros(())
            rotation_offset = positions.new_zeros(())
        if len(positions) >= 3:
            acceleration = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
            acceleration_loss = acceleration.square().mean()
        else:
            acceleration_loss = positions.new_zeros(())

        total_loss = (
            self.config.loss.translation_offset_weight * translation_offset
            + self.config.loss.rotation_offset_weight * rotation_offset
            + self.config.loss.acceleration_weight * acceleration_loss
        )
        return {
            "total": total_loss,
            "translation_offset": translation_offset,
            "rotation_offset": rotation_offset,
            "acceleration": acceleration_loss,
        }

    def _backward_uncertainty_one_camera_at_a_time(self) -> torch.Tensor:
        """Accumulate pose gradients without retaining another camera's graph."""
        zero = self.trajectory_model.initial_c2w.new_zeros(())
        if not self.config.uncertainty.enabled:
            return zero

        # Endpoints are fixed, so rendering them cannot contribute a parameter
        # gradient. Each interior pose is rebuilt to give this camera its own graph.
        camera_indices = range(1, len(self.trajectory_model.initial_c2w) - 1)
        num_cameras = len(camera_indices)
        if num_cameras == 0:
            return zero

        uncertainty_sum = zero
        loss_scale = self.config.loss.uncertainty_weight / num_cameras
        for camera_index in camera_indices:
            pose_c2w = self.trajectory_model(camera_index)
            render_output = self.renderer.render_uncertainty_pose(
                pose_c2w,
                self.reference_camera,
            )
            camera_loss = uncertainty_loss(render_output["uncertainty"])
            if loss_scale != 0:
                (loss_scale * camera_loss).backward()
            uncertainty_sum = uncertainty_sum + camera_loss.detach()

            # These references own this camera's autograd and CUDA rasterization
            # buffers. Dropping them here keeps peak memory independent of the
            # number of cameras.
            del camera_loss, render_output, pose_c2w

        return uncertainty_sum / num_cameras

    @torch.no_grad()
    def _render_previews(self):
        poses_c2w = self.trajectory_model()
        return [
            self.renderer.render_rgb_pose(pose_c2w, self.reference_camera)["render"]
            for pose_c2w in poses_c2w
        ]

    def run_train_iter(self, iteration: int):
        self.iteration = iteration
        start_time = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        uncertainty_loss = self._backward_uncertainty_one_camera_at_a_time()

        path_losses = self._get_path_losses()
        if path_losses["total"].requires_grad:
            path_losses["total"].backward()
        self.optimizer.step()

        photometric_loss = uncertainty_loss.new_zeros(())
        total_loss = (
            self.config.loss.uncertainty_weight * uncertainty_loss
            + path_losses["total"].detach()
            + self.config.loss.photometric_weight * photometric_loss
        )
        losses = {
            "total": total_loss,
            "uncertainty": uncertainty_loss,
            "photometric": photometric_loss,
            "translation_offset": path_losses["translation_offset"].detach(),
            "rotation_offset": path_losses["rotation_offset"].detach(),
            "acceleration": path_losses["acceleration"].detach(),
        }

        renders = (
            self._render_previews()
            if iteration % self.config.logging.preview_interval == 0
            else None
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        loss_value = float(losses["total"].detach())
        self.ema_loss = (
            loss_value
            if self.ema_loss is None
            else 0.4 * loss_value + 0.6 * self.ema_loss
        )
        self._log_iteration(losses, elapsed_ms)
        return {"renders": renders}, losses

    def _log_iteration(self, losses, elapsed_ms: float) -> None:
        if self.wandb_run is None:
            return
        metrics = {
            f"loss/{name}": float(value.detach())
            for name, value in losses.items()
        }
        metrics.update(
            {
                "time/iteration_ms": elapsed_ms,
                "trajectory/translation_delta_norm": float(
                    self.trajectory_model.translation_delta.detach().norm()
                ),
                "trajectory/rotation_delta_norm": float(
                    self.trajectory_model.rotation_delta.detach().norm()
                ),
            }
        )
        self.wandb_run.log(metrics, step=self.iteration)

    def save_previews(self, images, iteration: int) -> None:
        iteration_dir = self.preview_dir / f"iteration_{iteration:06d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for camera_index, image in enumerate(images):
            torchvision.utils.save_image(
                image.detach().clamp(0, 1),
                iteration_dir / f"camera_{camera_index:05d}.png",
            )

    def save_snapshot(self, iteration: int) -> None:
        poses = self.trajectory_model().detach().cpu().numpy()
        save_trajectory(
            self.output_dir / f"trajectory_{iteration:06d}.npz",
            poses,
            self.config.trajectory.path,
            self.config.trajectory.key,
        )

    def save_checkpoint(self, iteration: int) -> Path:
        state = {
            "iteration": iteration,
            "trajectory": self.trajectory_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        path = self.checkpoint_dir / f"iteration_{iteration:06d}.pth"
        torch.save(state, path)
        torch.save(state, self.checkpoint_dir / "latest.pth")
        print(f"Saved checkpoint: {path}")
        return path

    @torch.no_grad()
    def save_iteration_previews(self, iteration: int) -> None:
        """Save comparable visual artifacts for an optimization iteration."""
        if not self.config.uncertainty.enabled:
            raise RuntimeError(
                "Checkpoint uncertainty renders require uncertainty.enabled=true"
            )

        artifact_dir = self.preview_dir / f"iteration_{iteration:06d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        poses_c2w = self.trajectory_model().detach().cpu().numpy()
        interpolated_c2w = interpolate_trajectory(
            poses_c2w,
            self.config.checkpoint.intermediate_frames,
            up=self.config.trajectory.up,
        )

        self.renderer.render_uncertainty_keyframes(
            poses_c2w,
            self.reference_camera,
            artifact_dir / "waypoint_uncertainty",
        )
        self.renderer.render_video(
            interpolated_c2w,
            self.reference_camera,
            artifact_dir / "trajectory_rgb.mp4",
            self.config.checkpoint.fps,
            self.config.checkpoint.codec,
            output_type="rgb",
        )
        self.renderer.render_video(
            interpolated_c2w,
            self.reference_camera,
            artifact_dir / "trajectory_uncertainty.mp4",
            self.config.checkpoint.fps,
            self.config.checkpoint.codec,
            output_type="uncertainty",
        )
        print(f"Saved iteration previews: {artifact_dir}")

    def save_iteration_bundle(self, iteration: int) -> None:
        """Persist state, trajectory, and previews for one iteration."""
        self.save_checkpoint(iteration)
        self.save_snapshot(iteration)
        self.save_iteration_previews(iteration)

    def load_checkpoint(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.trajectory_model.load_state_dict(checkpoint["trajectory"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iteration = int(checkpoint["iteration"])
        print(f"Resumed trajectory checkpoint at iteration {self.iteration}: {path}")

    def train(self) -> None:
        start_iteration = self.iteration + 1
        final_iteration = self.config.optimization.iterations
        progress = tqdm(
            range(start_iteration, final_iteration + 1),
            desc="Trajectory training",
        )

        try:
            if self.iteration == 0:
                self.save_snapshot(0)
                self.save_iteration_previews(0)
            for iteration in progress:
                forward_output, _ = self.run_train_iter(iteration)
                progress.set_postfix(loss=f"{self.ema_loss:.6f}")

                if iteration % self.config.logging.preview_interval == 0:
                    self.save_previews(forward_output["renders"], iteration)
                del forward_output
                if (
                    iteration % self.config.checkpoint.interval == 0
                    and iteration != final_iteration
                ):
                    self.save_iteration_bundle(iteration)

            # Normal completion always persists the final state exactly once,
            # independent of checkpoint interval and resume position.
            self.save_iteration_bundle(final_iteration)
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()
