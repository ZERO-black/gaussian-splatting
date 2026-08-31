import json
import math
import time
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from analysis.knn import STATIC_METRICS, ply_property_name
from gaussian_renderer import GaussianModel
from scene import Scene
from trajectory.interpolation import interpolate_trajectory
from trajectory.io import resolve_camera_trajectory_path, save_trajectory
from trajectory.losses import (
    knn_loss,
    roll_alignment_loss,
    rotation_acceleration_loss,
    tangent_alignment_loss,
)
from trajectory.model import TrainableTrajectory
from trajectory.renderer import TrajectoryRenderer
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

try:
    import wandb
except ImportError:
    wandb = None


class TrajectoryTrainer:
    """Optimize a camera trajectory while keeping all Gaussians frozen."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.runtime.device)
        output_root = Path(config.output.directory).expanduser()
        run_name = getattr(config.output, "run_name", None)
        if run_name is not None:
            run_name = str(run_name)
            if (
                not run_name
                or Path(run_name).name != run_name
                or run_name in {".", ".."}
            ):
                raise ValueError("output.run_name must be a single non-empty path name")
        self.output_dir = (
            output_root / (run_name or time.strftime("%Y-%m-%d_%H-%M-%S"))
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
        if self.config.model.iteration == 0 or self.config.model.iteration < -1:
            raise ValueError("model.iteration must be positive or -1 for the latest")
        if self.config.knn.metric not in STATIC_METRICS:
            raise ValueError(
                f"knn.metric must be one of {STATIC_METRICS}, got "
                f"{self.config.knn.metric!r}"
            )
        if self.config.knn.k <= 0:
            raise ValueError("knn.k must be positive")
        if not isinstance(self.config.knn.normalize_by_camera_distance, bool):
            raise ValueError("knn.normalize_by_camera_distance must be boolean")
        if not isinstance(self.config.knn.multiply_by_splat_radius, bool):
            raise ValueError("knn.multiply_by_splat_radius must be boolean")
        if (
            self.config.knn.normalize_by_camera_distance
            and self.config.knn.multiply_by_splat_radius
        ):
            raise ValueError(
                "KNN camera-distance and splat-radius weighting are mutually exclusive"
            )
        if (
            not math.isfinite(float(self.config.knn.threshold))
            or self.config.knn.threshold < 0
        ):
            raise ValueError("knn.threshold must be a finite non-negative value")
        if self.config.trajectory.intermediate_waypoints < 0:
            raise ValueError("trajectory.intermediate_waypoints must be non-negative")
        spline_control_points = getattr(
            self.config.trajectory,
            "spline_control_points",
            None,
        )
        if (
            spline_control_points is not None
            and (
                not isinstance(spline_control_points, int)
                or isinstance(spline_control_points, bool)
                or spline_control_points < 1
            )
        ):
            raise ValueError(
                "trajectory.spline_control_points must be a positive integer or null"
            )
        if self.config.loss.roll_weight < 0:
            raise ValueError("loss.roll_weight must be non-negative")
        if self.config.loss.rotation_acceleration_weight < 0:
            raise ValueError("loss.rotation_acceleration_weight must be non-negative")
        if self.config.loss.tangent_weight < 0:
            raise ValueError("loss.tangent_weight must be non-negative")
        if (
            not math.isfinite(float(self.config.knn.background))
            or self.config.knn.background < 0
        ):
            raise ValueError("knn.background must be a finite non-negative value")
        if (
            self.config.knn.tail_threshold is not None
            and self.config.knn.tail_threshold <= 0
        ):
            raise ValueError("knn.tail_threshold must be positive when provided")

    def _prepare_output(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Trajectory output directory is not empty: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, self.output_dir / "config.yaml")
        print(f"Trajectory outputs: {self.output_dir}")

    def _setup_scene(self) -> None:
        safe_state(self.config.runtime.quiet)
        model_iteration = self.config.model.iteration
        if model_iteration == -1:
            model_iteration = searchForMaxIteration(
                str(Path(self.config.model.model_path) / "point_cloud")
            )
        self.knn_ply_path = self._resolve_knn_ply(model_iteration)

        self.gaussians = GaussianModel(self.config.model.sh_degree)
        self.scene = Scene(
            self.config.model,
            self.gaussians,
            load_iteration=model_iteration,
            shuffle=False,
            load_ply_path=str(self.knn_ply_path),
            load_camera_images=False,
        )
        self._freeze_gaussians()

        property_name = ply_property_name(self.config.knn.metric, self.config.knn.k)
        try:
            raw_knn_values = self.gaussians.get_knn_metric(property_name)
        except KeyError as exc:
            raise ValueError(
                f"KNN metric {property_name!r} is missing from {self.knn_ply_path}"
            ) from exc
        tail_threshold = self._resolve_tail_threshold(property_name)
        self.knn_values = torch.clamp(raw_knn_values / tail_threshold, 0.0, 1.0)
        self.knn_values.requires_grad_(False)
        self.config.knn.ply_path = str(self.knn_ply_path)
        self.config.knn.tail_threshold = tail_threshold
        OmegaConf.save(self.config, self.output_dir / "config.yaml")
        print(
            f"KNN objective: {property_name} from {self.knn_ply_path} "
            f"(tail normalization={tail_threshold:g}, "
            f"camera distance={self.config.knn.normalize_by_camera_distance}, "
            f"splat radius={self.config.knn.multiply_by_splat_radius}, "
            f"rendered threshold={self.config.knn.threshold:g})"
        )

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
            knn_background=torch.full(
                (3,),
                self.config.knn.background,
                dtype=torch.float32,
                device=self.device,
            ),
            knn_values=self.knn_values,
            knn_normalize_by_camera_distance=(
                self.config.knn.normalize_by_camera_distance
            ),
            knn_multiply_by_splat_radius=(
                self.config.knn.multiply_by_splat_radius
            ),
            knn_threshold=self.config.knn.threshold,
        )

    def _resolve_knn_ply(self, model_iteration: int) -> Path:
        configured_path = self.config.knn.ply_path
        if configured_path is None:
            path = (
                Path(self.config.model.model_path)
                / "knn_analysis"
                / f"iteration_{model_iteration}"
                / "point_cloud_knn.ply"
            )
        else:
            path = Path(configured_path).expanduser()
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"KNN-annotated PLY not found: {path}. Run analysis/knn.py "
                "for this model and iteration before optimizing the trajectory."
            )
        return path

    def _resolve_tail_threshold(self, property_name: str) -> float:
        configured_threshold = self.config.knn.tail_threshold
        if configured_threshold is not None:
            threshold = float(configured_threshold)
        else:
            summary_path = self.knn_ply_path.parent / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"KNN summary not found: {summary_path}. Set knn.tail_threshold "
                    "explicitly or keep summary.json beside point_cloud_knn.ply."
                )
            with summary_path.open() as stream:
                summary = json.load(stream)
            try:
                threshold = float(
                    summary["metrics"][str(self.config.knn.k)]
                    [self.config.knn.metric]["tail_threshold"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"No tail threshold for {property_name!r} in {summary_path}"
                ) from exc
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                f"Tail threshold for {property_name!r} must be finite and positive, "
                f"got {threshold}"
            )
        return threshold

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
        resolved_path = resolve_camera_trajectory_path(self.config.trajectory.path)
        self.trajectory_uses_saved_up = resolved_path.suffix.lower() == ".lookat"
        spline_control_points = getattr(
            self.config.trajectory,
            "spline_control_points",
            None,
        )
        self.trajectory_model = TrainableTrajectory.from_file(
            resolved_path,
            self.config.trajectory.key,
            self.config.trajectory.up,
            device=self.device,
            direction_key=self.config.trajectory.direction_key,
            intermediate_waypoints=self.config.trajectory.intermediate_waypoints,
            spline_control_points=spline_control_points,
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
            roll = roll_alignment_loss(
                poses[1:-1, :3, :3],
                -self.trajectory_model.initial_c2w[1:-1, :3, 1],
            )
        else:
            translation_offset = positions.new_zeros(())
            rotation_offset = positions.new_zeros(())
            roll = positions.new_zeros(())
        if len(positions) >= 3:
            acceleration = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
            acceleration_loss = acceleration.square().mean()
            tangent = tangent_alignment_loss(positions)
            rotation_acceleration = rotation_acceleration_loss(poses[:, :3, :3])
        else:
            acceleration_loss = positions.new_zeros(())
            tangent = positions.new_zeros(())
            rotation_acceleration = positions.new_zeros(())

        total_loss = (
            self.config.loss.translation_offset_weight * translation_offset
            + self.config.loss.rotation_offset_weight * rotation_offset
            + self.config.loss.roll_weight * roll
            + self.config.loss.acceleration_weight * acceleration_loss
            + self.config.loss.tangent_weight * tangent
            + self.config.loss.rotation_acceleration_weight * rotation_acceleration
        )
        return {
            "total": total_loss,
            "translation_offset": translation_offset,
            "rotation_offset": rotation_offset,
            "roll": roll,
            "acceleration": acceleration_loss,
            "tangent": tangent,
            "rotation_acceleration": rotation_acceleration,
        }

    def _backward_knn_one_camera_at_a_time(self) -> torch.Tensor:
        """Accumulate pose gradients without retaining another camera's graph."""
        zero = self.trajectory_model.initial_c2w.new_zeros(())

        # Endpoints are fixed, so rendering them cannot contribute a parameter
        # gradient. Each interior pose is rebuilt to give this camera its own graph.
        camera_indices = range(1, len(self.trajectory_model.initial_c2w) - 1)
        num_cameras = len(camera_indices)
        if num_cameras == 0:
            return zero

        knn_sum = zero
        loss_scale = self.config.loss.knn_weight / num_cameras
        for camera_index in camera_indices:
            pose_c2w = self.trajectory_model(camera_index)
            render_output = self.renderer.render_knn_pose(
                pose_c2w,
                self.reference_camera,
            )
            camera_loss = knn_loss(render_output["knn"])
            if loss_scale != 0:
                (loss_scale * camera_loss).backward()
            knn_sum = knn_sum + camera_loss.detach()

            # These references own this camera's autograd and CUDA rasterization
            # buffers. Dropping them here keeps peak memory independent of the
            # number of cameras.
            del camera_loss, render_output, pose_c2w

        return knn_sum / num_cameras

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
        rendered_knn_loss = self._backward_knn_one_camera_at_a_time()

        path_losses = self._get_path_losses()
        if path_losses["total"].requires_grad:
            path_losses["total"].backward()
        self.optimizer.step()

        photometric_loss = rendered_knn_loss.new_zeros(())
        total_loss = (
            self.config.loss.knn_weight * rendered_knn_loss
            + path_losses["total"].detach()
            + self.config.loss.photometric_weight * photometric_loss
        )
        losses = {
            "total": total_loss,
            "knn": rendered_knn_loss,
            "photometric": photometric_loss,
            "translation_offset": path_losses["translation_offset"].detach(),
            "rotation_offset": path_losses["rotation_offset"].detach(),
            "roll": path_losses["roll"].detach(),
            "acceleration": path_losses["acceleration"].detach(),
            "tangent": path_losses["tangent"].detach(),
            "rotation_acceleration": path_losses["rotation_acceleration"].detach(),
        }

        renders = (
            self._render_previews()
            if iteration % self.config.checkpoint.interval == 0
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
        artifact_dir = self.preview_dir / f"iteration_{iteration:06d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        poses_c2w = self.trajectory_model().detach().cpu().numpy()
        interpolated_c2w = interpolate_trajectory(
            poses_c2w,
            self.config.checkpoint.intermediate_frames,
            up=None if self.trajectory_uses_saved_up else self.config.trajectory.up,
        )

        self.renderer.render_knn_keyframes(
            poses_c2w,
            self.reference_camera,
            artifact_dir / "waypoint_knn",
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
            artifact_dir / "trajectory_knn.mp4",
            self.config.checkpoint.fps,
            self.config.checkpoint.codec,
            output_type="knn",
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

                if iteration % self.config.checkpoint.interval == 0:
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
