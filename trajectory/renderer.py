from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render, render_knn, render_uncertainty
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


class TrajectoryRenderer:
    """Render canonical C2W poses with a frozen Gaussian model."""

    def __init__(
        self,
        gaussians,
        pipeline,
        background: torch.Tensor,
        uncertainty_background: torch.Tensor = None,
        knn_background: torch.Tensor = None,
        knn_values: torch.Tensor = None,
    ):
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.uncertainty_background = uncertainty_background
        self.knn_background = knn_background
        self.knn_values = (
            knn_values.repeat(1, 3) if knn_values is not None else None
        )
        self.uncertainty_sh = (
            gaussians.get_change_feature.repeat(1, 1, 3)
            if uncertainty_background is not None
            else None
        )

    @torch.no_grad()
    def render_keyframes(self, poses_c2w, reference_camera, output_dir) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, pose in enumerate(tqdm(poses_c2w, desc="Keyframes")):
            image = self.render_rgb_pose(pose, reference_camera)["render"]
            torchvision.utils.save_image(image, output_dir / f"{index:05d}.png")

    @torch.no_grad()
    def render_uncertainty_keyframes(
        self,
        poses_c2w,
        reference_camera,
        output_dir,
    ) -> None:
        """Save the scalar uncertainty map at every trajectory waypoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, pose in enumerate(tqdm(poses_c2w, desc="Waypoint uncertainty")):
            uncertainty = self.render_uncertainty_pose(
                pose,
                reference_camera,
            )["uncertainty"]
            uncertainty_rgb = _uncertainty_to_rgb_uint8(uncertainty)
            cv2.imwrite(
                str(output_dir / f"{index:05d}.png"),
                cv2.cvtColor(uncertainty_rgb, cv2.COLOR_RGB2BGR),
            )

    @torch.no_grad()
    def render_knn_keyframes(self, poses_c2w, reference_camera, output_dir) -> None:
        """Save the normalized KNN cost map at every trajectory waypoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, pose in enumerate(tqdm(poses_c2w, desc="Waypoint KNN cost")):
            knn_map = self.render_knn_pose(pose, reference_camera)["knn"]
            knn_rgb = _scalar_to_rgb_uint8(knn_map, "knn")
            cv2.imwrite(
                str(output_dir / f"{index:05d}.png"),
                cv2.cvtColor(knn_rgb, cv2.COLOR_RGB2BGR),
            )

    @torch.no_grad()
    def render_video(
        self,
        poses_c2w: np.ndarray,
        reference_camera,
        output_path,
        fps: int,
        codec: str,
        save_frames: bool = False,
        output_type: str = "rgb",
    ) -> None:
        if len(codec) != 4:
            raise ValueError("OpenCV video codec must contain exactly four characters")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        size = (
            int(reference_camera.image_width),
            int(reference_camera.image_height),
        )
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        if output_type not in {"rgb", "uncertainty", "knn"}:
            raise ValueError("output_type must be 'rgb', 'uncertainty', or 'knn'")

        frame_dir = output_path.parent / f"{output_path.stem}_frames"
        if save_frames:
            frame_dir.mkdir(parents=True, exist_ok=True)

        try:
            description = {
                "rgb": "RGB video",
                "uncertainty": "Uncertainty video",
                "knn": "KNN cost video",
            }[output_type]
            for index, pose in enumerate(tqdm(poses_c2w, desc=description)):
                if output_type == "rgb":
                    image = self.render_rgb_pose(pose, reference_camera)["render"]
                    frame_rgb = (
                        image.detach().clamp(0, 1).mul(255).byte()
                        .permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                elif output_type == "uncertainty":
                    uncertainty = self.render_uncertainty_pose(
                        pose,
                        reference_camera,
                    )["uncertainty"]
                    frame_rgb = _uncertainty_to_rgb_uint8(uncertainty)
                else:
                    knn_map = self.render_knn_pose(pose, reference_camera)["knn"]
                    frame_rgb = _scalar_to_rgb_uint8(knn_map, "knn")
                if frame_rgb.shape[1::-1] != size:
                    raise ValueError("All reference cameras must have the same resolution")
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if save_frames:
                    cv2.imwrite(
                        str(frame_dir / f"{index:05d}.png"),
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    )
        finally:
            writer.release()

    def render_pose(self, pose_c2w, reference):
        outputs = self.render_rgb_pose(pose_c2w, reference)
        if self.uncertainty_background is not None:
            outputs.update(self.render_uncertainty_pose(pose_c2w, reference))
        if self.knn_background is not None:
            outputs.update(self.render_knn_pose(pose_c2w, reference))
        return outputs

    def render_rgb_pose(self, pose_c2w, reference):
        camera = _make_camera(pose_c2w, reference)
        return render(
            camera,
            self.gaussians,
            self.pipeline,
            self.background,
            use_trained_exp=False,
        )

    def render_uncertainty_pose(self, pose_c2w, reference):
        if self.uncertainty_background is None:
            raise RuntimeError("Uncertainty rendering is not configured")
        camera = _make_camera(pose_c2w, reference)
        return render_uncertainty(
            camera,
            self.gaussians,
            self.pipeline,
            self.uncertainty_background,
            uncertainty_sh=self.uncertainty_sh,
        )

    def render_knn_pose(self, pose_c2w, reference):
        if self.knn_background is None or self.knn_values is None:
            raise RuntimeError("KNN rendering is not configured")
        camera = _make_camera(pose_c2w, reference)
        return render_knn(
            camera,
            self.gaussians,
            self.pipeline,
            self.knn_background,
            self.knn_values,
        )


def _uncertainty_to_rgb_uint8(uncertainty: torch.Tensor) -> np.ndarray:
    """Map scalar uncertainty in [0, 1] to a fixed TURBO RGB visualization."""
    return _scalar_to_rgb_uint8(uncertainty, "uncertainty")


def _scalar_to_rgb_uint8(values: torch.Tensor, name: str) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(
            f"{name} must have shape [1, height, width], got {tuple(values.shape)}"
        )
    scalar = (
        values.detach().clamp(0, 1).mul(255).byte()
        .squeeze(0).contiguous().cpu().numpy()
    )
    bgr = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _make_camera(pose_c2w, reference) -> MiniCam:
    device = torch.device("cuda")
    c2w = torch.as_tensor(pose_c2w, dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w)
    world_view = w2c.transpose(0, 1).contiguous()
    projection = getProjectionMatrix(
        znear=reference.znear,
        zfar=reference.zfar,
        fovX=reference.FoVx,
        fovY=reference.FoVy,
    ).transpose(0, 1).to(device)
    full_projection = world_view @ projection
    return MiniCam(
        reference.image_width,
        reference.image_height,
        reference.FoVy,
        reference.FoVx,
        reference.znear,
        reference.zfar,
        world_view,
        full_projection,
    )
