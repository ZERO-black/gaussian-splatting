import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trajectory.io import (
    load_camera_trajectory,
    load_sibr_lookat_trajectory,
    save_sibr_lookat_trajectory,
)
from trajectory.model import TrainableTrajectory


SIBR_LOOKAT = """\
Cam0 -D origin=1,2,3 -D target=1,2,2 -D up=0,1,0 -D fovy=1 -D clip=0.01,1000
Cam1 -D origin=2,3,4 -D target=3,3,4 -D up=0,0.8,0.6 -D fovy=1 -D clip=0.01,1000
Cam2 -D origin=3,4,5 -D target=3,4,6 -D up=0,1,0 -D fovy=1 -D clip=0.01,1000
"""


class SibrTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_directory.name) / "camera.lookat"
        self.path.write_text(SIBR_LOOKAT)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_loads_sibr_axes_as_3dgs_c2w(self):
        poses = load_sibr_lookat_trajectory(self.path)

        self.assertEqual(poses.shape, (3, 4, 4))
        np.testing.assert_allclose(
            poses[:, :3, 3],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        )
        np.testing.assert_allclose(poses[0, :3, 0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(poses[0, :3, 1], [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(poses[0, :3, 2], [0, 0, -1], atol=1e-6)
        for rotation in poses[:, :3, :3]:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)

    def test_dispatches_sibr_key_camera_lookat(self):
        poses = load_camera_trajectory(
            self.path,
            key="unused",
            up=[0, -1, 0],
        )
        self.assertEqual(poses.shape, (3, 4, 4))

    def test_rejects_sibr_recorder_path(self):
        recorder_path = self.path.with_suffix(".path")
        recorder_path.write_bytes(b"native SIBR stream")
        with self.assertRaisesRegex(ValueError, "key-camera .lookat"):
            load_camera_trajectory(
                recorder_path,
                key="unused",
                up=[0, -1, 0],
            )

    def test_optimizer_preserves_saved_pose_at_zero_delta(self):
        poses = load_sibr_lookat_trajectory(self.path)
        model = TrainableTrajectory.from_file(
            self.path,
            key="unused",
            up=[0, -1, 0],
        )

        np.testing.assert_allclose(model().detach().numpy(), poses, atol=1e-6)
        loss = model(1)[:3, 3].sum() + model(1)[:3, 2].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(model.translation_delta.grad).all())
        self.assertTrue(torch.isfinite(model.rotation_delta.grad).all())

    def test_optimizer_adds_trainable_waypoints_for_two_camera_input(self):
        two_camera_path = Path(self.temp_directory.name) / "two_cameras.lookat"
        two_camera_path.write_text("\n".join(SIBR_LOOKAT.splitlines()[:2]) + "\n")

        model = TrainableTrajectory.from_file(
            two_camera_path,
            key="unused",
            up=[0, -1, 0],
            intermediate_waypoints=3,
        )

        self.assertEqual(model.initial_c2w.shape, (5, 4, 4))
        self.assertEqual(model.translation_delta.shape, (3, 3))
        self.assertEqual(model.rotation_delta.shape, (3, 3))
        np.testing.assert_allclose(
            model.initial_c2w[:, :3, 3].numpy(),
            [
                [1.0, 2.0, 3.0],
                [1.25, 2.25, 3.25],
                [1.5, 2.5, 3.5],
                [1.75, 2.75, 3.75],
                [2.0, 3.0, 4.0],
            ],
            atol=1e-6,
        )

    def test_sibr_lookat_writer_round_trip_with_fixed_up(self):
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        poses[:, :3, 1] = -up
        poses[:, :3, 2] = [0.0, 0.0, -1.0]
        poses[:, :3, 0] = [1.0, 0.0, 0.0]
        poses[1, :3, 3] = [1.0, 2.0, 3.0]
        output_path = Path(self.temp_directory.name) / "written.lookat"

        save_sibr_lookat_trajectory(output_path, poses, up, 45.0)

        loaded = load_sibr_lookat_trajectory(output_path)
        np.testing.assert_allclose(loaded, poses, atol=1e-6)

    def test_sibr_lookat_writer_preserves_per_pose_camera_up(self):
        poses = load_sibr_lookat_trajectory(self.path)
        output_path = Path(self.temp_directory.name) / "per_pose_up.lookat"

        save_sibr_lookat_trajectory(output_path, poses, None, 45.0)

        loaded = load_sibr_lookat_trajectory(output_path)
        np.testing.assert_allclose(loaded, poses, atol=1e-6)
        np.testing.assert_allclose(
            -loaded[:, :3, 1],
            -poses[:, :3, 1],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
