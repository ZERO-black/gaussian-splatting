import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from trajectory.interpolation import (
    interpolate_camera_pair_linear,
    interpolate_trajectory,
)


class CameraPairInterpolationTest(unittest.TestCase):
    def test_places_camera_centers_linearly_and_keeps_endpoints(self):
        start = np.eye(4, dtype=np.float32)
        end = np.eye(4, dtype=np.float32)
        end[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        end[:3, 3] = [3.0, 6.0, 9.0]

        poses = interpolate_camera_pair_linear(start, end, 2)

        self.assertEqual(poses.shape, (4, 4, 4))
        np.testing.assert_array_equal(poses[0], start)
        np.testing.assert_array_equal(poses[-1], end)
        np.testing.assert_allclose(
            poses[:, :3, 3],
            [[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]],
            atol=1e-6,
        )

        for rotation in poses[:, :3, :3]:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)

    def test_zero_intermediate_cameras_returns_only_endpoints(self):
        start = np.eye(4, dtype=np.float32)
        end = np.eye(4, dtype=np.float32)
        end[0, 3] = 1.0

        poses = interpolate_camera_pair_linear(start, end, 0)

        np.testing.assert_array_equal(poses, np.stack((start, end)))

    def test_rejects_negative_intermediate_camera_count(self):
        pose = np.eye(4, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            interpolate_camera_pair_linear(pose, pose, -1)

    def test_video_interpolation_inserts_frames_between_every_waypoint(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        poses[:, 0, 3] = np.arange(4)

        frames = interpolate_trajectory(poses, intermediate_frames=2)

        # Three segments, each contributing start + 2 intermediate frames,
        # followed by the final endpoint.
        self.assertEqual(frames.shape, (10, 4, 4))
        np.testing.assert_allclose(frames[:, 0, 3], np.arange(10) / 3, atol=1e-6)

    def test_cubic_position_path_passes_through_every_waypoint(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        poses[:, :2, 3] = [[0, 0], [1, 0], [1, 1], [2, 1]]
        intermediate_frames = 5

        frames = interpolate_trajectory(poses, intermediate_frames)

        waypoint_stride = intermediate_frames + 1
        np.testing.assert_array_equal(
            frames[::waypoint_stride, :3, 3],
            poses[:, :3, 3],
        )

    def test_cubic_position_path_removes_linear_corner_at_waypoint(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        poses[:, :2, 3] = [[0, 0], [1, 0], [1, 1]]
        intermediate_frames = 20

        frames = interpolate_trajectory(poses, intermediate_frames)
        waypoint_index = intermediate_frames + 1
        incoming = (
            frames[waypoint_index, :3, 3]
            - frames[waypoint_index - 1, :3, 3]
        )
        outgoing = (
            frames[waypoint_index + 1, :3, 3]
            - frames[waypoint_index, :3, 3]
        )
        cosine = np.dot(incoming, outgoing) / (
            np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        )
        turn_degrees = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        self.assertLess(turn_degrees, 10.0)

    def test_rotation_spline_smooths_angular_velocity_at_waypoint(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler(
            "y", [0.0, 10.0, 80.0], degrees=True
        ).as_matrix()
        intermediate_frames = 19

        frames = interpolate_trajectory(poses, intermediate_frames)
        relative = np.einsum(
            "nij,njk->nik",
            np.transpose(frames[:-1, :3, :3], (0, 2, 1)),
            frames[1:, :3, :3],
        )
        angular_steps = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        waypoint_boundary = intermediate_frames

        # Pairwise SLERP would jump directly from 0.5 to 3.5 degrees/frame.
        self.assertLess(
            abs(
                angular_steps[waypoint_boundary + 1]
                - angular_steps[waypoint_boundary]
            ),
            0.5,
        )

    def test_rotation_spline_preserves_key_rotations(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler(
            "xyz",
            [[0, 0, 0], [10, 20, 5], [-5, 40, 15], [20, 60, 0]],
            degrees=True,
        ).as_matrix()
        intermediate_frames = 4

        frames = interpolate_trajectory(poses, intermediate_frames)

        np.testing.assert_array_equal(
            frames[::intermediate_frames + 1],
            poses,
        )

    def test_fixed_up_removes_roll_without_resetting_pitch_at_keyframes(self):
        up = np.array([0.0, 0.0, 1.0])
        pitches = np.deg2rad([0.0, -30.0, 0.0])
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        for index, pitch in enumerate(pitches):
            forward = np.array([0.0, np.cos(pitch), np.sin(pitch)])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            poses[index, :3, :3] = np.stack((right, down, forward), axis=1)

        frames = interpolate_trajectory(poses, intermediate_frames=10, up=up)
        forward_pitch = np.rad2deg(np.arcsin(frames[:, 2, 2]))

        self.assertAlmostEqual(float(forward_pitch[11]), -30.0, places=4)
        self.assertLess(float(np.max(np.abs(np.diff(forward_pitch)))), 5.0)
        np.testing.assert_allclose(
            frames[:, :3, 2] / np.linalg.norm(frames[:, :3, 2], axis=1)[:, None],
            frames[:, :3, 2],
            atol=1e-6,
        )

    def test_fixed_up_does_not_restore_one_frame_roll_spikes(self):
        up = np.array([0.0, 0.0, 1.0])
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        for index, (yaw_degrees, roll_degrees) in enumerate(
            [(0.0, 0.0), (10.0, 35.0), (20.0, 0.0)]
        ):
            yaw = np.deg2rad(yaw_degrees)
            pitch = np.deg2rad(-10.0)
            forward = np.array(
                [
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]
            )
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            base = np.stack((right, down, forward), axis=1)
            local_roll = Rotation.from_rotvec(
                [0.0, 0.0, np.deg2rad(roll_degrees)]
            ).as_matrix()
            poses[index, :3, :3] = base @ local_roll

        frames = interpolate_trajectory(poses, intermediate_frames=10, up=up)
        relative = np.einsum(
            "nij,njk->nik",
            np.transpose(frames[:-1, :3, :3], (0, 2, 1)),
            frames[1:, :3, :3],
        )
        angular_steps = np.rad2deg(Rotation.from_matrix(relative).magnitude())

        self.assertLess(float(angular_steps.max()), 5.0)
        np.testing.assert_array_equal(frames[::11, :3, 3], poses[:, :3, 3])


if __name__ == "__main__":
    unittest.main()
