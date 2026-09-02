import unittest

import numpy as np

from generate_navigation_trajectory import (
    _camera_rotation,
    _interpolate_angle,
    _poses_from_positions_directions,
    _yaw_directions,
    _validate_config,
)
from generate_navigation_trajectories import _endpoint_distance, _output_path
from generate_navigation_trajectory import SafeSegment, ViewScenario
from omegaconf import OmegaConf
from pathlib import Path


class NavigationTrajectoryGeometryTest(unittest.TestCase):
    def test_yaw_headings_do_not_depend_on_scene_center(self):
        angles, directions = _yaw_directions(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            4,
        )

        np.testing.assert_allclose(angles, [0, np.pi / 2, np.pi, 3 * np.pi / 2])
        np.testing.assert_allclose(directions[0], [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(directions[1], [-1, 0, 0], atol=1e-6)

    def test_angle_interpolation_uses_short_arc(self):
        interpolated = _interpolate_angle(
            np.deg2rad(350), np.deg2rad(10), np.array([0.0, 0.5, 1.0])
        )
        unwrapped_degrees = np.rad2deg(interpolated)
        np.testing.assert_allclose(unwrapped_degrees, [350, 360, 370], atol=1e-6)

    def test_saved_poses_use_graphdeco_right_down_forward_axes(self):
        positions = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)
        directions = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)

        poses = _poses_from_positions_directions(positions, directions, up)

        np.testing.assert_allclose(
            poses[:, :3, 0], np.tile([1, 0, 0], (2, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            poses[:, :3, 1], np.tile([0, 0, -1], (2, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            poses[:, :3, 2], np.tile([0, 1, 0], (2, 1)), atol=1e-6
        )
        for pose in poses:
            np.testing.assert_allclose(
                pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6
            )

    def test_camera_rotation_rejects_vertical_forward(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            _camera_rotation(np.array([0, 0, 1]), np.array([0, 0, 1]))

    def test_endpoint_distance_treats_reversed_path_as_duplicate(self):
        def scenario(start, end):
            positions = np.asarray([start, end], dtype=np.float32)
            segment = SafeSegment(positions, 1.0, 0.0)
            return ViewScenario(
                segment=segment,
                directions=np.zeros((2, 3), dtype=np.float32),
                score=0.0,
                initial_scores=np.zeros(2),
                alternative_scores=np.zeros(2),
                alternative_directions=np.zeros((2, 3)),
                coverages=np.ones(2),
                start_heading_degrees=0.0,
                end_heading_degrees=0.0,
                minimum_forward_alignment=1.0,
            )

        first = scenario([0, 0, 1], [10, 0, 1])
        reversed_nearby = scenario([10.2, 0, 1], [0.1, 0, 1])

        self.assertAlmostEqual(
            _endpoint_distance(first, reversed_nearby), 0.2, places=5
        )

    def test_batch_output_path_formats_index(self):
        path = _output_path(
            __import__("pathlib").Path("/tmp/paths"),
            "navigation_{index:04d}.npz",
            7,
        )
        self.assertEqual(path.name, "navigation_0007.npz")

    def test_rejects_unknown_view_selection_mode(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "church_trajectory_generate_100.yaml"
        )
        config = OmegaConf.load(config_path)
        config.view.selection = "maximum_then_random"

        with self.assertRaisesRegex(ValueError, "view.selection"):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
